from typing import Tuple, Set, List, Dict

import torch
from torch import nn

from .controlnet import ControlledUnetModel, ControlNet
from .vae import AutoencoderKL
from .util import GroupNorm32
from .clip import FrozenOpenCLIPEmbedder
from .distributions import DiagonalGaussianDistribution
from ..utils.tilevae import VAEHook


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):
    """A controlled latent diffusion model that combines UNet, VAE, CLIP and ControlNet.

    This model implements a controlled version of Stable Diffusion that allows conditioning
    on both text and image inputs. It uses a ControlNet to process the image condition
    and modulate the main UNet diffusion model.

    Args:
        unet_cfg: Configuration dictionary for the controlled UNet model
        vae_cfg: Configuration dictionary for the VAE model
        clip_cfg: Configuration dictionary for the CLIP text encoder
        controlnet_cfg: Configuration dictionary for the ControlNet
        latent_scale_factor: Scaling factor applied to the latent space

    Attributes:
        unet (ControlledUnetModel): The main controlled UNet diffusion model
        vae (AutoencoderKL): VAE model for encoding/decoding images
        clip (FrozenOpenCLIPEmbedder): CLIP model for text encoding
        controlnet (ControlNet): ControlNet for processing image conditions
        scale_factor (float): Scaling factor for the latent space
        control_scales (List[float]): Scaling factors for each ControlNet layer output
    """

    def __init__(
        self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor
    ):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def load_pretrained_sd(
        self, sd: Dict[str, torch.Tensor]
    ) -> Tuple[Set[str], Set[str]]:
        """Load pretrained state dict into UNet, VAE and CLIP models.

        Args:
            sd: State dictionary containing pretrained weights

        Returns:
            A tuple containing:
                - Set of unused keys from the state dict
                - Set of missing keys that were not found in state dict

        Note:
            After loading, all models are set to eval mode and their parameters
            are frozen by disabling gradients.
        """
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        missing = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                if target_key not in sd:
                    missing.add(target_key)
                    continue
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=False)
        unused = set(sd.keys()) - used
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused, missing

    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        """Load ControlNet weights from a checkpoint state dict.

        Args:
            sd: State dictionary containing ControlNet weights
        """
        self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        """Initialize ControlNet weights from the UNet.

        For matching layers, copies weights from UNet. For layers with different sizes,
        pads with zeros. For unique ControlNet layers, keeps original initialization.

        Returns:
            A tuple containing:
                - Set of keys that were initialized with zero padding
                - Set of keys that kept their original initialization
        """
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch

    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        """Encode images to latent space using the VAE.

        Args:
            image: Input image tensor
            sample: If True, sample from the posterior, otherwise use the mode
            tiled: If True, process image in tiles to reduce memory usage
            tile_size: Size of tiles when tiled=True

        Returns:
            Latent representation of the input image
        """
        if tiled:
            def encoder(x: torch.Tensor) -> DiagonalGaussianDistribution:
                h = VAEHook(
                    self.vae.encoder,
                    tile_size=tile_size,
                    is_decoder=False,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode

        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            z = encoder(image).mode() * self.scale_factor
        return z

    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        """Decode latent representations to images using the VAE.

        Args:
            z: Input latent tensor
            tiled: If True, process in tiles to reduce memory usage
            tile_size: Size of tiles when tiled=True

        Returns:
            Decoded image tensor
        """
        if tiled:
            def decoder(z):
                z = self.vae.post_quant_conv(z)
                dec = VAEHook(
                    self.vae.decoder,
                    tile_size=tile_size,
                    is_decoder=True,
                    fast_decoder=False,
                    fast_encoder=False,
                    color_fix=True,
                )(z)
                return dec
        else:
            decoder = self.vae.decode
        return decoder(z / self.scale_factor)

    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: List[str],
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Prepare text and image conditions for the model.

        Args:
            cond_img: Conditioning image tensor
            txt: List of text prompts
            tiled: If True, use tiled VAE encoding
            tile_size: Size of tiles when tiled=True

        Returns:
            Dictionary containing encoded text and image conditions
        """
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(
                cond_img * 2 - 1,
                sample=False,
                tiled=tiled,
                tile_size=tile_size,
            ),
        )

    def forward(self, x_noisy, t, cond):
        """Forward pass of the model.

        Args:
            x_noisy: Noisy input tensor
            t: Timestep tensor
            cond: Dictionary containing text and image conditions

        Returns:
            Predicted noise tensor
        """
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        control = self.controlnet(x=x_noisy, hint=c_img, timesteps=t, context=c_txt)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(
            x=x_noisy,
            timesteps=t,
            context=c_txt,
            control=control,
            only_mid_control=False,
        )
        return eps

    def cast_dtype(self, dtype: torch.dtype) -> "ControlLDM":
        """Cast model parameters to specified dtype, keeping GroupNorm32 in float32.

        Args:
            dtype: Target dtype for model parameters

        Returns:
            Self for method chaining
        """
        self.unet.dtype = dtype
        self.controlnet.dtype = dtype
        # convert unet blocks to dtype
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.type(dtype)
        # convert controlnet blocks and zero-convs to dtype
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.type(dtype)

        def cast_groupnorm_32(m):
            if isinstance(m, GroupNorm32):
                m.type(torch.float32)

        # GroupNorm32 only works with float32
        for module in [
            self.unet.input_blocks,
            self.unet.middle_block,
            self.unet.output_blocks,
        ]:
            module.apply(cast_groupnorm_32)
        for module in [
            self.controlnet.input_blocks,
            self.controlnet.zero_convs,
            self.controlnet.middle_block,
            self.controlnet.middle_block_out,
        ]:
            module.apply(cast_groupnorm_32)

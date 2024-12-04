"""
This module implements a Variational Autoencoder (VAE) architecture with KL divergence regularization.

The VAE consists of several key components:
- Encoder: Compresses input images into a latent representation
- Decoder: Reconstructs images from the latent space
- Various attention mechanisms for improved feature extraction
- Utility layers like ResNet blocks, up/downsampling, and normalization

The architecture is based on the paper "Autoencoding beyond pixels using a learned similarity metric"
(https://arxiv.org/abs/1512.09300)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from typing import Optional, Any

from .distributions import DiagonalGaussianDistribution
from .config import Config, AttnMode


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Applies the Swish/SiLU activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Tensor with Swish activation applied
    """
    return x * torch.sigmoid(x)


def Normalize(in_channels: int, num_groups: int = 32) -> nn.Module:
    """Creates a GroupNorm normalization layer.
    
    Args:
        in_channels: Number of input channels
        num_groups: Number of groups to separate the channels into
        
    Returns:
        GroupNorm module
    """
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    """Upsamples input tensor by factor of 2 with optional convolution.
    
    Args:
        in_channels: Number of input channels
        with_conv: If True, applies 3x3 convolution after upsampling
    """
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsamples input tensor by factor of 2 with optional convolution.
    
    Args:
        in_channels: Number of input channels
        with_conv: If True, uses strided convolution, otherwise uses average pooling
    """
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """ResNet block with optional time embedding and dropout.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels. If None, same as in_channels
        conv_shortcut: If True, uses 3x3 conv for skip connection, else 1x1
        dropout: Dropout probability
        temb_channels: Number of time embedding channels. If >0, adds time conditioning
    """
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    """Self-attention block using vanilla attention implementation.
    
    Args:
        in_channels: Number of input channels
    """
    def __init__(self, in_channels: int):
        super().__init__()
        print(f"building AttnBlock (vanilla) with {in_channels} in_channels")

        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """Memory efficient attention block using xformers implementation.
    
    This is a single-head self-attention operation optimized for memory usage.
    See: https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    
    Args:
        in_channels: Number of input channels
    """
    def __init__(self, in_channels: int):
        super().__init__()
        print(
            f"building MemoryEfficientAttnBlock (xformers) with {in_channels} in_channels"
        )
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = Config.xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out


class SDPAttnBlock(nn.Module):
    """Attention block using PyTorch's scaled dot product attention.
    
    Args:
        in_channels: Number of input channels
    """
    def __init__(self, in_channels: int):
        super().__init__()
        print(f"building SDPAttnBlock (sdp) with {in_channels} in_channels")
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v)

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out


def make_attn(in_channels: int, attn_type: str = "vanilla", attn_kwargs=None) -> nn.Module:
    """Factory function to create attention blocks of different types.
    
    Args:
        in_channels: Number of input channels
        attn_type: Type of attention ("vanilla", "sdp", "xformers", "linear", "none")
        attn_kwargs: Additional kwargs for attention block
        
    Returns:
        Attention module of specified type
        
    Raises:
        NotImplementedError: If attn_type is not supported
    """
    assert attn_type in [
        "vanilla",
        "sdp", 
        "xformers",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "sdp":
        return SDPAttnBlock(in_channels)
    elif attn_type == "xformers":
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()


class Encoder(nn.Module):
    """Encoder module that compresses input into latent representation.
    
    Args:
        ch: Base channel count
        out_ch: Output channels
        ch_mult: Channel multiplier for each resolution
        num_res_blocks: Number of ResNet blocks per resolution
        attn_resolutions: Resolutions at which to apply attention
        dropout: Dropout probability
        resamp_with_conv: If True, use conv for up/downsampling
        in_channels: Number of input channels
        resolution: Input resolution
        z_channels: Number of channels in latent space
        double_z: If True, doubles latent channels
        use_linear_attn: If True, uses linear attention
        **ignore_kwargs: Additional unused kwargs
    """
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: list,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        use_linear_attn: bool = False,
        **ignore_kwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """Decoder module that reconstructs input from latent representation.
    
    Args:
        ch: Base channel count
        out_ch: Output channels
        ch_mult: Channel multiplier for each resolution
        num_res_blocks: Number of ResNet blocks per resolution
        attn_resolutions: Resolutions at which to apply attention
        dropout: Dropout probability
        resamp_with_conv: If True, use conv for up/downsampling
        in_channels: Number of input channels
        resolution: Input resolution
        z_channels: Number of channels in latent space
        give_pre_end: If True, return features before final conv
        tanh_out: If True, apply tanh to output
        use_linear_attn: If True, uses linear attention
        **ignorekwargs: Additional unused kwargs
    """
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: list,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        use_linear_attn: bool = False,
        **ignorekwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class AutoencoderKL(nn.Module):
    """VAE model with KL-regularized latent space.
    
    Args:
        ddconfig: Dictionary containing VAE architecture configuration
        embed_dim: Dimension of the latent space
    """
    def __init__(self, ddconfig: dict, embed_dim: int):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """Encode input to latent distribution.
        
        Args:
            x: Input tensor
            
        Returns:
            Gaussian distribution over latent space
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed image
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input: torch.Tensor, sample_posterior: bool = True) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
        """Forward pass through VAE.
        
        Args:
            input: Input tensor
            sample_posterior: If True, sample from posterior, else use mode
            
        Returns:
            Tuple of (reconstructed image, posterior distribution)
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

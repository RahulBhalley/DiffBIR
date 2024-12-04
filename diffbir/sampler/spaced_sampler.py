from typing import Optional, Tuple, Dict, Literal

import torch
import numpy as np
from tqdm import tqdm

from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model.cldm import ControlLDM
from ..utils.common import make_tiled_fn, trace_vram_usage


def space_timesteps(num_timesteps: int, section_counts: Union[str, List[int]]) -> Set[int]:
    """Create a list of timesteps to use from an original diffusion process.

    Takes the original number of timesteps and divides them into sections based on the provided
    section counts. Each section will have an equal portion of the original timesteps, strided
    according to the count for that section.

    Args:
        num_timesteps: The total number of timesteps in the original diffusion process.
        section_counts: Either a string containing comma-separated numbers indicating step counts
            per section, or a list of integers. As a special case, can be "ddimN" where N is the
            desired number of steps to use DDIM-style striding.

    Returns:
        A set of integers representing the selected timesteps from the original process.

    Examples:
        >>> space_timesteps(300, [10,15,20])
        # First 100 steps strided to 10 steps
        # Second 100 steps strided to 15 steps 
        # Final 100 steps strided to 20 steps

        >>> space_timesteps(1000, "ddim100")
        # 1000 steps strided to exactly 100 steps using DDIM spacing

    Raises:
        ValueError: If a section cannot be divided into the requested number of steps,
            or if DDIM striding cannot achieve the exact requested step count.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedSampler(Sampler):
    """A sampler that uses spaced timesteps for the diffusion process.

    This sampler implements a diffusion process using non-uniformly spaced timesteps.
    It supports both epsilon and v-parameterizations of the diffusion model.

    Args:
        betas: Array of beta schedule values for the diffusion process.
        parameterization: Either "eps" or "v" indicating the model's parameterization.
        rescale_cfg: Whether to rescale classifier-free guidance.

    Attributes:
        timesteps: Array of selected timesteps for the diffusion process.
        sqrt_alphas_cumprod: Precomputed values for sampling calculations.
        sqrt_one_minus_alphas_cumprod: Precomputed values for sampling calculations.
        sqrt_recip_alphas_cumprod: Precomputed values for sampling calculations.
        sqrt_recipm1_alphas_cumprod: Precomputed values for sampling calculations.
        posterior_variance: Precomputed posterior variance values.
        posterior_log_variance_clipped: Clipped log of posterior variance.
        posterior_mean_coef1: First coefficient for posterior mean calculation.
        posterior_mean_coef2: Second coefficient for posterior mean calculation.
    """

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> None:
        super().__init__(betas, parameterization, rescale_cfg)

    def make_schedule(self, num_steps: int) -> None:
        """Creates a diffusion schedule with the specified number of steps.

        Computes and stores various coefficients needed for the diffusion process
        based on the requested number of steps.

        Args:
            num_steps: Number of diffusion steps to use.
        """
        used_timesteps = space_timesteps(self.num_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.training_alphas_cumprod):
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        self.timesteps = np.array(
            sorted(list(used_timesteps)), dtype=np.int32
        )

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_alphas_cumprod", np.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", np.sqrt(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the mean and variance of the diffusion posterior.

        Args:
            x_start: The initial sample.
            x_t: The noisy sample at timestep t.
            t: The timestep.

        Returns:
            A tuple containing:
                - The posterior mean
                - The posterior variance
        """
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        return mean, variance

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Predicts x_0 from the noise eps using the epsilon parameterization.

        Args:
            x_t: The noisy sample at timestep t.
            t: The timestep.
            eps: The predicted noise.

        Returns:
            The predicted x_0.
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Predicts x_0 from v using the v-parameterization.

        Args:
            x_t: The noisy sample at timestep t.
            t: The timestep.
            v: The predicted v value.

        Returns:
            The predicted x_0.
        """
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def apply_model(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        """Applies the model with classifier-free guidance.

        Args:
            model: The diffusion model.
            x: The input tensor.
            model_t: The timestep tensor for the model.
            cond: The conditional inputs.
            uncond: The unconditional inputs (optional).
            cfg_scale: The classifier-free guidance scale.

        Returns:
            The model output with classifier-free guidance applied.
        """
        if uncond is None or cfg_scale == 1.0:
            model_output = model(x, model_t, cond)
        else:
            model_cond = model(x, model_t, cond)
            model_uncond = model(x, model_t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        return model_output

    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        """Samples from the model at a given timestep.

        Args:
            model: The diffusion model.
            x: The current sample.
            model_t: The timestep tensor for the model.
            t: The current timestep.
            cond: The conditional inputs.
            uncond: The unconditional inputs (optional).
            cfg_scale: The classifier-free guidance scale.

        Returns:
            The next sample in the diffusion process.
        """
        model_output = self.apply_model(model, x, model_t, cond, uncond, cfg_scale)
        if self.parameterization == "eps":
            pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_x0 = self._predict_xstart_from_v(x, t, model_output)
        mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: Optional[torch.Tensor] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """Generates samples using the diffusion model.

        Args:
            model: The diffusion model.
            device: The device to run on.
            steps: Number of diffusion steps.
            x_size: Size of the output tensor.
            cond: The conditional inputs.
            uncond: The unconditional inputs.
            cfg_scale: The classifier-free guidance scale.
            tiled: Whether to use tiled generation.
            tile_size: Size of tiles if using tiled generation.
            tile_stride: Stride between tiles if using tiled generation.
            x_T: Optional starting noise tensor.
            progress: Whether to show progress bar.

        Returns:
            The generated sample.
        """
        self.make_schedule(steps)
        self.to(device)
        if tiled:
            forward = model.forward
            model.forward = make_tiled_fn(
                lambda x_tile, t, cond, hi, hi_end, wi, wi_end: (
                    forward(
                        x_tile,
                        t,
                        {
                            "c_txt": cond["c_txt"],
                            "c_img": cond["c_img"][..., hi:hi_end, wi:wi_end],
                        },
                    )
                ),
                tile_size,
                tile_stride,
            )
        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x = x_T
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, disable=not progress)
        bs = x_size[0]

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            x = self.p_sample(
                model,
                x,
                model_t,
                t,
                cond,
                uncond,
                cur_cfg_scale,
            )

        if tiled:
            model.forward = forward
        return x

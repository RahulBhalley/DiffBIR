"""
Implementation of EDM (Elucidated Diffusion Models) sampler.

This module provides the EDMSampler class which implements various sampling algorithms
for diffusion models based on the EDM framework.
"""

from typing import Literal, Dict, Optional, Callable
import numpy as np
import torch

from .sampler import Sampler
from .k_diffusion import (
    sample_euler,
    sample_euler_ancestral,
    sample_heun,
    sample_dpm_2,
    sample_dpm_2_ancestral,
    sample_lms,
    sample_dpm_fast,
    sample_dpm_adaptive,
    sample_dpmpp_2s_ancestral,
    sample_dpmpp_sde,
    sample_dpmpp_2m,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m_sde,
    append_dims,
)
from ..model.cldm import ControlLDM
from ..utils.common import make_tiled_fn, trace_vram_usage


class EDMSampler(Sampler):
    """
    EDM sampler implementing various sampling algorithms for diffusion models.

    This class provides an implementation of different sampling methods like Euler,
    Heun, DPM-Solver, etc. for generating samples from diffusion models.

    Attributes:
        TYPE_TO_SOLVER (dict): Mapping of solver types to their implementation functions
            and required hyperparameters.
    """

    TYPE_TO_SOLVER = {
        "euler": (sample_euler, ("s_churn", "s_tmin", "s_tmax", "s_noise")),
        "euler_a": (sample_euler_ancestral, ("eta", "s_noise")),
        "heun": (sample_heun, ("s_churn", "s_tmin", "s_tmax", "s_noise")),
        "dpm_2": (sample_dpm_2, ("s_churn", "s_tmin", "s_tmax", "s_noise")),
        "dpm_2_a": (sample_dpm_2_ancestral, ("eta", "s_noise")),
        "lms": (sample_lms, ("order",)),
        # "dpm_fast": (sample_dpm_fast, ())
        "dpm++_2s_a": (sample_dpmpp_2s_ancestral, ("eta", "s_noise")),
        "dpm++_sde": (sample_dpmpp_sde, ("eta", "s_noise")),
        "dpm++_2m": (sample_dpmpp_2m, ()),
        "dpm++_2m_sde": (sample_dpmpp_2m_sde, ("eta", "s_noise")),
        "dpm++_3m_sde": (sample_dpmpp_3m_sde, ("eta", "s_noise")),
    }

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
        solver_type: str,
        s_churn: float,
        s_tmin: float,
        s_tmax: float,
        s_noise: float,
        eta: float,
        order: int,
    ) -> "EDMSampler":
        """
        Initialize the EDM sampler.

        Args:
            betas (np.ndarray): Noise schedule.
            parameterization (Literal["eps", "v"]): Model output parameterization.
            rescale_cfg (bool): Whether to rescale classifier-free guidance.
            solver_type (str): Type of solver to use.
            s_churn (float): Churn parameter for applicable solvers.
            s_tmin (float): Minimum timestep for churn.
            s_tmax (float): Maximum timestep for churn.
            s_noise (float): Noise level for sampling.
            eta (float): Noise parameter for ancestral sampling.
            order (int): Order for LMS solver.

        Returns:
            EDMSampler: Initialized sampler instance.
        """
        super().__init__(betas, parameterization, rescale_cfg)
        solver_type = solver_type[len("edm_") :]
        solver_fn, solver_hparams = self.TYPE_TO_SOLVER[solver_type]
        params = {
            "s_churn": s_churn,
            "s_tmin": s_tmin,
            "s_tmax": s_tmax,
            "s_noise": s_noise,
            "eta": eta,
            "order": order,
        }

        def wrapped_solver_fn(
            model, x, sigmas, extra_args=None, callback=None, disable=None
        ):
            return solver_fn(
                model=model,
                x=x,
                sigmas=sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **{k: params[k] for k in solver_hparams},
            )

        self.solver_fn = wrapped_solver_fn

    def make_schedule(self, steps: int) -> None:
        """
        Create a sampling schedule.

        Args:
            steps (int): Number of sampling steps.
        """
        timesteps = np.linspace(
            len(self.training_alphas_cumprod) - 1, 0, steps, endpoint=False
        ).astype(int)
        alphas_cumprod = self.training_alphas_cumprod[timesteps].copy()
        alphas_cumprod[0] = 1e-8
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        sigmas = np.append(sigmas, 0)
        timesteps = np.append(timesteps, 0)
        self.register("sigmas", sigmas)
        self.register("timesteps", timesteps, torch.long)

    def convert_to_denoiser(
        self,
        model: ControlLDM,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> Callable:
        """
        Convert the model to a denoising function.

        Args:
            model (ControlLDM): The control-guided diffusion model.
            cond (Dict[str, torch.Tensor]): Conditioning inputs.
            uncond (Optional[Dict[str, torch.Tensor]]): Unconditional inputs for CFG.
            cfg_scale (float): Classifier-free guidance scale.

        Returns:
            Callable: Denoising function that takes noisy input and noise level.
        """
        def denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            if self.parameterization == "eps":
                c_skip = torch.ones_like(sigma)
                c_out = -sigma
                c_in = 1 / (sigma**2 + 1.0) ** 0.5
                c_noise = sigma.clone()
            else:
                c_skip = 1.0 / (sigma**2 + 1.0)
                c_out = -sigma / (sigma**2 + 1.0) ** 0.5
                c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
                c_noise = sigma.clone()
            c_noise = self.timesteps[
                (c_noise - self.sigmas[:, None]).abs().argmin(dim=0).view(sigma.shape)
            ]
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, c_noise[0].item())

            c_in, c_out, c_skip = map(
                lambda c: append_dims(c, x.ndim), (c_in, c_out, c_skip)
            )
            if uncond is None or cfg_scale == 1.0:
                model_output = model(x * c_in, c_noise, cond) * c_out + x * c_skip
            else:
                model_cond = model(x * c_in, c_noise, cond) * c_out + x * c_skip
                model_uncond = model(x * c_in, c_noise, uncond) * c_out + x * c_skip
                model_output = model_uncond + cur_cfg_scale * (
                    model_cond - model_uncond
                )
            return model_output

        return denoiser

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: torch.Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using the EDM sampler.

        Args:
            model (ControlLDM): The control-guided diffusion model.
            device (str): Device to run sampling on.
            steps (int): Number of sampling steps.
            x_size (torch.Tuple[int]): Size of the output tensor.
            cond (Dict[str, torch.Tensor]): Conditioning inputs.
            uncond (Dict[str, torch.Tensor]): Unconditional inputs for CFG.
            cfg_scale (float): Classifier-free guidance scale.
            tiled (bool, optional): Whether to use tiled sampling. Defaults to False.
            tile_size (int, optional): Size of tiles for tiled sampling. Defaults to -1.
            tile_stride (int, optional): Stride between tiles. Defaults to -1.
            x_T (torch.Tensor | None, optional): Initial noise. Defaults to None.
            progress (bool, optional): Whether to show progress bar. Defaults to True.

        Returns:
            torch.Tensor: Generated samples.
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

        x = x_T * torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        denoiser = self.convert_to_denoiser(model, cond, uncond, cfg_scale)
        z = self.solver_fn(
            model=denoiser,
            x=x,
            sigmas=self.sigmas,
            extra_args=None,
            callback=None,
            disable=not progress,
        )
        if tiled:
            model.forward = forward
        return z

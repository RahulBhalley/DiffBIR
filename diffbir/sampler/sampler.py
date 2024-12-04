from typing import Literal, overload, Dict, Optional, Tuple
import math
import torch
from torch import nn
import numpy as np

from ..model.cldm import ControlLDM


class Sampler(nn.Module):
    """Base class for diffusion model samplers.

    This class implements core sampling functionality for diffusion models, including
    noise scheduling, classifier-free guidance scaling, and sampling methods.

    Args:
        betas (numpy.ndarray): Array of noise schedule beta values.
        parameterization (Literal["eps", "v"]): Model output parameterization, either
            predicting noise ("eps") or velocity ("v").
        rescale_cfg (bool): Whether to dynamically rescale classifier-free guidance.

    Attributes:
        num_timesteps (int): Number of diffusion timesteps.
        training_betas (numpy.ndarray): Beta schedule used during training.
        training_alphas_cumprod (numpy.ndarray): Cumulative product of (1-beta).
        context (dict): Storage for sampling context.
        parameterization (str): Model output parameterization type.
        rescale_cfg (bool): Whether CFG rescaling is enabled.
    """

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> "Sampler":
        super().__init__()
        self.num_timesteps = len(betas)
        self.training_betas = betas
        self.training_alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.context = {}
        self.parameterization = parameterization
        self.rescale_cfg = rescale_cfg

    def register(
        self, name: str, value: np.ndarray, dtype: torch.dtype = torch.float32
    ) -> None:
        """Register a buffer tensor with the sampler.

        Args:
            name (str): Name of the buffer.
            value (numpy.ndarray): Value to register as buffer.
            dtype (torch.dtype, optional): Data type for buffer. Defaults to torch.float32.
        """
        self.register_buffer(name, torch.tensor(value, dtype=dtype))

    def get_cfg_scale(self, default_cfg_scale: float, model_t: int) -> float:
        """Calculate dynamic classifier-free guidance scale.

        Implements cosine schedule for CFG scaling if rescaling is enabled.

        Args:
            default_cfg_scale (float): Base CFG scale value.
            model_t (int): Current timestep.

        Returns:
            float: Calculated CFG scale for current step.
        """
        if self.rescale_cfg and default_cfg_scale > 1:
            cfg_scale = 1 + default_cfg_scale * (
                (1 - math.cos(math.pi * ((1000 - model_t) / 1000) ** 5.0)) / 2
            )
        else:
            cfg_scale = default_cfg_scale
        return cfg_scale

    @overload
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
    ) -> torch.Tensor: ...

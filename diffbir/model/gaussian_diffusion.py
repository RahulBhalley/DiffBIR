from functools import partial
from typing import Tuple

import torch
from torch import nn
import numpy as np


def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3
) -> np.ndarray:
    """Create a beta schedule for the diffusion process.

    Args:
        schedule: The type of schedule to use. Options are:
            - 'linear': Linear schedule between sqrt(linear_start) and sqrt(linear_end)
            - 'cosine': Cosine schedule with scaling parameter cosine_s
            - 'sqrt_linear': Linear schedule between linear_start and linear_end
            - 'sqrt': Square root of linear schedule
        n_timestep: Number of timesteps in the diffusion process
        linear_start: Starting value for linear schedules
        linear_end: Ending value for linear schedules
        cosine_s: Scaling parameter for cosine schedule

    Returns:
        np.ndarray: Beta values for each timestep

    Raises:
        ValueError: If schedule type is not recognized
    """
    if schedule == "linear":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "sqrt":
        betas = (
            np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64) ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int]
) -> torch.Tensor:
    """Extract values from a tensor at given timesteps and reshape.

    Args:
        a: Source tensor to extract from
        t: Timestep indices to extract
        x_shape: Shape of the target tensor

    Returns:
        torch.Tensor: Extracted and reshaped values
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def enforce_zero_terminal_snr(betas: np.ndarray) -> np.ndarray:
    """Enforce zero terminal signal-to-noise ratio in the diffusion process.

    Implementation based on:
    https://arxiv.org/abs/2305.08891

    Args:
        betas: Original beta schedule

    Returns:
        np.ndarray: Modified beta schedule with zero terminal SNR
    """
    betas = torch.from_numpy(betas)
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas.numpy()


class Diffusion(nn.Module):
    """Diffusion model implementation.

    This class implements the core diffusion process, including forward diffusion (q) 
    and loss computation for reverse diffusion (p).

    Args:
        timesteps: Number of timesteps in the diffusion process
        beta_schedule: Schedule for beta values ('linear', 'cosine', 'sqrt_linear', 'sqrt')
        loss_type: Type of loss function to use ('l1' or 'l2')
        linear_start: Starting value for linear schedules
        linear_end: Ending value for linear schedules
        cosine_s: Scaling parameter for cosine schedule
        parameterization: Type of model output ('eps', 'x0', or 'v')
        zero_snr: Whether to enforce zero terminal SNR

    Attributes:
        num_timesteps (int): Number of diffusion timesteps
        parameterization (str): Type of model parameterization
        zero_snr (bool): Whether zero terminal SNR is enforced
        loss_type (str): Type of loss function
        betas (np.ndarray): Beta schedule values
    """

    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",
        zero_snr=False
    ):
        super().__init__()
        self.num_timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
        assert parameterization in [
            "eps",
            "x0",
            "v",
        ], "currently only supporting 'eps' and 'x0' and 'v'"
        self.parameterization = parameterization
        self.zero_snr = zero_snr
        self.loss_type = loss_type

        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        if zero_snr:
            betas = enforce_zero_terminal_snr(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

        self.betas = betas
        self.register("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def register(self, name: str, value: np.ndarray) -> None:
        """Register a buffer with the given name and value.

        Args:
            name: Name of the buffer
            value: Value to register as a buffer
        """
        self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample from the forward diffusion process q(x_t | x_0).

        Args:
            x_start: Initial sample (x_0)
            t: Timesteps to sample at
            noise: Random noise to add

        Returns:
            torch.Tensor: Diffused samples at timesteps t
        """
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_v(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the velocity prediction target.

        Args:
            x: Input tensor
            noise: Random noise
            t: Timesteps

        Returns:
            torch.Tensor: Velocity prediction target
        """
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred: torch.Tensor, target: torch.Tensor, mean: bool = True) -> torch.Tensor:
        """Compute loss between prediction and target.

        Args:
            pred: Model predictions
            target: Target values
            mean: Whether to return mean loss or per-element loss

        Returns:
            torch.Tensor: Computed loss values

        Raises:
            NotImplementedError: If loss_type is not recognized
        """
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, model: nn.Module, x_start: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute training losses for the reverse diffusion process.

        Args:
            model: The model to train
            x_start: Initial samples
            t: Timesteps
            cond: Conditioning information

        Returns:
            torch.Tensor: Computed loss value
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean()
        return loss_simple

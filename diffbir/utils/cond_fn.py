from typing import overload, Tuple
import torch
from torch.nn import functional as F


class Guidance:
    """Base class for guidance functions used in restoration.

    This class provides the interface and basic functionality for guidance functions
    that help steer the diffusion sampling process towards desired outputs.

    Args:
        scale (float): Gradient scale factor (denoted as `s` in paper). Larger values
            make final result closer to first stage model output.
        t_start (int): Timestep to start applying guidance. Must be > t_stop.
        t_stop (int): Timestep to stop applying guidance. Must be < t_start.
        space (str): Data space for computing loss ('rgb' or 'latent').
        repeat (int): Number of times to repeat guidance application.

    Note:
        The sampling process goes from t=1000 to t=0, so t_start should be larger
        than t_stop.

    Based on the GDP (Generative Diffusion Prior) approach:
    https://github.com/Fayeben/GenerativeDiffusionPrior
    """

    def __init__(
        self, scale: float, t_start: int, t_stop: int, space: str, repeat: int
    ) -> "Guidance":
        self.scale = scale * 3000
        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None
        self.space = space
        self.repeat = repeat

    def load_target(self, target: torch.Tensor) -> None:
        """Load target tensor to guide the restoration towards.

        Args:
            target (torch.Tensor): Target tensor to guide restoration towards.
        """
        self.target = target

    def __call__(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]:
        """Apply guidance to steer sampling.

        Args:
            target_x0 (torch.Tensor): Target tensor to guide towards.
            pred_x0 (torch.Tensor): Current predicted tensor.
            t (int): Current timestep.

        Returns:
            tuple: Contains:
                - torch.Tensor: Gradient to apply
                - float: Loss value
        """
        # avoid propagating gradient out of this scope
        pred_x0 = pred_x0.detach().clone()
        target_x0 = target_x0.detach().clone()
        return self._forward(target_x0, pred_x0, t)

    @overload
    def _forward(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]: ...


class MSEGuidance(Guidance):
    """Mean squared error based guidance.

    Computes guidance gradients based on MSE loss between predicted and target tensors.
    Inherits from base Guidance class.
    """

    def _forward(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]:
        """Compute MSE-based guidance.

        Args:
            target_x0 (torch.Tensor): Target tensor in [-1,1] range, NCHW format.
            pred_x0 (torch.Tensor): Predicted tensor in [-1,1] range, NCHW format.
            t (int): Current timestep.

        Returns:
            tuple: Contains:
                - torch.Tensor: Gradient scaled by guidance strength
                - float: MSE loss value
        """
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3)).sum()
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()


class WeightedMSEGuidance(Guidance):
    """Weighted MSE guidance using edge-aware weights.

    Computes guidance gradients using MSE loss weighted by edge information.
    Areas with stronger edges receive different weights in the loss computation.
    Inherits from base Guidance class.
    """

    def _get_weight(self, target: torch.Tensor) -> torch.Tensor:
        """Compute edge-aware weights for the target image.

        Converts RGB to grayscale, applies Sobel edge detection, and computes
        block-averaged edge magnitudes to create a weight map.

        Args:
            target (torch.Tensor): Target image tensor in [0,1] range.

        Returns:
            torch.Tensor: Weight map where edges receive different weights.
        """
        # convert RGB to G
        rgb_to_gray_kernel = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        target = torch.sum(
            target * rgb_to_gray_kernel.to(target.device), dim=1, keepdim=True
        )
        # initialize sobel kernel in x and y axis
        G_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
        G_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        G_x = torch.tensor(G_x, dtype=target.dtype, device=target.device)[None]
        G_y = torch.tensor(G_y, dtype=target.dtype, device=target.device)[None]
        G = torch.stack((G_x, G_y))

        target = F.pad(target, (1, 1, 1, 1), mode="replicate")  # padding = 1
        grad = F.conv2d(target, G, stride=1)
        mag = grad.pow(2).sum(dim=1, keepdim=True).sqrt()

        n, c, h, w = mag.size()
        block_size = 2
        blocks = (
            mag.view(n, c, h // block_size, block_size, w // block_size, block_size)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
        )
        block_mean = (
            blocks.sum(dim=(-2, -1), keepdim=True)
            .tanh()
            .repeat(1, 1, 1, 1, block_size, block_size)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
        )
        block_mean = block_mean.view(n, c, h, w)
        weight_map = 1 - block_mean

        return weight_map

    def _forward(
        self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, float]:
        """Compute weighted MSE guidance.

        Args:
            target_x0 (torch.Tensor): Target tensor in [-1,1] range, NCHW format.
            pred_x0 (torch.Tensor): Predicted tensor in [-1,1] range, NCHW format.
            t (int): Current timestep.

        Returns:
            tuple: Contains:
                - torch.Tensor: Gradient scaled by guidance strength
                - float: Weighted MSE loss value
        """
        with torch.no_grad():
            w = self._get_weight((target_x0 + 1) / 2)
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = ((pred_x0 - target_x0).pow(2) * w).mean((1, 2, 3)).sum()
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()

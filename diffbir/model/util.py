"""
Utility functions and modules for deep learning models.

This module contains various helper functions and custom PyTorch modules used across
the codebase, particularly for diffusion models. Many functions are adapted from
multiple open source implementations.

Adapted from:
- https://github.com/openai/improved-diffusion
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/openai/guided-diffusion
"""

import os
import math
from inspect import isfunction
import torch
import torch.nn as nn
import numpy as np
from einops import repeat


def exists(val):
    """Check if a value exists (is not None).

    Args:
        val: Any value to check

    Returns:
        bool: True if value is not None, False otherwise
    """
    return val is not None


def default(val, d):
    """Return a default value if the input value doesn't exist.

    Args:
        val: Value to check
        d: Default value or callable that returns default value

    Returns:
        Input value if it exists, otherwise the default value
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def checkpoint(func, inputs, params, flag):
    """Evaluate a function with gradient checkpointing.

    Allows for reduced memory usage during training by not caching intermediate
    activations, at the cost of extra compute in the backward pass.

    Args:
        func: Function to evaluate
        inputs: Sequence of input arguments for the function
        params: Sequence of parameters the function depends on
        flag: If False, disable gradient checkpointing

    Returns:
        Output of the function evaluation
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    """Custom autograd function implementing gradient checkpointing.

    This implementation properly handles parameters that don't require gradients,
    fixing issues with the original implementation when some model parameters
    have requires_grad=False.
    """

    @staticmethod
    def forward(ctx, run_function, length, *args):
        """Forward pass of checkpointing.

        Args:
            ctx: Context object to store information for backward pass
            run_function: Function to run
            length: Number of input tensors
            *args: Input tensors and parameters

        Returns:
            Output tensors from run_function
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        """Backward pass of checkpointing.

        Args:
            ctx: Context object with stored information
            *output_grads: Gradients of the loss with respect to outputs

        Returns:
            Tuple of gradients for all inputs
        """
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + [x for x in ctx.input_params if x.requires_grad],
            output_grads,
            allow_unused=True,
        )
        grads = list(grads)
        input_grads = []
        for tensor in ctx.input_tensors + ctx.input_params:
            if tensor.requires_grad:
                input_grads.append(grads.pop(0))
            else:
                input_grads.append(None)
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + tuple(input_grads)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1-D tensor of N indices, one per batch element
        dim: Dimension of the output embeddings
        max_period: Controls the minimum frequency of the embeddings
        repeat_only: If True, just repeat the timesteps dim times

    Returns:
        torch.Tensor: [N x dim] tensor of positional embeddings
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """Zero out the parameters of a module.

    Args:
        module: PyTorch module

    Returns:
        The module with zeroed parameters
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """Scale the parameters of a module.

    Args:
        module: PyTorch module
        scale: Scaling factor

    Returns:
        The module with scaled parameters
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions.

    Args:
        tensor: Input tensor

    Returns:
        Tensor with mean taken over all non-batch dimensions
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """Create a standard normalization layer.

    Args:
        channels: Number of input channels

    Returns:
        nn.Module: GroupNorm32 normalization module
    """
    return GroupNorm32(32, channels)


class SiLU(nn.Module):
    """SiLU activation function compatible with PyTorch < 1.7."""
    
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 conversion for improved precision."""
    
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module.

    Args:
        dims: Number of dimensions (1, 2, or 3)
        *args: Arguments to pass to convolution constructor
        **kwargs: Keyword arguments to pass to convolution constructor

    Returns:
        nn.Module: Convolution module of requested dimensionality

    Raises:
        ValueError: If dims is not 1, 2, or 3
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """Create a linear module.

    Args:
        *args: Arguments to pass to nn.Linear
        **kwargs: Keyword arguments to pass to nn.Linear

    Returns:
        nn.Linear module
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module.

    Args:
        dims: Number of dimensions (1, 2, or 3)
        *args: Arguments to pass to pooling constructor
        **kwargs: Keyword arguments to pass to pooling constructor

    Returns:
        nn.Module: Average pooling module of requested dimensionality

    Raises:
        ValueError: If dims is not 1, 2, or 3
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

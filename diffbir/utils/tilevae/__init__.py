"""
Tiled VAE module for efficient inference with large images.

This module provides functionality for running VAE inference on large images by
processing them in tiles to reduce memory usage.

Classes:
    VAEHook: A hook class that enables tiled VAE inference by intercepting and
        modifying the forward pass of a VAE model to process images in tiles.
"""

from .tilevae import VAEHook

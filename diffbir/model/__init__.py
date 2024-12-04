"""
DiffBIR Model Package
====================

This package contains the core model components for DiffBIR (Diffusion-Based Image Restoration).

Components
----------
config
    Configuration settings and parameters for the models

Neural Network Models
-------------------
ControlledUnetModel
    Modified UNet model with control signal conditioning
ControlNet
    Network that processes control signals for conditioning
AutoencoderKL
    Variational autoencoder with KL divergence regularization
FrozenOpenCLIPEmbedder
    Pre-trained CLIP text encoder with frozen weights
ControlLDM
    Latent diffusion model with control signal conditioning
Diffusion
    Base diffusion model implementation

Image Restoration Models
----------------------
SwinIR
    Swin Transformer for image restoration
RRDBNet
    Residual-in-Residual Dense Block Network
SCUNet
    U-Net with self-calibrated convolutions
"""

from . import config

from .controlnet import ControlledUnetModel, ControlNet
from .vae import AutoencoderKL
from .clip import FrozenOpenCLIPEmbedder

from .cldm import ControlLDM
from .gaussian_diffusion import Diffusion

from .swinir import SwinIR
from .bsrnet import RRDBNet
from .scunet import SCUNet

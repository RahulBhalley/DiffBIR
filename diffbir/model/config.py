"""Configuration module for attention mechanisms in the diffbir model.

This module handles the configuration and initialization of different attention mechanisms
available in the model. It checks for the availability of optimized attention implementations
like xformers and torch.nn.functional.scaled_dot_product_attention (SDP), and sets up the
appropriate attention mode based on availability and user preferences.

Available attention modes:
    - XFORMERS: Memory-efficient attention implementation using xformers library
    - SDP: Scaled dot product attention from PyTorch 2.0+
    - VANILLA: Standard attention implementation (most memory intensive)

The attention mode can be configured either automatically based on availability,
or manually through the ATTN_MODE environment variable.
"""

import os
from typing import Optional, Literal
from types import ModuleType
import enum
from packaging import version

import torch

# Check for SDP attention availability (PyTorch 2.0+)
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
else:
    SDP_IS_AVAILABLE = False

# Check for xformers availability
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


class AttnMode(enum.Enum):
    """Enumeration of available attention implementation modes.
    
    Attributes:
        SDP (int): Scaled dot product attention from PyTorch 2.0+
        XFORMERS (int): Memory-efficient attention from xformers library
        VANILLA (int): Standard attention implementation
    """
    SDP = 0
    XFORMERS = 1
    VANILLA = 2


class Config:
    """Global configuration class for attention mechanisms.
    
    Attributes:
        xformers (Optional[ModuleType]): Reference to xformers module if available
        attn_mode (AttnMode): Currently selected attention mode
    """
    xformers: Optional[ModuleType] = None
    attn_mode: AttnMode = AttnMode.VANILLA


# Initialize attention mode based on available implementations
if XFORMERS_IS_AVAILBLE:
    Config.attn_mode = AttnMode.XFORMERS
    print(f"use xformers attention as default")
elif SDP_IS_AVAILABLE:
    Config.attn_mode = AttnMode.SDP
    print(f"use sdp attention as default")
else:
    print(f"both sdp attention and xformers are not available, use vanilla attention (very expensive) as default")

if XFORMERS_IS_AVAILBLE:
    Config.xformers = xformers


# Allow manual override of attention mode through environment variable
ATTN_MODE = os.environ.get("ATTN_MODE", None)
if ATTN_MODE is not None:
    assert ATTN_MODE in ["vanilla", "sdp", "xformers"], "Invalid attention mode specified"
    if ATTN_MODE == "sdp":
        assert SDP_IS_AVAILABLE, "SDP attention requires PyTorch 2.0+"
        Config.attn_mode = AttnMode.SDP
    elif ATTN_MODE == "xformers":
        assert XFORMERS_IS_AVAILBLE, "xformers package is not available"
        Config.attn_mode = AttnMode.XFORMERS
    else:
        Config.attn_mode = AttnMode.VANILLA
    print(f"set attention mode to {ATTN_MODE}")
else:
    print("keep default attention mode")

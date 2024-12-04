"""OpenCLIP model and tokenizer for DiffBIR.

This module provides the core CLIP model and tokenization functionality used by DiffBIR
for text-image understanding and generation.

The CLIP model enables:
- Joint text-image embeddings
- Zero-shot image classification
- Text-guided image generation

The tokenizer handles:
- Text preprocessing for CLIP
- Converting text to token sequences
- Special token handling

Classes
-------
CLIP
    Contrastive Language-Image Pre-training model that learns joint embeddings.

Functions
---------
tokenize(text)
    Convert text strings into token sequences for CLIP model input.
"""

from .model import CLIP
from .tokenizer import tokenize

__all__ = ["CLIP", "tokenize"]

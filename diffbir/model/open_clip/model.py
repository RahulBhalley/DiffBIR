"""CLIP Model for joint vision-language understanding.

This module implements the CLIP (Contrastive Language-Image Pre-training) model architecture.
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.

The model consists of:
- A vision transformer for encoding images
- A text transformer for encoding text
- A projection layer to align both embeddings in the same space
- A learned temperature parameter for the contrastive loss

The model can be used to:
- Extract image features
- Extract text features 
- Compare image and text embeddings
- Zero-shot image classification
- Text-guided image generation

Classes
-------
CLIPVisionCfg
    Configuration dataclass for the vision transformer.
    
CLIPTextCfg
    Configuration dataclass for the text transformer.
    
CLIP
    Main model class implementing the CLIP architecture.

Functions
---------
get_cast_dtype(precision)
    Get torch dtype based on precision string.

_build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
    Build the vision transformer component.
    
_build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
    Build the text transformer component.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import LayerNormFp32, LayerNorm, QuickGELU, VisionTransformer, TextTransformer


@dataclass
class CLIPVisionCfg:
    """Configuration for the vision transformer component of CLIP.
    
    Parameters
    ----------
    layers : Union[Tuple[int, int, int, int], int], default=12
        Number of transformer layers.
    width : int, default=768
        Width of transformer layers.
    head_width : int, default=64
        Width of attention heads.
    mlp_ratio : float, default=4.0
        Ratio of MLP hidden dim to embedding dim.
    patch_size : int, default=16
        Size of image patches.
    image_size : Union[Tuple[int, int], int], default=224
        Input image size.
    ls_init_value : Optional[float], default=None
        Layer scale initial value.
    patch_dropout : float, default=0.
        Dropout rate for patches.
    input_patchnorm : bool, default=False
        Whether to use patch normalization.
    global_average_pool : bool, default=False
        Whether to use global average pooling.
    attentional_pool : bool, default=False
        Whether to use attentional pooling.
    n_queries : int, default=256
        Number of queries for attentional pooler.
    attn_pooler_heads : int, default=8
        Number of heads for attentional pooling.
    output_tokens : bool, default=False
        Whether to output tokens.
    timm_model_name : str, default=None
        Name of timm model to use.
    timm_model_pretrained : bool, default=False
        Whether to use pretrained timm weights.
    timm_pool : str, default='avg'
        Pooling type for timm model.
    timm_proj : str, default='linear'
        Projection type for timm model.
    timm_proj_bias : bool, default=False
        Whether to use bias in projection.
    timm_drop : float, default=0.
        Dropout rate.
    timm_drop_path : Optional[float], default=None
        Drop path rate.
    """
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.
    input_patchnorm: bool = False
    global_average_pool: bool = False
    attentional_pool: bool = False
    n_queries: int = 256
    attn_pooler_heads: int = 8
    output_tokens: bool = False

    timm_model_name: str = None
    timm_model_pretrained: bool = False
    timm_pool: str = 'avg'
    timm_proj: str = 'linear'
    timm_proj_bias: bool = False
    timm_drop: float = 0.
    timm_drop_path: Optional[float] = None


@dataclass
class CLIPTextCfg:
    """Configuration for the text transformer component of CLIP.
    
    Parameters
    ----------
    context_length : int, default=77
        Maximum sequence length.
    vocab_size : int, default=49408
        Size of vocabulary.
    width : int, default=512
        Width of transformer layers.
    heads : int, default=8
        Number of attention heads.
    layers : int, default=12
        Number of transformer layers.
    ls_init_value : Optional[float], default=None
        Layer scale initial value.
    hf_model_name : str, default=None
        HuggingFace model name.
    hf_tokenizer_name : str, default=None
        HuggingFace tokenizer name.
    hf_model_pretrained : bool, default=True
        Whether to use pretrained HF model.
    proj : str, default='mlp'
        Type of projection layer.
    pooler_type : str, default='mean_pooler'
        Type of pooling.
    embed_cls : bool, default=False
        Whether to embed CLS token.
    pad_id : int, default=0
        Padding token ID.
    output_tokens : bool, default=False
        Whether to output tokens.
    """
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


def get_cast_dtype(precision: str):
    """Get torch dtype based on precision string.
    
    Parameters
    ----------
    precision : str
        Precision identifier ('bf16' or 'fp16').
        
    Returns
    -------
    torch.dtype or None
        Corresponding torch dtype.
    """
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    """Build the vision transformer component.
    
    Parameters
    ----------
    embed_dim : int
        Dimension of final embedding.
    vision_cfg : CLIPVisionCfg
        Vision model configuration.
    quick_gelu : bool, default=False
        Whether to use QuickGELU activation.
    cast_dtype : Optional[torch.dtype], default=None
        Data type for casting.
        
    Returns
    -------
    VisionTransformer
        Vision transformer model.
    """
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        input_patchnorm=vision_cfg.input_patchnorm,
        global_average_pool=vision_cfg.global_average_pool,
        attentional_pool=vision_cfg.attentional_pool,
        n_queries=vision_cfg.n_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    """Build the text transformer component.
    
    Parameters
    ----------
    embed_dim : int
        Dimension of final embedding.
    text_cfg : CLIPTextCfg
        Text model configuration.
    quick_gelu : bool, default=False
        Whether to use QuickGELU activation.
    cast_dtype : Optional[torch.dtype], default=None
        Data type for casting.
        
    Returns
    -------
    TextTransformer
        Text transformer model.
    """
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text


class CLIP(nn.Module):
    """CLIP (Contrastive Language-Image Pre-training) model.
    
    This implements the CLIP model that learns joint representations of images and text.
    
    Parameters
    ----------
    embed_dim : int
        Dimension of the joint embedding space.
    vision_cfg : CLIPVisionCfg
        Configuration for vision transformer.
    text_cfg : CLIPTextCfg
        Configuration for text transformer.
    quick_gelu : bool, default=False
        Whether to use QuickGELU activation.
    cast_dtype : Optional[torch.dtype], default=None
        Data type for casting operations.
    output_dict : bool, default=False
        Whether to return outputs as dictionary.
        
    Attributes
    ----------
    visual : VisionTransformer
        Vision encoder.
    transformer : nn.Module
        Text encoder transformer.
    token_embedding : nn.Embedding
        Text token embeddings.
    positional_embedding : nn.Parameter
        Positional embeddings for text.
    ln_final : LayerNorm
        Final layer normalization.
    text_projection : nn.Parameter
        Text projection matrix.
    logit_scale : nn.Parameter
        Learned temperature parameter.
    """
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        """Lock the image tower parameters for fine-tuning.
        
        Parameters
        ----------
        unlocked_groups : int, default=0
            Number of layer groups to leave unlocked.
        freeze_bn_stats : bool, default=False
            Whether to freeze batch norm statistics.
        """
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing.
        
        Parameters
        ----------
        enable : bool, default=True
            Whether to enable gradient checkpointing.
        """
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        """Encode images into the joint embedding space.
        
        Parameters
        ----------
        image : torch.Tensor
            Batch of images.
        normalize : bool, default=False
            Whether to normalize the output features.
            
        Returns
        -------
        torch.Tensor
            Image features.
        """
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        """Encode text into the joint embedding space.
        
        Parameters
        ----------
        text : torch.Tensor
            Batch of tokenized text.
        normalize : bool, default=False
            Whether to normalize the output features.
            
        Returns
        -------
        torch.Tensor
            Text features.
        """
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        """Forward pass computing image and text features.
        
        Parameters
        ----------
        image : Optional[torch.Tensor], default=None
            Batch of images.
        text : Optional[torch.Tensor], default=None
            Batch of tokenized text.
            
        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]
            Image features, text features and logit scale, either as tuple or dict.
        """
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()

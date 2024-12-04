"""
Transformer modules and utilities for OpenCLIP model.

This module contains various transformer components including layer normalization,
attention mechanisms, and other building blocks used in the OpenCLIP architecture.
"""

import collections
from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple
from itertools import repeat

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def _ntuple(n):
    """
    Creates a function that converts a single value or iterable into an n-tuple.
    
    Args:
        n (int): The length of the target tuple
        
    Returns:
        callable: A function that converts input into an n-tuple
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class LayerNormFp32(nn.LayerNorm):
    """
    Layer normalization module that casts input to float32 for better numerical stability.
    
    Inherits from torch.nn.LayerNorm but performs computation in float32 precision
    regardless of input dtype.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """
    Layer normalization that preserves input dtype.
    
    Similar to standard LayerNorm but ensures output dtype matches input dtype.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    """
    Fast approximate version of GELU activation function.
    
    Uses sigmoid approximation instead of error function.
    Note: This implementation is slower than nn.GELU and uses more GPU memory.
    """
    
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    """
    Learnable scaling for network layers.
    
    Args:
        dim (int): Number of input channels
        init_values (float, optional): Initial value for scaling. Defaults to 1e-5
        inplace (bool, optional): Whether to perform scaling in-place. Defaults to False
    """
    
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    Randomly drops patches from input sequences during training.
    
    Implementation based on https://arxiv.org/abs/2212.00794
    
    Args:
        prob (float): Probability of dropping each patch
        exclude_first_token (bool, optional): Whether to exclude CLS token from dropout. Defaults to True
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    """
    Multi-head attention module with optional scaled cosine attention.
    
    Args:
        dim (int): Total dimension of the model
        num_heads (int, optional): Number of attention heads. Defaults to 8
        qkv_bias (bool, optional): If True, adds bias to query, key, value projections. Defaults to True
        scaled_cosine (bool, optional): If True, uses scaled cosine attention. Defaults to False
        scale_heads (bool, optional): If True, adds learnable scaling per head. Defaults to False
        logit_scale_max (float, optional): Maximum logit scale for scaled cosine attention. Defaults to log(1/0.01)
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0
        proj_drop (float, optional): Dropout rate for projection outputs. Defaults to 0.0
    """

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    """
    Learnable attention pooling layer.
    
    Args:
        d_model (int): Model dimension for queries
        context_dim (int): Dimension of context (keys/values)
        n_head (int, optional): Number of attention heads. Defaults to 8
        n_queries (int, optional): Number of learnable queries. Defaults to 256
        norm_layer (callable, optional): Normalization layer. Defaults to LayerNorm
    """

    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class ResidualAttentionBlock(nn.Module):
    """Residual attention block with optional cross-attention.

    This module implements a residual attention block that can be used for both self-attention
    and cross-attention. It consists of:
    1. Layer normalization + multi-head attention
    2. Layer normalization + MLP
    Both branches have residual connections and optional layer scaling.

    Parameters
    ----------
    d_model : int
        Dimension of the model (embedding dimension)
    n_head : int 
        Number of attention heads
    mlp_ratio : float, optional
        Ratio of MLP hidden dimension to embedding dimension, by default 4.0
    ls_init_value : float, optional
        Initial value for layer scale. If None, layer scale is disabled
    act_layer : callable, optional
        Activation layer constructor, by default nn.GELU
    norm_layer : callable, optional
        Normalization layer constructor, by default LayerNorm
    is_cross_attention : bool, optional
        Whether to use cross-attention, by default False

    Attributes
    ----------
    ln_1 : nn.Module
        First layer normalization
    attn : nn.MultiheadAttention
        Multi-head attention module
    ls_1 : nn.Module
        First layer scale or identity
    ln_1_kv : nn.Module, optional
        Layer norm for key/value in cross-attention
    ln_2 : nn.Module
        Second layer normalization
    mlp : nn.Sequential
        MLP block with activation
    ls_2 : nn.Module
        Second layer scale or identity
    """

    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention mechanism.

        Parameters
        ----------
        q_x : torch.Tensor
            Query tensor
        k_x : torch.Tensor, optional
            Key tensor, defaults to query if None
        v_x : torch.Tensor, optional
            Value tensor, defaults to query if None
        attn_mask : torch.Tensor, optional
            Attention mask tensor

        Returns
        -------
        torch.Tensor
            Output tensor after attention
        """
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        q_x : torch.Tensor
            Query tensor
        k_x : torch.Tensor, optional
            Key tensor for cross-attention
        v_x : torch.Tensor, optional
            Value tensor for cross-attention
        attn_mask : torch.Tensor, optional
            Attention mask tensor

        Returns
        -------
        torch.Tensor
            Output tensor after residual attention block
        """
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    """Custom residual attention block with additional features.

    This module extends the base ResidualAttentionBlock with additional scaling options
    and normalization layers. It supports:
    - Cosine attention scaling
    - Head scaling
    - Attention scaling
    - FC layer scaling

    Parameters
    ----------
    d_model : int
        Dimension of the model
    n_head : int
        Number of attention heads
    mlp_ratio : float, optional
        MLP hidden dimension ratio, by default 4.0
    ls_init_value : float, optional
        Layer scale initial value, by default None
    act_layer : callable, optional
        Activation layer constructor, by default nn.GELU
    norm_layer : callable, optional
        Normalization layer constructor, by default LayerNorm
    scale_cosine_attn : bool, optional
        Whether to scale cosine attention, by default False
    scale_heads : bool, optional
        Whether to scale attention heads, by default False
    scale_attn : bool, optional
        Whether to add normalization after attention, by default False
    scale_fc : bool, optional
        Whether to add normalization in MLP, by default False
    """

    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model, n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        attn_mask : torch.Tensor, optional
            Attention mask tensor

        Returns
        -------
        torch.Tensor
            Output tensor after custom residual attention block
        """
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    """Transformer encoder stack.

    A stack of transformer encoder layers with support for gradient checkpointing.

    Parameters
    ----------
    width : int
        Model dimension
    layers : int
        Number of transformer layers
    heads : int
        Number of attention heads
    mlp_ratio : float, optional
        MLP hidden dimension ratio, by default 4.0
    ls_init_value : float, optional
        Layer scale initial value, by default None
    act_layer : callable, optional
        Activation layer constructor, by default nn.GELU
    norm_layer : callable, optional
        Normalization layer constructor, by default LayerNorm

    Attributes
    ----------
    width : int
        Model dimension
    layers : int
        Number of layers
    grad_checkpointing : bool
        Whether gradient checkpointing is enabled
    resblocks : nn.ModuleList
        List of transformer encoder layers
    """

    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        """Get the dtype used for casting.

        Returns
        -------
        torch.dtype
            Dtype used for casting weights
        """
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        attn_mask : torch.Tensor, optional
            Attention mask tensor

        Returns
        -------
        torch.Tensor
            Output tensor after transformer encoder stack
        """
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image encoding.

    A PyTorch implementation of the Vision Transformer model that processes images through 
    patch embedding, positional encoding, and transformer layers to produce image features.

    Parameters
    ----------
    image_size : int
        Size of input image (assumed square). For rectangular images, will be converted to (image_size, image_size)
    patch_size : int 
        Size of patches to divide image into. Determines the sequence length
    width : int
        Dimension of transformer embeddings
    layers : int
        Number of transformer layers
    heads : int
        Number of attention heads per transformer layer
    mlp_ratio : float
        Ratio of mlp hidden dim to embedding dim
    ls_init_value : float, optional
        Layer scale initial value. If None, layer scaling is disabled
    global_average_pool : bool, default=False
        If True, average pool all tokens, otherwise use [CLS] token
    attentional_pool : bool, default=False
        If True, use attention pooling instead of [CLS] token or average pooling
    n_queries : int, default=256
        Number of learnable queries for attention pooling
    attn_pooler_heads : int, default=8
        Number of heads in attention pooler
    output_dim : int, default=512
        Dimension of output features
    patch_dropout : float, default=0.
        Probability of dropping patches during training
    input_patchnorm : bool, default=False
        If True, normalize patch embeddings
    act_layer : callable, default=nn.GELU
        Activation layer constructor
    norm_layer : callable, default=LayerNorm 
        Normalization layer constructor
    output_tokens : bool, default=False
        If True, return intermediate tokens in addition to pooled output

    Attributes
    ----------
    output_tokens : bool
        Whether to output intermediate tokens
    image_size : tuple
        Height and width of input image
    patch_size : tuple
        Height and width of each patch
    grid_size : tuple
        Number of patches in height and width dimensions
    output_dim : int
        Dimension of output features
    """
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            attentional_pool: bool = False,
            n_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            input_patchnorm: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False
    ):
        super().__init__()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """Lock (freeze) all model weights except for the final unlocked_groups.
        
        Parameters
        ----------
        unlocked_groups : int, default=0
            Number of groups from the end to leave unlocked
        freeze_bn_stats : bool, default=False
            If True, freeze batch norm statistics
        """
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        """Initialize model parameters.
        
        Currently a no-op, but kept for future initialization schemes.
        """
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing.

        Parameters
        ----------
        enable : bool, default=True
            If True, enables gradient checkpointing
        """
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool features globally.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, sequence_length, hidden_dim)

        Returns
        -------
        tuple
            Pooled features and remaining tokens
        """
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor):
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor or tuple
            If output_tokens is False, returns pooled features of shape (batch_size, output_dim)
            If output_tokens is True, returns tuple of pooled features and intermediate tokens
        """
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled


class TextTransformer(nn.Module):
    """Text transformer encoder for CLIP model.

    This module implements a transformer encoder for processing text sequences. It includes token 
    and positional embeddings, multiple transformer layers, and optional CLS token handling.

    Parameters
    ----------
    context_length : int, default=77
        Maximum sequence length for input text
    vocab_size : int, default=49408
        Size of token vocabulary
    width : int, default=512
        Dimension of transformer embeddings
    heads : int, default=8
        Number of attention heads per transformer layer
    layers : int, default=12
        Number of transformer layers
    ls_init_value : float, optional
        Layer scale initial value. If None, layer scaling is disabled
    output_dim : int, default=512
        Dimension of output features
    act_layer : callable, default=nn.GELU
        Activation layer constructor
    norm_layer : callable, default=LayerNorm
        Normalization layer constructor
    embed_cls : bool, default=False
        If True, add learnable CLS token embedding
    pad_id : int, default=0
        Token ID used for padding
    output_tokens : bool, default=False
        If True, return intermediate tokens in addition to pooled output

    Attributes
    ----------
    output_tokens : bool
        Whether to output intermediate tokens
    context_length : int
        Maximum sequence length
    vocab_size : int
        Size of token vocabulary
    width : int
        Dimension of transformer embeddings
    output_dim : int
        Dimension of output features
    heads : int
        Number of attention heads
    pad_id : int
        Padding token ID
    text_projection : nn.Parameter
        Projection matrix for final output
    cls_emb : nn.Parameter or None
        CLS token embedding if embed_cls=True
    token_embedding : nn.Embedding
        Token embedding layer
    positional_embedding : nn.Parameter
        Positional embedding weights
    transformer : Transformer
        Main transformer encoder
    ln_final : nn.Module
        Final layer normalization
    attn_mask : torch.Tensor
        Causal attention mask
    """

    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            embed_cls: bool = False,
            pad_id: int = 0,
            output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        """Initialize model parameters.

        Initializes embeddings and transformer weights using normal distributions with 
        carefully chosen standard deviations based on model dimensions.
        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing.

        Parameters
        ----------
        enable : bool, default=True
            If True, enables gradient checkpointing for memory efficiency
        """
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        """Build causal attention mask.

        Returns
        -------
        torch.Tensor
            Causal attention mask of shape (num_pos, num_pos) with -inf in upper triangle
        """
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        """Build attention mask for CLS token.

        Parameters
        ----------
        text : torch.Tensor
            Input text tensor
        cast_dtype : torch.dtype
            Data type for mask tensor

        Returns
        -------
        torch.Tensor
            Attention mask for CLS token
        """
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        """Repeat tensor along batch dimension.

        Parameters
        ----------
        t : torch.Tensor
            Input tensor
        N : int
            Number of repeats

        Returns
        -------
        torch.Tensor
            Repeated tensor
        """
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        """Forward pass through text transformer.

        Parameters
        ----------
        text : torch.Tensor
            Input text tokens of shape (batch_size, seq_len)

        Returns
        -------
        torch.Tensor or tuple
            If output_tokens is False, returns pooled features of shape (batch_size, output_dim)
            If output_tokens is True, returns tuple of pooled features and intermediate tokens
        """
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    """A transformer model that processes both text and image inputs through cross-attention.
    
    This transformer extends the base Transformer class by adding cross-attention layers
    that allow text features to attend to image features. It processes text embeddings
    through self-attention blocks followed by cross-attention with image embeddings.

    Parameters
    ----------
    width : int
        Hidden dimension size/width of the transformer.
    layers : int 
        Number of transformer layers.
    heads : int
        Number of attention heads.
    context_length : int, optional
        Maximum sequence length for text inputs, by default 77.
    mlp_ratio : float, optional
        Ratio of MLP hidden dimension to transformer width, by default 4.0.
    ls_init_value : float, optional
        Layer scale initial value. If None, layer scaling is disabled.
    act_layer : callable, optional
        Activation layer constructor, by default nn.GELU.
    norm_layer : callable, optional
        Normalization layer constructor, by default LayerNorm.
    output_dim : int, optional
        Output projection dimension, by default 512.
    """

    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        """Initialize transformer parameters.

        Initializes the weights of self-attention, cross-attention, and MLP layers
        using normal distributions with carefully chosen standard deviations based
        on the transformer's width and number of layers.
        """
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """Build causal attention mask.

        Creates a causal (triangular) attention mask that prevents tokens from
        attending to future tokens in the sequence.

        Returns
        -------
        torch.Tensor
            Attention mask of shape (context_length, context_length) filled with
            -inf in the upper triangle.
        """
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        """Forward pass through the multimodal transformer.

        Processes text embeddings through alternating self-attention and
        cross-attention with image embeddings.

        Parameters
        ----------
        image_embs : torch.Tensor
            Image embeddings of shape (batch_size, seq_len, width)
        text_embs : torch.Tensor
            Text embeddings of shape (batch_size, seq_len, width)

        Returns
        -------
        torch.Tensor
            Processed text embeddings of shape (batch_size, seq_len, output_dim)
        """
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing.

        Parameters
        ----------
        enable : bool, optional
            If True, enables gradient checkpointing for memory efficiency,
            by default True
        """
        self.grad_checkpointing = enable

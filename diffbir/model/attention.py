"""
This module implements various attention mechanisms and related components for transformer architectures.

The module provides different attention implementations optimized for various use cases:
- Standard cross attention
- Memory efficient cross attention using xformers
- Scaled dot product attention
"""

from packaging import version
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from .util import checkpoint, zero_module, exists, default
from .config import Config, AttnMode


# CrossAttn precision handling
import os

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


class GEGLU(nn.Module):
    """Gated Gaussian Error Linear Unit activation function.
    
    This implements a gated variant of GELU activation for improved performance.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after applying gated GELU activation
        """
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Standard feed-forward network with optional GLU activation.

    Args:
        dim (int): Input dimension
        dim_out (int, optional): Output dimension. Defaults to input dimension
        mult (int, optional): Multiplier for inner dimension. Defaults to 4
        glu (bool, optional): Whether to use GLU activation. Defaults to False
        dropout (float, optional): Dropout probability. Defaults to 0.0
    """

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after feed-forward transformation
        """
        return self.net(x)


def Normalize(in_channels):
    """Creates a GroupNorm normalization layer.

    Args:
        in_channels (int): Number of input channels

    Returns:
        nn.GroupNorm: Normalization layer
    """
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class CrossAttention(nn.Module):
    """Standard cross-attention mechanism.

    Implements the traditional transformer attention mechanism that can attend
    between a query and a context sequence.

    Args:
        query_dim (int): Dimension of query tensor
        context_dim (int, optional): Dimension of context tensor. Defaults to query_dim
        heads (int, optional): Number of attention heads. Defaults to 8
        dim_head (int, optional): Dimension of each attention head. Defaults to 64
        dropout (float, optional): Dropout probability. Defaults to 0.0
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (vanilla). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        Args:
            x (torch.Tensor): Query tensor
            context (torch.Tensor, optional): Context tensor. Defaults to query tensor
            mask (torch.Tensor, optional): Attention mask. Defaults to None

        Returns:
            torch.Tensor: Output after applying attention
        """
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(
                enabled=False,
                device_type="cuda" if str(x.device).startswith("cuda") else "cpu",
            ):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        sim = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    """Memory efficient cross-attention using xformers.

    Implements an optimized version of attention that uses less memory through
    the xformers library.

    Args:
        query_dim (int): Dimension of query tensor
        context_dim (int, optional): Dimension of context tensor. Defaults to query_dim
        heads (int, optional): Number of attention heads. Defaults to 8
        dim_head (int, optional): Dimension of each attention head. Defaults to 64
        dropout (float, optional): Dropout probability. Defaults to 0.0
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (xformers). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        """
        Args:
            x (torch.Tensor): Query tensor
            context (torch.Tensor, optional): Context tensor. Defaults to query tensor
            mask (torch.Tensor, optional): Attention mask. Defaults to None

        Returns:
            torch.Tensor: Output after applying attention

        Raises:
            NotImplementedError: If mask is provided (not yet supported)
        """
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = Config.xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class SDPCrossAttention(nn.Module):
    """Scaled dot-product cross-attention.

    Implements cross-attention using PyTorch's native scaled dot-product attention.

    Args:
        query_dim (int): Dimension of query tensor
        context_dim (int, optional): Dimension of context tensor. Defaults to query_dim
        heads (int, optional): Number of attention heads. Defaults to 8
        dim_head (int, optional): Dimension of each attention head. Defaults to 64
        dropout (float, optional): Dropout probability. Defaults to 0.0
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__} (sdp). Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        Args:
            x (torch.Tensor): Query tensor
            context (torch.Tensor, optional): Context tensor. Defaults to query tensor
            mask (torch.Tensor, optional): Attention mask. Defaults to None

        Returns:
            torch.Tensor: Output after applying attention

        Raises:
            NotImplementedError: If mask is provided (not yet supported)
        """
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = F.scaled_dot_product_attention(q, k, v)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """Basic transformer block with self-attention and cross-attention.

    This implements a standard transformer block containing self-attention, cross-attention,
    and feed-forward layers with residual connections and layer normalization.

    Parameters
    ----------
    dim : int
        Input dimension/channels
    n_heads : int 
        Number of attention heads
    d_head : int
        Dimension of each attention head
    dropout : float, optional
        Dropout probability, by default 0.0
    context_dim : int, optional
        Context dimension for cross-attention. If None, cross-attention becomes self-attention
    gated_ff : bool, optional
        Whether to use gated feed-forward network, by default True
    checkpoint : bool, optional
        Whether to use gradient checkpointing, by default True
    disable_self_attn : bool, optional
        Whether to disable self-attention and only use cross-attention, by default False

    Attributes
    ----------
    ATTENTION_MODES : dict
        Maps attention mode enums to attention layer classes
    attn1 : nn.Module
        First attention layer (self-attention or cross-attention)
    attn2 : nn.Module 
        Second attention layer (cross-attention)
    ff : FeedForward
        Feed-forward network
    norm1 : nn.LayerNorm
        Layer norm before first attention
    norm2 : nn.LayerNorm
        Layer norm before second attention
    norm3 : nn.LayerNorm
        Layer norm before feed-forward
    """

    ATTENTION_MODES = {
        AttnMode.VANILLA: CrossAttention,  # vanilla attention
        AttnMode.XFORMERS: MemoryEfficientCrossAttention,
        AttnMode.SDP: SDPCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        attn_cls = self.ATTENTION_MODES[Config.attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        """Forward pass through transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, dim)
        context : torch.Tensor, optional
            Context tensor for cross-attention, by default None

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, dim)
        """
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = (
            self.attn1(
                self.norm1(x), context=context if self.disable_self_attn else None
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """Transformer for processing image-like data with spatial dimensions.

    This transformer first projects the input and reshapes it to sequence form,
    applies standard transformer layers, then reshapes back to image form.
    Supports both convolutional and linear projections for efficiency.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    n_heads : int
        Number of attention heads
    d_head : int
        Dimension of each attention head
    depth : int, optional
        Number of transformer blocks, by default 1
    dropout : float, optional
        Dropout probability, by default 0.0
    context_dim : int or list of int, optional
        Context dimensions for cross-attention. If list, should match depth
    disable_self_attn : bool, optional
        Whether to disable self-attention, by default False
    use_linear : bool, optional
        Whether to use linear projections instead of 1x1 convs, by default False
    use_checkpoint : bool, optional
        Whether to use gradient checkpointing, by default True

    Attributes
    ----------
    norm : nn.Module
        Input normalization layer
    proj_in : nn.Module
        Input projection (conv or linear)
    transformer_blocks : nn.ModuleList
        List of transformer blocks
    proj_out : nn.Module
        Output projection (conv or linear)
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        """Forward pass through spatial transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
        context : torch.Tensor or list of torch.Tensor, optional
            Context tensor(s) for cross-attention. If list, should match transformer depth

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, channels, height, width)
        """
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

"""
Attention implementations for Tiled VAE.

This module provides different attention implementations that can be used with Tiled VAE.
The implementations are modified from sd_hijack_optimizations.py to remove residual connections
and normalization, allowing Tiled VAE to support different types of attention mechanisms.

Functions:
    get_attn_func(): Returns the appropriate attention function based on config
    forward(x): Basic attention implementation using matrix multiplication
    xformers_forward(x): Memory-efficient attention using xformers library
    sdp_forward(x): Attention using PyTorch's scaled dot product attention
"""
import torch
from torch.nn import functional as F
from einops import rearrange

from ...model.config import Config, AttnMode


def get_attn_func():
    """Returns the configured attention function.
    
    Returns:
        callable: One of forward(), xformers_forward(), or sdp_forward() based on Config.attn_mode
    """
    return {
        AttnMode.VANILLA: forward,
        AttnMode.XFORMERS: xformers_forward,
        AttnMode.SDP: sdp_forward,
    }[Config.attn_mode]


def forward(self, x):
    """Basic attention implementation using matrix multiplication.
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W]
        
    Returns:
        torch.Tensor: Attention output of shape [B, C, H, W]
    """
    h_ = x
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    q = q.reshape(b, c, h * w)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(b, c, h * w)  # b,c,hw
    w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c) ** (-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b, c, h * w)
    w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
    h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)
    return h_


def xformers_forward(self, x):
    """Memory-efficient attention implementation using xformers library.
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W]
        
    Returns:
        torch.Tensor: Attention output of shape [B, C, H, W]
    """
    h_ = x
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(B, t.shape[1], 1, C)
        .permute(0, 2, 1, 3)
        .reshape(B * 1, t.shape[1], C)
        .contiguous(),
        (q, k, v),
    )
    out = Config.xformers.ops.memory_efficient_attention(
        q, k, v, attn_bias=None, op=self.attention_op
    )

    out = (
        out.unsqueeze(0)
        .reshape(B, 1, out.shape[1], C)
        .permute(0, 2, 1, 3)
        .reshape(B, out.shape[1], C)
    )
    out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
    out = self.proj_out(out)
    return out


def sdp_forward(self, x):
    """Attention implementation using PyTorch's scaled dot product attention.
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W]
        
    Returns:
        torch.Tensor: Attention output of shape [B, C, H, W]
    """
    h_ = x
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(B, t.shape[1], 1, C)
        .permute(0, 2, 1, 3)
        .reshape(B * 1, t.shape[1], C)
        .contiguous(),
        (q, k, v),
    )
    out = F.scaled_dot_product_attention(q, k, v)

    out = (
        out.unsqueeze(0)
        .reshape(B, 1, out.shape[1], C)
        .permute(0, 2, 1, 3)
        .reshape(B, out.shape[1], C)
    )
    out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
    out = self.proj_out(out)
    return out

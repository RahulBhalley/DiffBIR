"""
SwinIR: Image Restoration Using Swin Transformer
https://arxiv.org/abs/2108.10257

This module implements the SwinIR architecture for image restoration tasks like super-resolution,
denoising and JPEG artifact reduction. The model uses Swin Transformer blocks for feature extraction
and various upsampling strategies for reconstruction.

Key components:
- Shallow feature extraction using convolutions
- Deep feature extraction using Swin Transformer blocks
- High quality image reconstruction with different upsampling options
- Support for classical SR, lightweight SR and real-world SR

Originally written by Ze Liu, modified by Jingyun Liang.
Adapted from DifFace (https://github.com/zsyOAOA/DifFace/blob/master/models/swinir.py)
"""

import math
from typing import Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """
    Multi-layer perceptron module used in Transformer blocks.
    
    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden features. Defaults to in_features.
        out_features (int, optional): Number of output features. Defaults to in_features.
        act_layer (nn.Module): Activation layer. Defaults to nn.GELU.
        drop (float): Dropout rate. Defaults to 0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, L, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Partition input tensor into non-overlapping windows.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C)
        window_size (int): Window size

    Returns:
        torch.Tensor: Windows tensor of shape (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    
    Args:
        windows (torch.Tensor): Windows tensor of shape (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        torch.Tensor: Reversed tensor of shape (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted window.

    Args:
        dim (int): Number of input channels
        window_size (tuple[int]): The height and width of the window
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # Fix: Pass indexing="ij" to avoid warning
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (num_windows*B, N, C)
            mask (torch.Tensor, optional): Attention mask of shape (num_windows, Wh*Ww, Wh*Ww). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        """Extra representation of module parameters."""
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        """Calculate FLOPs for computing attention."""
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        num_heads (int): Number of attention heads
        window_size (int): Window size
        shift_size (int): Shift size for SW-MSA
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        """Calculate attention mask for SW-MSA."""
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C)
            x_size (tuple[int]): Spatial size of input tensor

        Returns:
            torch.Tensor: Output tensor of shape (B, H*W, C)
        """
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        """Extra representation of module parameters."""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        """Calculate FLOPs."""
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature
        dim (int): Number of input channels
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, H/2*W/2, 2*C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        """Extra representation of module parameters."""
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """Calculate FLOPs."""
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        depth (int): Number of blocks
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C)
            x_size (tuple[int]): Spatial size of input tensor

        Returns:
            torch.Tensor: Output tensor of shape (B, H*W, C) or (B, H/2*W/2, 2*C)
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        """Extra representation of module parameters."""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """Calculate FLOPs."""
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        depth (int): Number of blocks
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size. Default: 4
        resi_connection (str): The convolutional block before residual connection. Default: '1conv'
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C)
            x_size (tuple[int]): Spatial size of input tensor

        Returns:
            torch.Tensor: Output tensor of shape (B, H*W, C)
        """
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        """Calculate FLOPs."""
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding.

    Args:
        img_size (int): Image size. Default: 224
        patch_size (int): Patch token size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

# Originally borrowed from DifFace (https://github.com/zsyOAOA/DifFace/blob/master/models/swinir.py)

import math
from typing import Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    
    This module implements the window-based multi-head self attention mechanism described in the 
    Swin Transformer paper. It supports both shifted and non-shifted window configurations.
    
    The attention is computed within local windows, with relative position embeddings added to 
    capture spatial relationships. The module can process both shifted and non-shifted windows
    by using an optional attention mask.

    Args:
        dim (int): Number of input channels/features
        window_size (tuple[int]): The height and width of each attention window
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): If True, adds learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None
        attn_drop (float, optional): Dropout rate for attention weights. Default: 0.0
        proj_drop (float, optional): Dropout rate for output. Default: 0.0

    Shape:
        - Input: (B*nW, N, C) where B is batch size, nW is number of windows, N is window size*window size,
                C is input channels
        - Output: (B*nW, N, C) same shape as input
        
    Examples::
        >>> window_size = (7,7)
        >>> attn = WindowAttention(dim=96, window_size=window_size, num_heads=4)
        >>> x = torch.randn(8, 49, 96)  # B*nW=8, N=7*7, C=96
        >>> out = attn(x)  # (8, 49, 96)
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define learnable relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # Fix: Pass indexing="ij" to avoid warning
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Linear layers for query, key, value projections and output projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative position bias table
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward pass of window attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B*nW, N, C)
            mask (torch.Tensor, optional): Attention mask of shape (nW, N, N), where N is window size squared.
                                         Value should be between (-inf, 0]. Defaults to None.
                                         
        Returns:
            torch.Tensor: Output tensor of shape (B*nW, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Scale query
        q = q * self.scale
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        """Extra string representation of the module."""
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        """Calculate FLOPs for computing attention.
        
        Args:
            N (int): Number of tokens in one window (window_size * window_size)
            
        Returns:
            int: Number of FLOPs
        """
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    This module implements a single Swin Transformer block that consists of:
    1. Layer normalization
    2. Window-based multi-head self attention (W-MSA) or shifted window based multi-head self-attention (SW-MSA)
    3. Layer normalization 
    4. MLP with GELU activation

    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution (H, W)
        num_heads (int): Number of attention heads
        window_size (int, optional): Window size for attention. Default: 7
        shift_size (int, optional): Shift size for SW-MSA. Default: 0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm

    Attributes:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        shift_size (int): Shift size for SW-MSA
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        norm1 (nn.Module): First normalization layer
        attn (WindowAttention): Window attention module
        drop_path (nn.Module): Drop path layer
        norm2 (nn.Module): Second normalization layer
        mlp (Mlp): MLP module
        attn_mask (torch.Tensor): Attention mask for SW-MSA
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        """Calculate attention mask for SW-MSA.
        
        Args:
            x_size (tuple[int]): Input resolution

        Returns:
            torch.Tensor: Attention mask with shape (nW, window_size*window_size, window_size*window_size)
        """
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, C)
            x_size (tuple[int]): Input resolution

        Returns:
            torch.Tensor: Output tensor with shape (B, L, C)
        """
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        """Extra string representation.

        Returns:
            str: String representation of module
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        """Calculate FLOPs.

        Returns:
            float: Number of FLOPs
        """
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """Patch Merging Layer that merges multiple patches into one.

    This layer merges multiple patches into one patch by concatenating features from 
    adjacent patches and applying linear projection. It reduces spatial dimensions 
    while increasing the channel dimension.

    Args:
        input_resolution (tuple[int]): Resolution of input feature, (H, W)
        dim (int): Number of input channels
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm

    Shape:
        - Input: (B, H*W, C)
        - Output: (B, H*W/4, 2*C)
        where B is batch size, H and W are height and width, C is channel dimension

    Examples::

        >>> input = torch.randn(1, 64*64, 96)
        >>> layer = PatchMerging((64,64), 96)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([1, 1024, 192])
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, H*W/4, 2*C)

        Raises:
            AssertionError: If input size doesn't match expected dimensions
            AssertionError: If input resolution is not even
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # Gather features from 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        """Extra string representation.

        Returns:
            str: String containing input resolution and dimension
        """
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """Calculate FLOPs (floating point operations).

        Returns:
            int: Number of FLOPs
        """
        H, W = self.input_resolution
        flops = H * W * self.dim  # norm
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim  # reduction
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    This layer contains multiple Swin Transformer blocks and an optional downsampling layer.
    It processes input features through a sequence of transformer blocks with window attention
    and optional shifted window attention.

    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution (H, W)
        depth (int): Number of blocks in this layer
        num_heads (int): Number of attention heads
        window_size (int): Local window size for attention
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop (float): Attention dropout rate. Default: 0
        drop_path (float | tuple[float]): Stochastic depth rate. Default: 0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False

    Shape:
        - Input: (B, H*W, C)
        - Output: (B, H*W, C) or (B, H*W/4, 2*C) if downsample is used
        where B is batch size, H and W are height and width, C is channel dimension

    Examples::

        >>> input = torch.randn(1, 64*64, 96)
        >>> layer = BasicLayer(96, (64,64), 2, 6, 8)
        >>> output = layer(input, (64,64))
        >>> print(output.shape)
        torch.Size([1, 4096, 96])
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C)
            x_size (tuple[int]): Spatial size of input (H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, H*W, C) or (B, H*W/4, 2*C)
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        """Extra string representation.

        Returns:
            str: String containing layer parameters
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """Calculate FLOPs.

        Returns:
            float: Number of FLOPs
        """
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    This module implements a residual block using Swin Transformer layers. It consists of:
    1. A residual Swin Transformer group
    2. A convolutional layer (either 1 or 3 convs)
    3. Patch embedding/unembedding layers

    The module processes input features through the transformer group, applies convolutions,
    and adds a residual connection.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        depth (int): Number of Swin Transformer blocks in the residual group.
        num_heads (int): Number of attention heads in each transformer block.
        window_size (int): Size of local attention window.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension.
            Default: 4.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            Default: None
        drop (float, optional): Dropout rate after attention and MLP. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            Default: None
        use_checkpoint (bool, optional): Whether to use checkpointing to save memory.
            Default: False
        img_size (int, optional): Input image size. Default: 224
        patch_size (int, optional): Patch size for embedding. Default: 4
        resi_connection (str, optional): Type of convolutional block before residual connection.
            Options: '1conv' or '3conv'. Default: '1conv'

    Shape:
        - Input: (B, L, C) where L = H*W
        - Output: (B, L, C)
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C)
            x_size (tuple[int]): Spatial size of input (H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, L, C)
        """
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        """Calculate FLOPs.

        Returns:
            float: Number of FLOPs
        """
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    This module converts an image tensor into a sequence of flattened patches.
    It can optionally apply normalization after the flattening.

    Args:
        img_size (int, optional): Input image size. Default: 224
        patch_size (int, optional): Size of each patch. Default: 4
        in_chans (int, optional): Number of input channels. Default: 3
        embed_dim (int, optional): Embedding dimension. Default: 96
        norm_layer (nn.Module, optional): Normalization layer. Default: None

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, L, embed_dim) where L = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, L, embed_dim)
        """
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """Calculate FLOPs.

        Returns:
            float: Number of FLOPs
        """
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    """Patch Unembedding.

    This module converts a sequence of embedded patches back to image-like spatial dimensions.
    It is the inverse operation of PatchEmbed.

    Args:
        img_size (int, optional): Output image size. Default: 224
        patch_size (int, optional): Size of each patch. Default: 4
        in_chans (int, optional): Number of input channels. Default: 3
        embed_dim (int, optional): Embedding dimension. Default: 96
        norm_layer (nn.Module, optional): Normalization layer. Default: None

    Shape:
        - Input: (B, L, embed_dim)
        - Output: (B, embed_dim, H, W)
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, embed_dim)
            x_size (tuple[int]): Output spatial size (H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, embed_dim, H, W)
        """
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        """Calculate FLOPs.

        Returns:
            float: Number of FLOPs
        """
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    This module upsamples a feature map using either pixel shuffle (for 2^n scaling)
    or pixel shuffle with scale 3.

    Args:
        scale (int): Upsampling scale factor. Must be either 2^n or 3.
        num_feat (int): Number of input/output channels.

    Shape:
        - Input: (B, num_feat, H, W)
        - Output: (B, num_feat, H*scale, W*scale)
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """Single-step upsampling module.

    A lightweight version of Upsample that always uses a single convolution followed by pixel shuffle.
    Used in lightweight SR to save parameters.

    Args:
        scale (int): Upsampling scale factor. Must be either 2^n or 3.
        num_feat (int): Number of input channels.
        num_out_ch (int): Number of output channels.
        input_resolution (tuple[int], optional): Input resolution for FLOP calculation.

    Shape:
        - Input: (B, num_feat, H, W)
        - Output: (B, num_out_ch, H*scale, W*scale)
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        """Calculate FLOPs.

        Returns:
            float: Number of FLOPs
        """
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

class SwinIR(nn.Module):
    """SwinIR: Image Restoration Using Swin Transformer.

    This is the main SwinIR model class that implements the architecture described in:
    "SwinIR: Image Restoration Using Swin Transformer"

    The model consists of three main components:
    1. Shallow feature extraction using convolutional layers
    2. Deep feature extraction using Swin Transformer blocks 
    3. High quality image reconstruction with upsampling

    Parameters
    ----------
    img_size : int, optional
        Input image size, by default 64
    patch_size : int, optional
        Patch size for embedding, by default 1
    in_chans : int, optional
        Number of input image channels, by default 3
    embed_dim : int, optional
        Dimension of embedded features, by default 96
    depths : list[int], optional
        Number of Swin blocks in each layer, by default [6,6,6,6]
    num_heads : list[int], optional
        Number of attention heads in each layer, by default [6,6,6,6]
    window_size : int, optional
        Size of local attention window, by default 7
    mlp_ratio : float, optional
        MLP hidden dimension ratio, by default 4.0
    qkv_bias : bool, optional
        If True, add bias to query/key/value projections, by default True
    qk_scale : float, optional
        Override default qk scale of head_dim ** -0.5 if set, by default None
    drop_rate : float, optional
        Dropout rate after embedding, by default 0.0
    attn_drop_rate : float, optional
        Attention dropout rate, by default 0.0
    drop_path_rate : float, optional
        Stochastic depth rate, by default 0.1
    norm_layer : nn.Module, optional
        Normalization layer, by default nn.LayerNorm
    ape : bool, optional
        If True, add absolute position embedding, by default False
    patch_norm : bool, optional
        If True, add normalization after patch embedding, by default True
    use_checkpoint : bool, optional
        Whether to use checkpointing to save memory, by default False
    sf : int, optional
        Upscaling factor, by default 4
    img_range : float, optional
        Image value range, by default 1.0
    upsampler : str, optional
        The upsampling module type. Options are:
        - 'pixelshuffle': Classical SR with pixel shuffle
        - 'pixelshuffledirect': Lightweight SR with direct pixel shuffle
        - 'nearest+conv': Real-world SR with nearest neighbor + conv
        - '': No upsampling (for denoising etc.)
        By default ''
    resi_connection : str, optional
        The conv block before residual connection.
        Options are '1conv' or '3conv', by default '1conv'
    unshuffle : bool, optional
        If True, apply pixel unshuffle at input, by default False
    unshuffle_scale : int, optional
        Scale factor for pixel unshuffle, by default None
    hq_key : str, optional
        Key for high quality images in training data, by default "jpg"
    lq_key : str, optional
        Key for low quality images in training data, by default "hint"
    learning_rate : float, optional
        Initial learning rate, by default None
    weight_decay : float, optional
        Weight decay (L2 penalty), by default None

    Attributes
    ----------
    img_range : float
        Image value range
    mean : torch.Tensor
        Mean values for normalization
    upscale : int
        Upscaling factor
    upsampler : str
        Type of upsampling module
    window_size : int
        Size of attention window
    unshuffle_scale : int
        Scale for pixel unshuffle
    unshuffle : bool
        Whether to use pixel unshuffle
    num_layers : int
        Number of transformer layers
    embed_dim : int
        Embedding dimension
    ape : bool
        Whether to use absolute position embedding
    patch_norm : bool
        Whether to use patch normalization
    num_features : int
        Number of feature channels
    mlp_ratio : float
        MLP expansion ratio

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the model
    forward_features(x: torch.Tensor) -> torch.Tensor
        Forward pass through feature extraction layers
    check_image_size(x: torch.Tensor) -> torch.Tensor
        Pad input image to be compatible with window size
    flops() -> int
        Calculate total floating point operations
    no_weight_decay() -> Set[str]
        Get parameter names that should not use weight decay
    no_weight_decay_keywords() -> Set[str]
        Get parameter name keywords that should not use weight decay
    _init_weights(m: nn.Module) -> None
        Initialize model weights

    Examples
    --------
    >>> model = SwinIR(upscaler='pixelshuffle', img_size=64, window_size=8)
    >>> x = torch.randn(1, 3, 64, 64)
    >>> output = model(x)
    >>> print(output.shape)
    torch.Size([1, 3, 256, 256])
    """

    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        sf=4,
        img_range=1.,
        upsampler='',
        resi_connection='1conv',
        unshuffle=False,
        unshuffle_scale=None,
        hq_key: str="jpg",
        lq_key: str="hint",
        learning_rate: float=None,
        weight_decay: float=None
    ) -> "SwinIR":
        super(SwinIR, self).__init__()
        num_in_ch = in_chans * (unshuffle_scale**2) if unshuffle else in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = sf
        self.upsampler = upsampler
        self.window_size = window_size
        self.unshuffle_scale = unshuffle_scale
        self.unshuffle = unshuffle

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        if unshuffle:
            assert unshuffle_scale is not None
            self.conv_first = nn.Sequential(
                nn.PixelUnshuffle(sf),
                nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1),
            )
        else:
            self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1)
            )

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(sf, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(
                sf, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1])
            )
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            elif self.upscale == 8:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
                self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize the weights.

        Parameters
        ----------
        m : nn.Module
            Module to initialize
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Get parameter names that should not use weight decay.

        Returns
        -------
        Set[str]
            Set of parameter names
        """
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> Set[str]:
        """Get parameter name keywords that should not use weight decay.

        Returns
        -------
        Set[str]
            Set of parameter name keywords
        """
        return {'relative_position_bias_table'}

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Check and pad input image size to be compatible with window size.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor

        Returns
        -------
        torch.Tensor
            Padded image tensor
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output features
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Output image tensor of shape (B, C, H*upscale, W*upscale)
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            elif self.upscale == 8:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
                x = self.lrelu(self.conv_up3(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self) -> int:
        """Calculate total floating point operations.

        Returns
        -------
        int
            Number of FLOPs
        """
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

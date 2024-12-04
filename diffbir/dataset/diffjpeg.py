# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/diffjpeg.py
"""Differentiable JPEG compression/decompression operations.

This module implements differentiable JPEG compression and decompression operations
that can be used in deep learning models. The implementation is based on the DiffJPEG
project (https://github.com/mlomnitz/DiffJPEG) with modifications to handle images
not divisible by 8 following the approach described in:
https://dsp.stackexchange.com/questions/35339/jpeg-dct-padding/35343#35343

The module provides:
- Quantization tables for luminance (y_table) and chrominance (c_table) channels
- Differentiable rounding function
- Quality factor calculation
- RGB to YCbCr color conversion
- Chroma subsampling operations

These components allow for end-to-end training through JPEG compression.
"""

import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------ utils ------------------------#

# Standard JPEG luminance quantization table
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
y_table = nn.Parameter(torch.from_numpy(y_table))

# Standard JPEG chrominance quantization table
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """Differentiable rounding function.
    
    This function implements a differentiable approximation of rounding
    by adding a cubic correction term to the standard round operation.
    This allows gradients to flow through the rounding operation during
    backpropagation.

    Args:
        x (Tensor): Input tensor to be rounded

    Returns:
        Tensor: Differentiably rounded tensor
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """Calculate compression factor from JPEG quality setting.

    Converts the JPEG quality setting (0-100) to the internal scaling factor
    used for the quantization tables. Uses the standard JPEG quality-to-factor
    formula where quality < 50 uses a different scaling than quality >= 50.

    Args:
        quality (float): JPEG quality setting in range [0, 100]
            Higher values mean better quality and less compression.

    Returns:
        float: Scaling factor for quantization tables
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


# ------------------------ compression ------------------------#
class RGB2YCbCrJpeg(nn.Module):
    """Converts RGB images to YCbCr color space using JPEG standard conversion.

    This module implements the standard JPEG color space conversion from RGB to YCbCr.
    The conversion uses the following matrix multiplication:
    [Y]   = [0.299    0.587    0.114   ] [R]   [0  ]
    [Cb]  = [-0.1687  -0.3313  0.5     ] [G] + [128]
    [Cr]  = [0.5      -0.4187  -0.0813 ] [B]   [128]

    The output Y channel represents luminance while Cb and Cr represent chrominance.
    This separation allows JPEG to compress chrominance channels more aggressively
    since human vision is less sensitive to color detail.
    """

    def __init__(self):
        super(RGB2YCbCrJpeg, self).__init__()
        matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
                          dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        """Convert RGB image batch to YCbCr color space.

        Args:
            image (Tensor): Input RGB images of shape (batch, 3, height, width)
                Values should be in range [0, 255]

        Returns:
            Tensor: YCbCr images of shape (batch, height, width, 3)
                Y channel in range [0, 255]
                Cb and Cr channels in range [0, 255] centered at 128
        """
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        return result.view(image.shape)


class ChromaSubsampling(nn.Module):
    """Performs 4:2:0 chroma subsampling on YCbCr images.

    This module implements chroma subsampling by averaging 2x2 blocks in the
    chrominance channels (Cb and Cr) while keeping the luminance (Y) at full
    resolution. This is the most common subsampling scheme used in JPEG.

    The 4:2:0 designation means:
    - 4: Full resolution Y sampling
    - 2: Half resolution Cb/Cr sampling horizontally
    - 0: Half resolution Cb/Cr sampling vertically
    """

    def __init__(self):
        super(ChromaSubsampling, self).__init__()

    def forward(self, image):
        """Apply chroma subsampling to YCbCr image batch.

        Args:
            image (Tensor): Input YCbCr images of shape (batch, height, width, 3)

        Returns:
            tuple: Contains:
                - y (Tensor): Y channel at full resolution (batch, height, width)
                - cb (Tensor): Cb channel subsampled (batch, height/2, width/2)
                - cr (Tensor): Cr channel subsampled (batch, height/2, width/2)
        """
        image_2 = image.permute(0, 3, 1, 2).clone()
        cb = F.avg_pool2d(image_2[:, 1, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2), count_include_pad=False)
        cr = F.avg_pool2d(image_2[:, 2, :, :].unsqueeze(1), kernel_size=2, stride=(2, 2), count_include_pad=False)
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class BlockSplitting(nn.Module):
    """Splits images into 8x8 blocks for DCT transformation.

    This module divides input images into non-overlapping 8x8 blocks which is
    the standard block size used in JPEG compression. The DCT is then applied
    to each block independently.

    The block size of 8x8 was chosen in JPEG as a trade-off between:
    - Compression efficiency
    - Computational complexity
    - Blocking artifacts
    """

    def __init__(self):
        super(BlockSplitting, self).__init__()
        self.k = 8

    def forward(self, image):
        """Split image into 8x8 blocks.

        Args:
            image (Tensor): Input image of shape (batch, height, width)
                Height and width must be multiples of 8

        Returns:
            Tensor: Blocks of shape (batch, num_blocks, 8, 8)
                where num_blocks = (height * width) / 64
        """
        height, _ = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class DCT8x8(nn.Module):
    """Applies 2D Discrete Cosine Transform to 8x8 blocks.

    This module implements the 2D Type-II DCT used in JPEG compression.
    The DCT converts spatial domain data into frequency domain coefficients.
    Lower frequency coefficients (top-left) typically contain more important
    image information than higher frequency ones (bottom-right).

    The transform is separable and orthogonal. The implementation uses
    a precomputed tensor of cosine values for efficiency.
    """

    def __init__(self):
        super(DCT8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        """Apply DCT to image blocks.

        Args:
            image (Tensor): Input image blocks of shape (batch, num_blocks, 8, 8)
                Values should be centered by subtracting 128

        Returns:
            Tensor: DCT coefficients of shape (batch, num_blocks, 8, 8)
        """
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class YQuantize(nn.Module):
    """JPEG quantization for Y (luminance) channel.

    This module applies JPEG standard quantization to DCT coefficients of the
    luminance channel. It uses the standard JPEG luminance quantization table
    scaled by a quality factor.

    Quantization reduces the precision of the DCT coefficients, enabling better
    compression at the cost of image quality. The quantization table is designed
    to preserve low frequency coefficients (more important visually) while more
    aggressively quantizing high frequency ones.

    Args:
        rounding (function): Differentiable rounding function to use
    """

    def __init__(self, rounding):
        super(YQuantize, self).__init__()
        self.rounding = rounding
        self.y_table = y_table

    def forward(self, image, factor=1):
        """Quantize DCT coefficients.

        Args:
            image (Tensor): DCT coefficients of shape (batch, num_blocks, 8, 8)
            factor (float or Tensor): Quality factor for scaling quantization table
                If float, same factor used for whole batch
                If Tensor, should be shape (batch,) for per-image factors

        Returns:
            Tensor: Quantized coefficients of shape (batch, num_blocks, 8, 8)
        """
        if isinstance(factor, (int, float)):
            image = image.float() / (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        image = self.rounding(image)
        return image


class CQuantize(nn.Module):
    """JPEG quantization for CbCr (chrominance) channels.

    This module applies JPEG standard quantization to DCT coefficients of the
    chrominance channels (Cb and Cr). It uses the standard JPEG chrominance 
    quantization table scaled by a quality factor.

    Quantization reduces the precision of the DCT coefficients, enabling better
    compression at the cost of image quality. The chrominance quantization table
    is generally more aggressive than the luminance table since human vision is
    less sensitive to color detail.

    Args:
        rounding (function): Differentiable rounding function to use
    """

    def __init__(self, rounding):
        super(CQuantize, self).__init__()
        self.rounding = rounding
        self.c_table = c_table

    def forward(self, image, factor=1):
        """Quantize DCT coefficients for chrominance channels.

        Args:
            image (Tensor): DCT coefficients of shape (batch, num_blocks, 8, 8)
            factor (float or Tensor): Quality factor for scaling quantization table
                If float, same factor used for whole batch
                If Tensor, should be shape (batch,) for per-image factors

        Returns:
            Tensor: Quantized coefficients of shape (batch, num_blocks, 8, 8)
        """
        if isinstance(factor, (int, float)):
            image = image.float() / (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        image = self.rounding(image)
        return image


class CompressJpeg(nn.Module):
    """Full JPEG compression algorithm.
    
    This module implements the complete JPEG compression pipeline:
    1. Color space conversion from RGB to YCbCr
    2. Chroma subsampling
    3. Block splitting into 8x8 blocks
    4. DCT transform
    5. Quantization (different for Y and CbCr channels)

    The compression quality can be controlled via the factor parameter,
    where lower values result in higher compression and lower quality.

    Args:
        rounding (function, optional): Differentiable rounding function to use.
            Defaults to torch.round.
    """

    def __init__(self, rounding=torch.round):
        super(CompressJpeg, self).__init__()
        self.l1 = nn.Sequential(RGB2YCbCrJpeg(), ChromaSubsampling())
        self.l2 = nn.Sequential(BlockSplitting(), DCT8x8())
        self.c_quantize = CQuantize(rounding=rounding)
        self.y_quantize = YQuantize(rounding=rounding)

    def forward(self, image, factor=1):
        """Compress an RGB image using JPEG algorithm.

        Args:
            image (Tensor): Input RGB image tensor of shape (batch, 3, height, width).
                Expected to be in range [0, 1].
            factor (float or Tensor): Quality factor for compression.
                Lower values = higher compression, lower quality.
                If float, same factor used for whole batch.
                If Tensor, should be shape (batch,) for per-image factors.

        Returns:
            tuple(Tensor, Tensor, Tensor): Tuple containing compressed Y, Cb, and Cr
                components, each of shape (batch, num_blocks, 8, 8).
        """
        y, cb, cr = self.l1(image * 255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp, factor=factor)
            else:
                comp = self.y_quantize(comp, factor=factor)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


# ------------------------ decompression ------------------------#


class YDequantize(nn.Module):
    """Dequantize Y (luminance) channel by multiplying with scaled quantization table.

    This module reverses the quantization step of JPEG compression for the luminance channel.
    It multiplies the compressed values by the standard JPEG luminance quantization table,
    scaled by the quality factor.

    The dequantization step is essential for recovering the DCT coefficients before
    inverse DCT transform can be applied.
    """

    def __init__(self):
        super(YDequantize, self).__init__()
        self.y_table = y_table

    def forward(self, image, factor=1):
        """Dequantize luminance values.

        Args:
            image (Tensor): Quantized Y channel coefficients of shape (batch, num_blocks, 8, 8)
            factor (float or Tensor): Quality factor for scaling quantization table.
                If float, same factor used for whole batch.
                If Tensor, should be shape (batch,) for per-image factors.

        Returns:
            Tensor: Dequantized coefficients of shape (batch, num_blocks, 8, 8)
        """
        if isinstance(factor, (int, float)):
            out = image * (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = image * table
        return out


class CDequantize(nn.Module):
    """Dequantize Cb/Cr (chrominance) channels by multiplying with scaled quantization table.

    This module reverses the quantization step of JPEG compression for the chrominance channels.
    It multiplies the compressed values by the standard JPEG chrominance quantization table,
    scaled by the quality factor.

    The chrominance channels typically use more aggressive quantization than luminance
    since human vision is less sensitive to color detail.
    """

    def __init__(self):
        super(CDequantize, self).__init__()
        self.c_table = c_table

    def forward(self, image, factor=1):
        """Dequantize chrominance values.

        Args:
            image (Tensor): Quantized Cb or Cr channel coefficients of shape (batch, num_blocks, 8, 8)
            factor (float or Tensor): Quality factor for scaling quantization table.
                If float, same factor used for whole batch.
                If Tensor, should be shape (batch,) for per-image factors.

        Returns:
            Tensor: Dequantized coefficients of shape (batch, num_blocks, 8, 8)
        """
        if isinstance(factor, (int, float)):
            out = image * (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = image * table
        return out


class iDCT8x8(nn.Module):
    """Inverse Discrete Cosine Transform on 8x8 blocks.

    This module implements the inverse DCT transform used in JPEG decompression.
    It converts frequency domain coefficients back to spatial domain pixel values.
    
    The implementation uses the separable property of 2D DCT, representing the transform
    as a product of 1D DCTs. The normalization factors (alpha) and cosine terms are
    precomputed for efficiency.
    """

    def __init__(self):
        super(iDCT8x8, self).__init__()
        # Normalization factors for DC and AC components
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        # Precompute cosine terms for all possible x,y,u,v combinations
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        """Apply inverse DCT to 8x8 blocks.

        Args:
            image (Tensor): DCT coefficients of shape (batch, num_blocks, 8, 8)

        Returns:
            Tensor: Spatial domain values of shape (batch, num_blocks, 8, 8),
                   with values centered around 128
        """
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class BlockMerging(nn.Module):
    """Merge 8x8 blocks back into complete image.

    This module reconstructs the full image from the 8x8 blocks used in JPEG compression.
    It rearranges the blocks into their original spatial positions in the image.
    """

    def __init__(self):
        super(BlockMerging, self).__init__()

    def forward(self, patches, height, width):
        """Merge blocks into image.

        Args:
            patches (Tensor): Image blocks of shape (batch, height*width/64, 8, 8)
            height (int): Original image height
            width (int): Original image width

        Returns:
            Tensor: Merged image of shape (batch, height, width)
        """
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class ChromaUpsampling(nn.Module):
    """Performs 4:2:0 chroma upsampling on YCbCr images.

    This module upsamples the chrominance channels (Cb and Cr) from half resolution
    back to full resolution by repeating values in 2x2 blocks. This reverses the
    chroma subsampling done during compression.

    The upsampling is done by:
    1. Repeating each value 2x horizontally and vertically 
    2. Reshaping to match the full resolution dimensions
    3. Concatenating with the full resolution Y channel
    """

    def __init__(self):
        super(ChromaUpsampling, self).__init__()

    def forward(self, y, cb, cr):
        """Upsample chrominance channels and combine with luminance.

        Args:
            y (Tensor): Luminance channel of shape (batch, height, width)
            cb (Tensor): Cb chrominance channel of shape (batch, height/2, width/2)
            cr (Tensor): Cr chrominance channel of shape (batch, height/2, width/2)

        Returns:
            Tensor: Combined YCbCr image of shape (batch, height, width, 3)
                   with upsampled chroma channels
        """

        def repeat(x, k=2):
            """Helper function to repeat values in both dimensions.
            
            Args:
                x (Tensor): Input tensor to repeat
                k (int): Repeat factor (default: 2)
                
            Returns:
                Tensor: Repeated tensor with shape (batch, height*k, width*k)
            """
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class YCbCr2RGBJpeg(nn.Module):
    """Converts YCbCr images to RGB color space using JPEG standard conversion.

    This module implements the standard JPEG color space conversion from YCbCr to RGB.
    The conversion uses the following matrix multiplication:
    [R]   = [1.0     0.0      1.402   ] [Y  ]
    [G]   = [1.0    -0.344   -0.714   ] [Cb-128]
    [B]   = [1.0     1.772    0.0     ] [Cr-128]

    The input Cb and Cr values are shifted by -128 before multiplication since they
    are centered at 128 in YCbCr space.
    """

    def __init__(self):
        super(YCbCr2RGBJpeg, self).__init__()

        matrix = np.array([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        """Convert YCbCr image batch to RGB color space.

        Args:
            image (Tensor): Input YCbCr images of shape (batch, height, width, 3)
                Y channel in range [0, 255]
                Cb and Cr channels in range [0, 255] centered at 128

        Returns:
            Tensor: RGB images of shape (batch, 3, height, width)
                Values in range [0, 255]
        """
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        return result.view(image.shape).permute(0, 3, 1, 2)


class DeCompressJpeg(nn.Module):
    """Full JPEG decompression pipeline.

    This module implements the complete JPEG decompression process:
    1. Dequantize DCT coefficients using quantization tables
    2. Apply inverse DCT to convert to spatial domain
    3. Merge 8x8 blocks back into complete image planes
    4. Upsample chroma channels from 4:2:0 format
    5. Convert from YCbCr to RGB color space

    The implementation follows the standard JPEG decompression specification
    while maintaining differentiability for use in deep learning models.

    Args:
        rounding (function): Rounding function to use during dequantization.
            Default: torch.round
    """

    def __init__(self, rounding=torch.round):
        super(DeCompressJpeg, self).__init__()
        self.c_dequantize = CDequantize()
        self.y_dequantize = YDequantize()
        self.idct = iDCT8x8()
        self.merging = BlockMerging()
        self.chroma = ChromaUpsampling()
        self.colors = YCbCr2RGBJpeg()

    def forward(self, y, cb, cr, imgh, imgw, factor=1):
        """Decompress JPEG data to RGB image.

        Args:
            y (Tensor): Luminance DCT coefficients, shape (batch, h*w/64, 8, 8)
            cb (Tensor): Cb chrominance DCT coefficients, shape (batch, h*w/256, 8, 8)
            cr (Tensor): Cr chrominance DCT coefficients, shape (batch, h*w/256, 8, 8)
            imgh (int): Original image height
            imgw (int): Original image width
            factor (float): Quality factor for dequantization. Default: 1

        Returns:
            Tensor: Decompressed RGB image of shape (batch, 3, height, width)
                   with values in range [0, 1]
        """
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor=factor)
                height, width = int(imgh / 2), int(imgw / 2)
            else:
                comp = self.y_dequantize(components[k], factor=factor)
                height, width = imgh, imgw
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255 * torch.ones_like(image), torch.max(torch.zeros_like(image), image))
        return image / 255


# ------------------------ main DiffJPEG ------------------------ #


class DiffJPEG(nn.Module):
    """Differentiable JPEG compression/decompression layer.
    
    This module implements a differentiable version of JPEG compression and decompression.
    The implementation produces results slightly different from cv2's JPEG implementation
    but supports batch processing and can be used in deep learning models with gradient flow.

    The compression process follows these steps:
    1. Pad input image to multiple of 16x16 if needed
    2. Convert RGB to YCbCr color space and downsample chrominance
    3. Apply DCT transform on 8x8 blocks
    4. Quantize DCT coefficients based on quality factor
    5. Dequantize and reverse the process to recover the image

    Args:
        differentiable (bool): If True, uses custom differentiable rounding function during
            quantization/dequantization. If False, uses standard torch.round. Default: True

    Example:
        >>> jpeg = DiffJPEG(differentiable=True)
        >>> input = torch.rand(4, 3, 64, 64)  # BCHW format, range [0,1]
        >>> quality = 80  # JPEG quality factor (1-100)
        >>> compressed = jpeg(input, quality)  # Compressed-decompressed result
    """

    def __init__(self, differentiable=True):
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round

        self.compress = CompressJpeg(rounding=rounding)
        self.decompress = DeCompressJpeg(rounding=rounding)

    def forward(self, x, quality):
        """Apply JPEG compression-decompression to input images.

        Args:
            x (Tensor): Input image tensor in BCHW format, RGB channels, 
                with values normalized to range [0, 1]
            quality (float or Tensor): JPEG quality factor(s). If float/int, applies same
                quality to whole batch. If Tensor, should have batch_size elements for
                per-image quality factors. Quality should be between 1 and 100.

        Returns:
            Tensor: Compressed-decompressed image tensor in same format as input.
                Values are in range [0, 1].

        Note:
            Input tensors must have heights and widths that are multiples of 16.
            If not, zero padding will be automatically added and removed.
        """
        factor = quality
        if isinstance(factor, (int, float)):
            factor = quality_to_factor(factor)
        else:
            for i in range(factor.size(0)):
                factor[i] = quality_to_factor(factor[i])
        h, w = x.size()[-2:]
        h_pad, w_pad = 0, 0
        # JPEG compression requires image dimensions to be multiples of 16
        # for proper 8x8 block processing with chroma subsampling
        if h % 16 != 0:
            h_pad = 16 - h % 16
        if w % 16 != 0:
            w_pad = 16 - w % 16
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)

        y, cb, cr = self.compress(x, factor=factor)
        recovered = self.decompress(y, cb, cr, (h + h_pad), (w + w_pad), factor=factor)
        recovered = recovered[:, :, 0:h, 0:w]
        return recovered.contiguous()

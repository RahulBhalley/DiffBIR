from typing import Mapping, Any, Tuple, Callable, Dict, Literal
import importlib
import os
from urllib.parse import urlparse

import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from torch.hub import download_url_to_file, get_dir


def get_obj_from_str(string: str, reload: bool = False) -> Any:
    """Get object from a string path.
    
    Args:
        string: String path to the object, e.g. "module.submodule.class"
        reload: Whether to reload the module before getting the object
        
    Returns:
        The object specified by the string path
        
    Raises:
        ImportError: If the module or object cannot be imported
        AttributeError: If the object does not exist in the module
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    """Instantiate an object from a config dictionary.
    
    Args:
        config: Dictionary containing target class path and parameters.
               Must have a "target" key with class path string.
               May have a "params" key with constructor arguments.
               
    Returns:
        Instantiated object
        
    Raises:
        KeyError: If config does not contain "target" key
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def wavelet_blur(image: Tensor, radius: int):
    """Apply wavelet blur to the input tensor using convolution.
    
    Args:
        image: Input tensor of shape (1, 3, H, W)
        radius: Dilation radius for the convolution kernel
        
    Returns:
        Blurred tensor of same shape as input
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """Decompose image into high and low frequency components using wavelets.
    
    Args:
        image: Input tensor to decompose
        levels: Number of wavelet decomposition levels
        
    Returns:
        Tuple of (high_frequency, low_frequency) components
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2**i
        low_freq = wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    """Reconstruct image combining content high frequency with style low frequency.
    
    Args:
        content_feat: Content image features tensor
        style_feat: Style image features tensor
        
    Returns:
        Reconstructed tensor combining content structure with style color
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file from URL, downloading if necessary.

    Args:
        url: URL to download from
        model_dir: Directory to save downloaded file. If None, uses pytorch hub_dir
        progress: Whether to show download progress bar
        file_name: Name to save file as. If None, uses name from URL
        
    Returns:
        Path to the downloaded file
        
    References:
        https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    """Load model state dict from URL.
    
    Args:
        url: URL to download model from
        
    Returns:
        Model state dict
    """
    sd_path = load_file_from_url(url, model_dir="weights")
    sd = torch.load(sd_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def sliding_windows(
    h: int, w: int, tile_size: int, tile_stride: int
) -> Tuple[int, int, int, int]:
    """Generate coordinates for sliding windows over an image.
    
    Args:
        h: Image height
        w: Image width 
        tile_size: Size of each square tile
        tile_stride: Stride between tiles
        
    Returns:
        List of tuples (y_start, y_end, x_start, x_end) for each window
    """
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generate Gaussian weights for tile blending.
    
    Args:
        tile_width: Width of tile
        tile_height: Height of tile
        
    Returns:
        2D numpy array of Gaussian weights
        
    References:
        https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
    """
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (
        latent_width - 1
    ) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(
            -(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)
        )
        / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)
    ]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(
            -(y - midpoint)
            * (y - midpoint)
            / (latent_height * latent_height)
            / (2 * var)
        )
        / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)
    ]
    weights = np.outer(y_probs, x_probs)
    return weights

def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    stride: int,
    scale_type: Literal["up", "down"] = "up",
    scale: int = 1,
    channel: int | None = None,
    weight: Literal["uniform", "gaussian"] = "gaussian",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    progress: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a tiled version of a tensor processing function.

    This function wraps another function to process large tensors in tiles, which helps
    reduce memory usage. The tiles can optionally be scaled up or down and blended 
    together using uniform or gaussian weights.

    Args:
        fn: Function that processes a single tensor tile
        size: Size of each tile (both height and width)
        stride: Stride between tiles
        scale_type: Whether to scale tiles up or down. Either "up" or "down"
        scale: Scale factor to apply to tiles
        channel: Number of channels in output tensor. If None, uses input channels
        weight: Type of weights for blending tiles. Either "uniform" or "gaussian"
        dtype: Data type of output tensor. If None, uses input dtype
        device: Device for output tensor. If None, uses input device
        progress: Whether to show progress bar during processing

    Returns:
        A wrapped function that processes tensors in tiles

    Example:
        >>> def double_tensor(x):
        ...     return x * 2
        >>> tiled_fn = make_tiled_fn(double_tensor, size=64, stride=32)
        >>> output = tiled_fn(large_tensor)

    Notes:
        - The input tensor must be 4D with shape (batch, channels, height, width)
        - Tiles are processed with overlap determined by stride
        - Overlapping regions are blended using the specified weighting
        - Memory usage scales with tile size rather than full tensor size
    """
    # Only split the first input of function.
    def tiled_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if scale_type == "up":
            scale_fn = lambda n: int(n * scale)
        else:
            scale_fn = lambda n: int(n // scale)

        b, c, h, w = x.size()
        out_dtype = dtype or x.dtype
        out_device = device or x.device
        out_channel = channel or c
        out = torch.zeros(
            (b, out_channel, scale_fn(h), scale_fn(w)),
            dtype=out_dtype,
            device=out_device,
        )
        count = torch.zeros_like(out, dtype=torch.float32)
        weight_size = scale_fn(size)
        weights = (
            gaussian_weights(weight_size, weight_size)[None, None]
            if weight == "gaussian"
            else np.ones((1, 1, weight_size, weight_size))
        )
        weights = torch.tensor(
            weights,
            dtype=out_dtype,
            device=out_device,
        )

        indices = sliding_windows(h, w, size, stride)
        pbar = tqdm(
            indices, desc=f"Tiled Processing", disable=not progress, leave=False
        )
        for hi, hi_end, wi, wi_end in pbar:
            x_tile = x[..., hi:hi_end, wi:wi_end]
            out_hi, out_hi_end, out_wi, out_wi_end = map(
                scale_fn, (hi, hi_end, wi, wi_end)
            )
            if len(args) or len(kwargs):
                kwargs.update(dict(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
            out[..., out_hi:out_hi_end, out_wi:out_wi_end] += (
                fn(x_tile, *args, **kwargs) * weights
            )
            count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights
        out = out / count
        return out

    return tiled_fn


TRACE_VRAM = int(os.environ.get("TRACE_VRAM", False))


def trace_vram_usage(tag: str) -> Callable:
    """Decorator to trace VRAM usage before and after function execution.

    Args:
        tag: String identifier for the trace output

    Returns:
        Decorator function that wraps the target function

    Example:
        >>> @trace_vram_usage("model_forward")
        ... def forward_pass(model, x):
        ...     return model(x)
    """
    def wrapper_1(func: Callable) -> Callable:
        if not TRACE_VRAM:
            return func

        def wrapper_2(*args, **kwargs):
            peak_before = torch.cuda.max_memory_allocated() / (1024**3)
            ret = func(*args, **kwargs)
            torch.cuda.synchronize()
            peak_after = torch.cuda.max_memory_allocated() / (1024**3)
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            print(
                f"{YELLOW}VRAM peak before {tag}: {peak_before:.5f} GB, "
                f"after: {peak_after:.5f} GB{RESET}"
            )
            return ret

        return wrapper_2

    return wrapper_1


class VRAMPeakMonitor:
    """Context manager to monitor VRAM usage peaks.

    Args:
        tag: String identifier for the trace output

    Example:
        >>> with VRAMPeakMonitor("model_inference") as monitor:
        ...     output = model(input_tensor)
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __enter__(self):
        self.peak_before = torch.cuda.max_memory_allocated() / (1024**3)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        peak_after = torch.cuda.max_memory_allocated() / (1024**3)
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        if TRACE_VRAM:
            print(
                f"{YELLOW}VRAM peak before {self.tag}: {self.peak_before:.2f} GB, "
                f"after: {peak_after:.2f} GB{RESET}"
            )
        return False


def log_txt_as_img(wh, xc):
    """Convert text captions to image tensors.

    Args:
        wh: Tuple of (width, height) for output images
        xc: List of text captions to convert

    Returns:
        Tensor of shape (batch_size, channels, height, width) containing rendered text

    Notes:
        - Renders white background with black text
        - Uses default system font
        - Automatically wraps text to fit width
        - Normalizes output to [-1, 1] range
    """
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(
            xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def to(obj, device):
    """Recursively move objects to specified device.

    Args:
        obj: Object to move - can be tensor, dict, tuple, list or other
        device: Target device to move objects to

    Returns:
        Object with all contained tensors moved to specified device

    Example:
        >>> model_output = to(model_output, "cuda:0")
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to(v, device) for v in obj)
    if isinstance(obj, list):
        return [to(v, device) for v in obj]
    return obj


def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images using PyTorch.

    Implements the ITU-R BT.601 conversion for standard-definition television.
    See https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion for details.

    The conversion uses the following formulas:
    Y  =  0.257R + 0.504G + 0.098B + 16
    Cb = -0.148R - 0.291G + 0.439B + 128
    Cr =  0.439R - 0.368G - 0.071B + 128

    Args:
        img (torch.Tensor): Input RGB images with shape (n, 3, h, w).
            Values should be in range [0, 1] and in float format.
        y_only (bool, optional): If True, returns only Y channel. Defaults to False.

    Returns:
        torch.Tensor: YCbCr images with shape (n, 3, h, w) if y_only=False,
            or (n, 1, h, w) if y_only=True. Values are in range [0, 1].

    Note:
        The conversion matrix and constants are scaled by 255 since input is [0,1].
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = (
            torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
        )
    else:
        weight = torch.tensor(
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ]
        ).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = (
            torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
        )

    out_img = out_img / 255.0
    return out_img


def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images using PyTorch.

    PSNR is a metric that measures the quality of reconstructed images. It is calculated
    as the ratio between the maximum possible power of a signal and the power of corrupting
    noise that affects the fidelity of its representation.

    The formula used is:
    PSNR = 10 * log10(MAX^2 / MSE)
    where MAX is the maximum possible pixel value (1.0 in this case) and MSE is the
    mean squared error between the images.

    Args:
        img (torch.Tensor): First image with shape (n, 3/1, h, w).
            Values should be in range [0, 1].
        img2 (torch.Tensor): Second image with shape (n, 3/1, h, w).
            Values should be in range [0, 1].
        crop_border (int): Number of pixels to crop from each border of the images.
            These pixels are excluded from the PSNR calculation.
        test_y_channel (bool, optional): If True, converts images to YCbCr and
            calculates PSNR on Y channel only. Defaults to False.

    Returns:
        torch.Tensor: PSNR values for each image in the batch.

    Raises:
        AssertionError: If input images have different shapes.

    Note:
        A small epsilon (1e-8) is added to the MSE to avoid log(0) errors.
    """
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))

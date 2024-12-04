from typing import List, Dict
import random
import math
import os

import numpy as np
from PIL import Image
import cv2
import polars as pl
import torch
from torch.nn import functional as F

from .diffjpeg import DiffJPEG


def load_file_list(file_list_path: str) -> List[Dict[str, str]]:
    """Load a list of image files from a text file.

    Args:
        file_list_path (str): Path to a text file containing image file paths, one per line

    Returns:
        List[Dict[str, str]]: List of dictionaries, each containing:
            - image_path: Path to the image file
            - prompt: Empty string placeholder for prompt
    """
    files = []
    with open(file_list_path, "r") as fin:
        for line in fin:
            path = line.strip()
            if path:
                files.append({"image_path": path, "prompt": ""})
    return files


def load_file_metas(file_metas: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Load image metadata from parquet files.

    Args:
        file_metas (List[Dict[str, str]]): List of dictionaries containing:
            - file_list: Path to parquet file
            - image_path_key: Column name for image path
            - short_prompt_key: Column name for short prompt
            - long_prompt_key: Column name for long prompt

    Returns:
        List[Dict[str, str]]: List of dictionaries, each containing:
            - image_path: Path to the image file
            - short_prompt: Short text prompt
            - long_prompt: Long text prompt
    """
    files = []
    for file_meta in file_metas:
        file_list_path = file_meta["file_list"]
        image_path_key = file_meta["image_path_key"]
        short_prompt_key = file_meta["short_prompt_key"]
        long_prompt_key = file_meta["long_prompt_key"]
        ext = os.path.splitext(file_list_path)[1].lower()
        assert ext == ".parquet", f"only support parquet format"
        df = pl.read_parquet(file_list_path)
        for row in df.iter_rows(named=True):
            files.append(
                {
                    "image_path": row[image_path_key],
                    "short_prompt": row[short_prompt_key],
                    "long_prompt": row[long_prompt_key],
                }
            )
    return files


def center_crop_arr(pil_image, image_size):
    """Center crop a PIL image to a specified size.
    
    Adapted from: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py

    The image is first downsampled using box filtering if it's more than 2x the target size,
    then resized to slightly larger than target using bicubic interpolation,
    and finally center cropped to exact target size.

    Args:
        pil_image (PIL.Image): Input PIL image
        image_size (int): Target size for both height and width

    Returns:
        np.ndarray: Center cropped image array of shape (image_size, image_size, channels)
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """Randomly crop a PIL image to a specified size.
    
    Adapted from: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py

    The image is first resized so its smaller dimension is randomly sized between
    min_crop_frac and max_crop_frac times the target size, then randomly cropped
    to the target size.

    Args:
        pil_image (PIL.Image): Input PIL image
        image_size (int): Target size for both height and width
        min_crop_frac (float): Minimum fraction of target size to use for initial resize
        max_crop_frac (float): Maximum fraction of target size to use for initial resize

    Returns:
        np.ndarray: Randomly cropped image array of shape (image_size, image_size, channels)
    """
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment images with horizontal flips and rotations.
    
    Adapted from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py

    Applies random horizontal flips and/or rotations (0, 90, 180, 270 degrees) to images.
    Can also handle optical flow maps, adjusting the flow vectors accordingly.

    Args:
        imgs (list[ndarray] | ndarray): Input images to augment
        hflip (bool): Whether to apply random horizontal flips
        rotation (bool): Whether to apply random rotations
        flows (list[ndarray] | ndarray, optional): Optical flow maps to augment
        return_status (bool): Whether to return the applied augmentation status

    Returns:
        list[ndarray] | ndarray: Augmented images
        list[ndarray] | ndarray: Augmented flows (if flows provided)
        tuple[bool]: (hflip, vflip, rot90) statuses (if return_status=True)
    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D. Applies 2D convolution with the given kernel.

    Args:
        img (Tensor): Input image tensor of shape (b, c, h, w), where:
            b: batch size
            c: number of channels 
            h: height
            w: width
        kernel (Tensor): Convolution kernel of shape (b, k, k), where:
            b: batch size (can be 1 to apply same kernel to all images)
            k: kernel size (must be odd)

    Returns:
        Tensor: Filtered image tensor of shape (b, c, h, w)

    Raises:
        ValueError: If kernel size is even
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        # img: torch.Tensor
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(torch.nn.Module):
    """Unsharp Masking (USM) sharpening filter implemented as a PyTorch module.
    
    This filter sharpens images by subtracting a blurred version from the original
    and adding back a weighted version of the high-frequency residual details.
    
    Args:
        radius (int): Radius of Gaussian blur kernel. Must be odd. Default: 50
        sigma (float): Standard deviation for Gaussian kernel. Default: 0
            If 0, calculated based on radius.
            
    Attributes:
        radius (int): Stored radius value
        kernel (Tensor): Pre-computed Gaussian blur kernel
    """

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, img, weight=0.5, threshold=10):
        """Apply USM sharpening to input image.
        
        Args:
            img (Tensor): Input image tensor of shape (b, c, h, w)
            weight (float): Weight of sharpening effect. Default: 0.5
            threshold (float): Minimum difference for sharpening. Default: 10
                Only differences > threshold/255 are sharpened.
                
        Returns:
            Tensor: Sharpened image tensor of shape (b, c, h, w)
        """
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img

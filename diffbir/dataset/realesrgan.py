from typing import Dict, Sequence, Mapping, Any, Optional, List
import math
import random
import time
import io

import numpy as np
import torch
from torch.utils import data
from PIL import Image

from .degradation import circular_lowpass_kernel, random_mixed_kernels
from .utils import augment, random_crop_arr, center_crop_arr, load_file_metas
from ..utils.common import instantiate_from_config


class RealESRGANDataset(data.Dataset):
    """Dataset that generates degraded images following the RealESRGAN approach.
    
    This dataset loads high quality images and applies a series of degradations to simulate
    low quality images. The degradation process includes:
    1. Loading and cropping high quality images
    2. Applying data augmentation (horizontal flip and rotation)
    3. Generating and applying two stages of blur kernels
    4. Optionally applying a final sinc filter
    
    The dataset also handles text prompts associated with the images, with options for
    using short or long prompts.

    Args:
        file_metas (List[Dict[str, str]]): List of image metadata containing paths and prompts
        p_long_prompt (float): Probability of using long prompt instead of short prompt
        file_backend_cfg (Mapping[str, Any]): Configuration for file backend system
        out_size (int): Output image size (both height and width)
        crop_type (str): Type of cropping - 'none', 'center' or 'random'
        use_hflip (bool): Whether to use horizontal flip augmentation
        use_rot (bool): Whether to use rotation augmentation
        blur_kernel_size (int): Size of blur kernel for first degradation stage
        kernel_list (Sequence[str]): List of kernel types for first stage
        kernel_prob (Sequence[float]): Probability of each kernel type for first stage
        blur_sigma (Sequence[float]): Sigma range for Gaussian blur in first stage
        betag_range (Sequence[float]): Beta range for Gaussian kernel in first stage
        betap_range (Sequence[float]): Beta range for Plateau kernel in first stage
        sinc_prob (float): Probability of using sinc kernel in first stage
        blur_kernel_size2 (int): Size of blur kernel for second degradation stage
        kernel_list2 (Sequence[str]): List of kernel types for second stage
        kernel_prob2 (Sequence[float]): Probability of each kernel type for second stage
        blur_sigma2 (Sequence[float]): Sigma range for Gaussian blur in second stage
        betag_range2 (Sequence[float]): Beta range for Gaussian kernel in second stage
        betap_range2 (Sequence[float]): Beta range for Plateau kernel in second stage
        sinc_prob2 (float): Probability of using sinc kernel in second stage
        final_sinc_prob (float): Probability of applying final sinc filter
        p_empty_prompt (float): Probability of using empty prompt
    """

    def __init__(
        self,
        file_metas: List[Dict[str, str]],
        p_long_prompt: float,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool,
        # blur kernel settings of the first degradation stage
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        betag_range: Sequence[float],
        betap_range: Sequence[float],
        sinc_prob: float,
        # blur kernel settings of the second degradation stage
        blur_kernel_size2: int,
        kernel_list2: Sequence[str],
        kernel_prob2: Sequence[float],
        blur_sigma2: Sequence[float],
        betag_range2: Sequence[float],
        betap_range2: Sequence[float],
        sinc_prob2: float,
        final_sinc_prob: float,
        p_empty_prompt: float,
    ) -> "RealESRGANDataset":
        super(RealESRGANDataset, self).__init__()
        self.file_metas = file_metas
        self.image_files = load_file_metas(file_metas)
        self.p_long_prompt = p_long_prompt
        assert (
            0 <= p_long_prompt <= 1
        ), f"p_long_prompt {p_long_prompt} should be a probability between [0, 1]"
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]

        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.betag_range = betag_range
        self.betap_range = betap_range
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2

        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob

        self.use_hflip = use_hflip
        self.use_rot = use_rot

        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        self.p_empty_prompt = p_empty_prompt

    def load_gt_image(
        self, image_path: str, max_retry: int = 5
    ) -> Optional[np.ndarray]:
        """Load and process a ground truth image.
        
        Args:
            image_path (str): Path to the image file
            max_retry (int): Maximum number of retries for loading image
            
        Returns:
            Optional[np.ndarray]: Loaded and processed image array, or None if loading fails
        """
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            try:
                image_bytes = self.file_backend.get(image_path)
            except:
                # file does not exist
                return None
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)

        try:
            # failed to decode image bytes
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            return None

        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.
        
        This method:
        1. Loads a high quality image
        2. Applies data augmentation
        3. Generates degradation kernels
        4. Prepares the associated text prompt
        
        Args:
            index (int): Index of the sample to get
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'hq': High quality image tensor
                - 'kernel1': First degradation kernel
                - 'kernel2': Second degradation kernel
                - 'sinc_kernel': Final sinc kernel
                - 'txt': Associated text prompt
        """
        # -------------------------------- Load hq images -------------------------------- #
        # load gt image
        img_gt = None
        while img_gt is None:
            # load meta file
            image_file = self.image_files[index]
            gt_path = image_file["image_path"]
            p = np.random.uniform()
            if p < self.p_long_prompt:
                prompt = image_file["long_prompt"]
            else:
                prompt = image_file["short_prompt"]
            img_gt = self.load_gt_image(gt_path)
            if img_gt is None:
                print(f"failed to load {gt_path}, try another image")
                index = random.randint(0, len(self) - 1)

        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
        img_hq = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        if np.random.uniform() < self.p_empty_prompt:
            prompt = ""

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(img_hq[..., ::-1].transpose(2, 0, 1).copy()).float()
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return {
            "hq": img_hq,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "txt": prompt,
        }

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.image_files)

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import (
    BSRNetPipeline,
    SwinIRPipeline,
)
from ..model import RRDBNet, SwinIR


class BSRInferenceLoop(InferenceLoop):
    """Inference loop for blind super-resolution models.
    
    This class handles loading and running inference with different versions of BSR models:
    - v1: Uses SwinIR with general weights
    - v2: Uses BSRNet 
    - v2.1: Uses SwinIR with RealESRGAN weights
    
    The loop manages loading the appropriate cleaner model and pipeline based on the version.
    """

    def load_cleaner(self) -> None:
        """Loads and initializes the cleaner model based on version.
        
        The cleaner model is either SwinIR or BSRNet depending on the version:
        - v1: SwinIR with general weights
        - v2: BSRNet
        - Other: SwinIR with RealESRGAN weights
        
        The model is loaded from a config file and pretrained weights are downloaded.
        The model is moved to the specified device and put in eval mode.
        """
        if self.args.version == "v1":
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_general"]
        elif self.args.version == "v2":
            config = "configs/inference/bsrnet.yaml"
            weight = MODELS["bsrnet"]
        else:
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_realesrgan"]
        self.cleaner: RRDBNet | SwinIR = instantiate_from_config(OmegaConf.load(config))
        model_weight = load_model_from_url(weight)
        self.cleaner.load_state_dict(model_weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        """Initializes the appropriate pipeline based on version.
        
        Creates either a SwinIRPipeline or BSRNetPipeline depending on version:
        - v1/v2.1: SwinIRPipeline
        - Other: BSRNetPipeline
        
        The pipeline is initialized with the cleaner model, conditional diffusion model,
        diffusion model, conditioning function and device.
        """
        if self.args.version == "v1" or self.args.version == "v2.1":
            self.pipeline = SwinIRPipeline(
                self.cleaner,
                self.cldm,
                self.diffusion,
                self.cond_fn,
                self.args.device,
            )
        else:
            self.pipeline = BSRNetPipeline(
                self.cleaner,
                self.cldm,
                self.diffusion,
                self.cond_fn,
                self.args.device,
                self.args.upscale,
            )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        """Post-processes the loaded low quality image.
        
        For v1 and v2.1, upscales the image using bicubic interpolation before processing.
        
        Args:
            lq: Low quality input image
            
        Returns:
            Processed image as numpy array
        """
        if self.args.version == "v1" or self.args.version == "v2.1":
            lq = lq.resize(
                tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
            )
        return super().after_load_lq(lq)

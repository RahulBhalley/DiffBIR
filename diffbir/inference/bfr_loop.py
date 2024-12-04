import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..pipeline import SwinIRPipeline
from ..model import SwinIR


class BFRInferenceLoop(InferenceLoop):
    """Blind Face Restoration inference loop.
    
    This class implements the inference pipeline for blind face restoration,
    which includes loading a SwinIR model for cleaning/restoration and setting up
    the full inference pipeline.
    """

    def load_cleaner(self) -> None:
        """Load and initialize the SwinIR face restoration model.
        
        Loads the SwinIR model configuration from yaml, downloads pretrained weights,
        and moves the model to the specified device. The model is set to evaluation mode.
        """
        self.cleaner: SwinIR = instantiate_from_config(
            OmegaConf.load("configs/inference/swinir.yaml")
        )
        weight = load_model_from_url(MODELS["swinir_face"])
        self.cleaner.load_state_dict(weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        """Initialize the full SwinIR restoration pipeline.
        
        Creates a SwinIRPipeline instance that combines the cleaner model with
        the conditional diffusion model and other components.
        """
        self.pipeline = SwinIRPipeline(
            self.cleaner, self.cldm, self.diffusion, self.cond_fn, self.args.device
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        """Post-process the loaded low quality image.
        
        Args:
            lq (Image.Image): The loaded low quality input image
            
        Returns:
            np.ndarray: The processed image after resizing
            
        This method upscales the input image using bicubic interpolation based on
        the specified upscale factor before passing it to the parent class processing.
        """
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)

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
    SwinIRPipeline,
    SCUNetPipeline,
)
from ..model import SwinIR, SCUNet


class BIDInferenceLoop(InferenceLoop):
    """Blind Image Deblurring inference loop.
    
    This class handles the inference pipeline for blind image deblurring tasks.
    It loads appropriate models and pipelines based on the specified version
    and processes images through them.
    """

    def load_cleaner(self) -> None:
        """Load and initialize the image cleaning model.
        
        Based on the version argument, loads either SwinIR or SCUNet model
        with appropriate configurations and weights. The model is moved to
        the specified device and set to evaluation mode.
        
        Version mappings:
        - v1: SwinIR with general weights
        - v2: SCUNet with PSNR-optimized weights  
        - others: SwinIR with Real-ESRGAN weights
        """
        if self.args.version == "v1":
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_general"]
        elif self.args.version == "v2":
            config = "configs/inference/scunet.yaml"
            weight = MODELS["scunet_psnr"]
        else:
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_realesrgan"]
        self.cleaner: SCUNet | SwinIR = instantiate_from_config(OmegaConf.load(config))
        model_weight = load_model_from_url(weight)
        self.cleaner.load_state_dict(model_weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        """Initialize the appropriate inference pipeline.
        
        Selects and instantiates either SwinIR or SCUNet pipeline based on
        the version argument. The pipeline combines the cleaner model with
        other components needed for inference.
        
        Version mappings:
        - v1 or v2.1: SwinIRPipeline
        - others: SCUNetPipeline
        """
        if self.args.version == "v1" or self.args.version == "v2.1":
            pipeline_class = SwinIRPipeline
        else:
            pipeline_class = SCUNetPipeline
        self.pipeline = pipeline_class(
            self.cleaner,
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        """Post-process the loaded low quality image.
        
        Args:
            lq (Image.Image): Input low quality image
            
        Returns:
            np.ndarray: Processed image array after upscaling
            
        The image is upscaled using bicubic interpolation according to the
        specified upscale factor before being processed by the parent class.
        """
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)

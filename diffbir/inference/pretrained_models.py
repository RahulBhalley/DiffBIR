"""Pretrained model weights and configurations for DiffBIR.

This module defines the URLs and configurations for all pretrained models used in DiffBIR inference.
The models are organized by version (v1, v2, v2.1) and task type (BSR, BFR, BID).

Each version has a specific architecture and training setup:

DiffBIR-v1:
    All tasks use pretrained stable diffusion v2.1 (sd_v2.1) as the base model.
    
    BSR (Blind Super Resolution):
        - Stage 1: SwinIR model trained on ImageNet-1k with Real-ESRGAN degradation
        - Stage 2: IRControlNet trained on ImageNet-1k
    
    BFR (Blind Face Restoration): 
        - Stage 1: SwinIR pretrained on FFHQ dataset (from DifFace project)
        - Stage 2: IRControlNet trained specifically on FFHQ face dataset
    
    BID (Blind Image Deblurring):
        Uses same models as BSR task

DiffBIR-v2:
    All tasks share:
        - Pretrained stable diffusion v2.1 base model
        - Common stage 2 IRControlNet model
    
    Task-specific stage 1 models:
        BSR: BSRNet from BSRGAN project
        BFR: FFHQ-pretrained SwinIR from DifFace
        BID: SCUNet-PSNR from SCUNet project

DiffBIR-v2.1:
    All tasks use:
        - Pretrained stable diffusion v2.1 with ZSNR optimization
        - Common stage 2 IRControlNet v2.1 model
    
    Task-specific stage 1 models:
        BSR: SwinIR trained on ImageNet-1k with Real-ESRGAN degradation
        BFR: FFHQ-pretrained SwinIR from DifFace
        BID: Same as BSR task

Model URLs:
    Stage 1 Models:
        - BSRNet: Original BSRGAN implementation
        - SwinIR variants: Face-specific and general versions
        - SCUNet: PSNR-optimized version
    
    Base Models:
        - Stable Diffusion 2.1: Original and ZSNR-optimized versions
    
    IRControlNet:
        - Version-specific weights for face and general tasks
"""

MODELS = {
    # Stage 1 model weights - Image preprocessing networks
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    # Note: Using legacy face model version for paper consistency
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    "swinir_realesrgan": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth",
    
    # Base Stable Diffusion model weights
    "sd_v2.1": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "sd_v2.1_zsnr": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt",
    
    # IRControlNet weights - Task and version specific
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth", 
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth",
    "v2.1": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt",
}

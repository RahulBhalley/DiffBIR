import os
import json
import requests
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

def download_file(url: str, dest_path: str, desc: Optional[str] = None) -> None:
    """Download a file from URL to destination path with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    desc = desc or os.path.basename(dest_path)
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)
    
    with open(dest_path, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()

def fetch_diffbir_models(version: str = "v2", task: str = "sr"):
    """Fetch DiffBIR models from various sources based on version and task."""
    # Define all available models
    models = {
        # Stage 1 model weights
        "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
        "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
        "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
        "swinir_realesrgan": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth",
        
        # Base Stable Diffusion model weights
        "sd_v2.1": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
        "sd_v2.1_zsnr": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt",
        
        # IRControlNet weights
        "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
        "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
        "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth",
        "v2.1": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt"
    }
    
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Determine which models to download based on version and task
    required_models = []
    
    # Base SD model
    if version == "v2.1":
        required_models.append(("sd_v2.1_zsnr", models["sd_v2.1_zsnr"]))
    else:
        required_models.append(("sd_v2.1", models["sd_v2.1"]))
    
    # Stage 1 model based on version and task
    if version == "v1":
        if task in ["face", "unaligned_face"]:
            required_models.extend([
                ("swinir_face", models["swinir_face"]),
                ("v1_face", models["v1_face"])
            ])
        elif task in ["sr", "denoise"]:
            required_models.extend([
                ("swinir_general", models["swinir_general"]),
                ("v1_general", models["v1_general"])
            ])
    elif version == "v2":
        if task == "sr":
            required_models.extend([
                ("bsrnet", models["bsrnet"]),
                ("v2", models["v2"])
            ])
        elif task in ["face", "unaligned_face"]:
            required_models.extend([
                ("swinir_face", models["swinir_face"]),
                ("v2", models["v2"])
            ])
        elif task == "denoise":
            required_models.extend([
                ("scunet_psnr", models["scunet_psnr"]),
                ("v2", models["v2"])
            ])
    else:  # v2.1
        if task == "sr" or task == "denoise":
            required_models.extend([
                ("swinir_realesrgan", models["swinir_realesrgan"]),
                ("v2.1", models["v2.1"])
            ])
        elif task in ["face", "unaligned_face"]:
            required_models.extend([
                ("swinir_face", models["swinir_face"]),
                ("v2.1", models["v2.1"])
            ])
    
    # Download required models
    for model_name, url in required_models:
        dest_path = weights_dir / f"{model_name}.pth"
        if not dest_path.exists():
            print(f"Downloading {model_name}...")
            download_file(url, str(dest_path))
        else:
            print(f"Model {model_name} already exists, skipping...")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, choices=["v1", "v2", "v2.1"], default="v2")
    parser.add_argument("--task", type=str, choices=["sr", "face", "denoise", "unaligned_face"], default="sr")
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("weights", exist_ok=True)
    
    # Fetch DiffBIR models based on version and task
    print(f"Fetching DiffBIR models for version {args.version}, task {args.task}...")
    fetch_diffbir_models(args.version, args.task)

if __name__ == "__main__":
    main()

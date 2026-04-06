#!/usr/bin/env python3
"""
Download and setup ArcFace embedding model for robust real-world conditions.

Supported models:
1. InsightFace buffalo_s (balanced, ~500MB, 112x112)
2. InsightFace r50 (accurate, ~700MB, 112x112)
3. MobileNet v3 + ArcFace lightweight (~50MB)

For production robustness across diverse lighting/distance/weather conditions,
buffalo_s or r50 are recommended.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_CONFIGS = {
    "buffalo_s": {
        "name": "InsightFace Buffalo-S (Balanced)",
        "description": "Moderate size, good accuracy, handles diverse conditions",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        "size_mb": 500,
        "input_size": 112,
        "requires_unzip": True,
    },
    "buffalo_l": {
        "name": "InsightFace Buffalo-L (Maximum Accuracy)",
        "description": "Larger model, best accuracy, recommended for fixed roster",
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "size_mb": 700,
        "input_size": 112,
        "requires_unzip": True,
    },
    "mobilenet_arcface": {
        "name": "MobileNet v3 + ArcFace (Lightweight)",
        "description": "Ultra-lightweight, real-time on CPU, decent accuracy",
        "url": "https://github.com/uyuyu/insightface/releases/download/v0.5.1/mobilenet_1.onnx",
        "size_mb": 50,
        "input_size": 112,
        "requires_unzip": False,
    },
}


def download_file(url: str, output_path: str, label: str = "file") -> None:
    """Download file with progress bar."""
    print(f"[DOWNLOAD] {label}")
    print(f"           URL: {url}")
    print(f"           Target: {output_path}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, 100.0 * downloaded / total_size)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r           Progress: {pct:6.1f}% ({mb_done:.1f}MB / {mb_total:.1f}MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print(f"\r[OK] Downloaded to {output_path}\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to download: {e}")
        raise


def setup_insightface_model(model_key: str, output_dir: str) -> None:
    """Download and extract InsightFace model."""
    cfg = MODEL_CONFIGS[model_key]
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{model_key}.zip")
    download_file(cfg["url"], output_path, label=cfg["name"])
    
    if cfg["requires_unzip"]:
        import zipfile
        print(f"[EXTRACT] Extracting {model_key}...")
        with zipfile.ZipFile(output_path, "r") as zf:
            zf.extractall(output_dir)
        os.remove(output_path)
        print(f"[OK] Extracted to {output_dir}\n")


def setup_metadata(output_dir: str, model_key: str) -> None:
    """Create model metadata JSON."""
    cfg = MODEL_CONFIGS[model_key]
    metadata = {
        "model_key": model_key,
        "name": cfg["name"],
        "description": cfg["description"],
        "input_size": cfg["input_size"],
        "embedding_dim": 512,  # Standard ArcFace output dimension
        "model_type": "arcface",
    }
    
    metadata_path = os.path.join(output_dir, "model_config.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Created metadata: {metadata_path}\n")


def list_models() -> None:
    """Print available models."""
    print("\n[AVAILABLE MODELS]\n")
    for key, cfg in MODEL_CONFIGS.items():
        print(f"{key:20} | {cfg['name']:40} | {cfg['size_mb']:4}MB")
        print(f"{' '*20} | {cfg['description']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and setup ArcFace embedding models for diverse real-world conditions."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="buffalo_s",
        help="Model to download (default: buffalo_s for balanced speed/accuracy).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "models" / "arcface_mobilenet"),
        help="Output directory for model files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit.",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    print(f"\n[SETUP] ArcFace Embedding Model")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Config: {MODEL_CONFIGS[args.model]['name']}")
    print(f"[INFO] Size: {MODEL_CONFIGS[args.model]['size_mb']}MB")
    print()
    
    setup_insightface_model(args.model, args.output_dir)
    setup_metadata(args.output_dir, args.model)
    
    print(f"[SUCCESS] Model ready at: {args.output_dir}")
    print(f"[NEXT] Run trainer: python src/arcface_mobilenet/trainer.py --model-dir {args.output_dir}")


if __name__ == "__main__":
    main()

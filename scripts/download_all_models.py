#!/usr/bin/env python3
"""Download *all* runtime weights for EAIW into E:\\EAIW\\models.

This is the one-command entry point for a fresh clone:

    python scripts/download_all_models.py

Downloads (existing files are skipped / verified):
  - GoogLeNet, VGG-19, LaMa, CycleGAN x2, Real-ESRGAN, VQGAN (historical)
  - ModelScope text-to-video-1.7B (Diffusers fp16, 12 files)
  - Stable Diffusion 1.4 Diffusers weights (lab 05 stand-in for gated SD1.3)
  - Qwen3-VL-2B-Instruct (Transformer Dream)
  - CLIP ViT-L/14 (optional Transformer Dream semantic mode)

Nothing is written to C:.  The script sets HF_HOME/TEMP to E: before importing
the downloaders.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
os.environ.setdefault("TEMP", str(ROOT / "temp"))
os.environ.setdefault("TMP", str(ROOT / "temp"))
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"


def run_historical() -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "download_historical_weights.py"), "--all", "--jobs", "4"]
    print("[1/4] historical weights + ModelScope T2V")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def run_sd() -> None:
    print("[2/4] Stable Diffusion 1.4 Diffusers")
    from models.sd_downloader import ensure_sd_weights

    ensure_sd_weights(jobs=2)


def run_qwen() -> None:
    print("[3/4] Qwen3-VL-2B-Instruct")
    from transformer_dream.download_model import main as qwen_main

    qwen_main()


def run_clip() -> None:
    print("[4/4] CLIP ViT-L/14")
    try:
        from transformer_dream.download_clip import main as clip_main

        clip_main()
    except Exception as exc:  # noqa: BLE001
        print(f"CLIP optional download failed: {exc}")


def main() -> int:
    print("EAIW model downloader")
    print(f"root: {ROOT}")
    print(f"cache/temp: {os.environ['HF_HOME']} / {os.environ['TEMP']}")
    print("C: is not used for model files.\n")
    run_historical()
    run_sd()
    run_qwen()
    run_clip()
    print("\nAll downloads finished.  Start: python main.py --server")
    print("Transformer Dream: python -m transformer_dream.server --port 7866")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

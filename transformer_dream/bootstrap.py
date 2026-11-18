"""Transformer Dream Engine — lightweight Qwen3-VL-2B feature visualizer.

All caches and temporary files deliberately live on E: because this machine's
C: drive is nearly full.  Import this module *before* torch/transformers.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "models" / "hf_cache"
TEMP = ROOT / "temp"
OUTPUTS = ROOT / "outputs" / "transformer_dream"
QWEN_DIR = ROOT / "models" / "qwen3-vl-2b-instruct"

for p in (CACHE, TEMP, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch_cache"))
os.environ.setdefault("TEMP", str(TEMP))
os.environ.setdefault("TMP", str(TEMP))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(ROOT / "__pycache__"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

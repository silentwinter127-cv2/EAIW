from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QWEN_DIR = ROOT / "models" / "qwen3-vl-2b-instruct"
OUTPUTS = ROOT / "outputs" / "transformer_dream"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

PATCH_SIZE = 16
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
CHANNELS = 3

DEFAULT_PROMPT = "a cat wearing a crown"
DEFAULT_NEGATIVE = "a plain background"
DEFAULT_STEPS = 240
DEFAULT_SIZE = 256
DEFAULT_LR = 0.02
DEFAULT_TV = 2.0

M_LAYERS = 28
V_LAYERS = 24

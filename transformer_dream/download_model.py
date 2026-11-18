r"""Download Qwen3-VL-2B-Instruct to E: (never C:).

    python -m transformer_dream.download_model

The download is ~4.3 GB.  It uses E:\EAIW\models\hf_cache as the HF cache and
writes the model itself to E:\EAIW\models\qwen3-vl-2b-instruct.
"""
from __future__ import annotations

from . import bootstrap  # noqa: F401

import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from .config import QWEN_DIR


def main() -> None:
    from huggingface_hub import snapshot_download

    QWEN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Qwen3-VL-2B-Instruct -> {QWEN_DIR}")
    snapshot_download(
        "Qwen/Qwen3-VL-2B-Instruct",
        local_dir=str(QWEN_DIR),
        max_workers=4,
    )
    print("done")


if __name__ == "__main__":
    main()

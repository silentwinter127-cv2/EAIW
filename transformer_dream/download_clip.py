"""Download CLIP ViT-L/14 to E: for the optional semantic mode."""
from __future__ import annotations

from . import bootstrap  # noqa: F401

import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"


def main() -> None:
    from huggingface_hub import snapshot_download
    from .clip_runtime import CLIP_DIR

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CLIP ViT-L/14 -> {CLIP_DIR}")
    snapshot_download(
        "openai/clip-vit-large-patch14",
        local_dir=str(CLIP_DIR),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.py"],
        max_workers=4,
    )
    print("done")


if __name__ == "__main__":
    main()

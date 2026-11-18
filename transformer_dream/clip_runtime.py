from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import ROOT

CLIP_DIR = ROOT / "models" / "clip-vit-large-patch14"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipRuntime:
    """Optional CLIP ViT-L/14 semantic prior.

    CLIP is loaded *instead of* Qwen (the engine unloads Qwen first), so the
    combined footprint never exceeds one model on a 12GB card.
    """

    _instance: Optional["ClipRuntime"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.load_seconds = None

    @classmethod
    def get(cls) -> "ClipRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not CLIP_DIR.is_dir():
            raise FileNotFoundError(
                f"CLIP ViT-L/14 not found at {CLIP_DIR}. "
                "Download it to E: with transformer_dream/download_clip.py"
            )
        from transformers import CLIPModel, CLIPProcessor

        import time

        t0 = time.time()
        self.processor = CLIPProcessor.from_pretrained(str(CLIP_DIR), local_files_only=True)
        self.model = CLIPModel.from_pretrained(
            str(CLIP_DIR), dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True
        ).to("cuda").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.load_seconds = round(time.time() - t0, 2)
        torch.cuda.empty_cache()

    def text_embed(self, prompts: List[str]) -> torch.Tensor:
        self.load()
        z = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            out = self.model.get_text_features(**z)
            e = (out.text_embeds if hasattr(out, "text_embeds") else out.pooler_output).float()
        e = e / e.norm(dim=-1, keepdim=True)
        e = e.mean(dim=0)
        return e / (e.norm() + 1e-8)

    def image_embed(self, image: torch.Tensor, size: int = 224) -> torch.Tensor:
        self.load()
        x = F.interpolate(image, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
        mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        out = self.model.get_image_features(pixel_values=x)
        e = (out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output).float()
        return e / (e.norm(dim=-1, keepdim=True) + 1e-8)

    def status(self) -> Dict[str, Any]:
        info = {
            "clip_exists": CLIP_DIR.is_dir(),
            "clip_loaded": self.loaded,
            "clip_dir": str(CLIP_DIR),
            "clip_load_seconds": self.load_seconds,
        }
        if self.loaded and torch.cuda.is_available():
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        return info

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

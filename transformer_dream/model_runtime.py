from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .config import M_LAYERS, QWEN_DIR, V_LAYERS
from .image_ops import patchify


class QwenRuntime:
    """One warm Qwen3-VL-2B instance, frozen except for the input pixels."""

    _instance: Optional["QwenRuntime"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.image_token_id = None
        self.visual_blocks = []
        self.visual_layers = V_LAYERS
        self.text_layers = M_LAYERS
        self.load_seconds = None

    @classmethod
    def get(cls) -> "QwenRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not QWEN_DIR.is_dir():
            raise FileNotFoundError(
                f"Qwen3-VL-2B not found at {QWEN_DIR}. "
                "Download it to E: (never C:): "
                "python -c \"from huggingface_hub import snapshot_download; "
                "snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir=r'E:\\EAIW\\models\\qwen3-vl-2b-instruct')\""
            )
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(str(QWEN_DIR), local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(QWEN_DIR),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        self.model = self.model.to("cuda").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.config.use_cache = False
        self.image_token_id = int(self.model.config.image_token_id)
        self.visual_blocks = list(self.model.model.visual.blocks)
        self.visual_layers = len(self.visual_blocks)
        self.text_layers = len(self.model.model.language_model.layers)
        self.load_seconds = round(time.time() - t0, 2)
        torch.cuda.empty_cache()

    def prepare_inputs(self, prompt: str, size: int = 256) -> Dict[str, Any]:
        self.load()
        dummy = Image.new("RGB", (int(size), int(size)), (128, 128, 128))
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        z = self.processor(text=[text], images=[dummy], return_tensors="pt")
        out = {k: v.to("cuda") for k, v in z.items() if k != "pixel_values"}
        ids = out["input_ids"][0]
        pos = (ids == self.image_token_id).nonzero().flatten()
        out["_image_positions"] = pos
        out["_grid"] = [int(out["image_grid_thw"][0, 1]), int(out["image_grid_thw"][0, 2])]
        out["_prompt_pos"] = self.last_prompt_index(ids)
        return out

    def last_prompt_index(self, ids: torch.Tensor) -> int:
        """Index of the last user-prompt token (before <|im_end|>)."""
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        hits = (ids == im_end).nonzero().flatten()
        if hits.numel():
            return max(0, int(hits[0]) - 1)
        return int(ids.numel() - 1)

    def forward(self, pixel_values: torch.Tensor, inputs: Dict[str, Any], hidden: bool = False):
        kwargs: Dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": pixel_values,
            "image_grid_thw": inputs["image_grid_thw"],
            "use_cache": False,
            "return_dict": True,
        }
        if inputs.get("mm_token_type_ids") is not None:
            kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        if hidden:
            kwargs["output_hidden_states"] = True
        return self.model(**kwargs)

    def text_prompt_vector(self, prompt: str, layer: int) -> torch.Tensor:
        self.load()
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        z = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(
                input_ids=z["input_ids"],
                attention_mask=z["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        pos = self.last_prompt_index(z["input_ids"][0])
        return out.hidden_states[layer][0, pos].float()

    def text_hidden(self, prompt: str):
        self.load()
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        z = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model(
                input_ids=z["input_ids"],
                attention_mask=z["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return out.hidden_states

    def readout(self, prompt: str, image: torch.Tensor, layer: int = 20) -> Dict[str, Any]:
        """One forward pass at the final image, returning lens tokens and stats."""
        self.load()
        size = int(image.shape[-1])
        inputs = self.prepare_inputs(prompt, size)
        pix = patchify(image)
        with torch.no_grad():
            out = self.forward(pix, inputs, hidden=True)
        hs = out.hidden_states
        from .lens import lens_for_layer, lens_image_tokens

        return {
            "last": lens_for_layer(self.model, hs, self.tokenizer, layer),
            "image": lens_image_tokens(self.model, hs, self.tokenizer, layer, inputs["_image_positions"]),
        }

    def vision_stats(self, prompt: str, image: torch.Tensor, layers: List[int]) -> List[Dict[str, Any]]:
        self.load()
        inputs = self.prepare_inputs(prompt, int(image.shape[-1]))
        captures: Dict[int, torch.Tensor] = {}
        handles = []
        for l in layers:
            idx = int(l) - 1
            if idx < 0 or idx >= len(self.visual_blocks):
                continue

            def hook(_module, _args, output, idx=idx):
                x = output[0] if isinstance(output, (tuple, list)) else output
                captures[idx] = x.detach().float()

            handles.append(self.visual_blocks[idx].register_forward_hook(hook))
        try:
            with torch.no_grad():
                self.forward(patchify(image), inputs)
        finally:
            for h in handles:
                h.remove()
        out = []
        for l in layers:
            idx = int(l) - 1
            if idx in captures:
                x = captures[idx]
                out.append({
                    "layer": int(l),
                    "shape": list(x.shape),
                    "mean": round(float(x.mean()), 4),
                    "std": round(float(x.std()), 4),
                    "mean_abs": round(float(x.abs().mean()), 4),
                })
        return out

    def status(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "model": "Qwen3-VL-2B-Instruct",
            "model_dir": str(QWEN_DIR),
            "model_exists": QWEN_DIR.is_dir(),
            "loaded": self.loaded,
            "text_layers": self.text_layers,
            "vision_layers": self.visual_layers,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "load_seconds": self.load_seconds,
        }
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info.update({
                "gpu": torch.cuda.get_device_name(0),
                "vram_free_gb": round(free / 2**30, 2),
                "vram_total_gb": round(total / 2**30, 2),
                "vram_allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2) if self.loaded else 0.0,
            })
        return info

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.processor = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

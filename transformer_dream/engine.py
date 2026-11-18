from __future__ import annotations

import base64
import io
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import (
    DEFAULT_LR,
    DEFAULT_NEGATIVE,
    DEFAULT_PROMPT,
    DEFAULT_SIZE,
    DEFAULT_STEPS,
    DEFAULT_TV,
    OUTPUTS,
)
from .image_ops import (
    augment,
    jitter,
    normalize_grad,
    patchify,
    pil_from_tensor,
    save_tensor,
    tensor_to_data_url,
    tv_loss,
    validate_size,
)
from .lens import lens_for_layer, lens_table
from .model_runtime import QwenRuntime
from .clip_runtime import ClipRuntime


class DreamEngine:
    def __init__(self) -> None:
        self.runtime = QwenRuntime.get()
        self.clip = ClipRuntime.get()
        import threading

        self._lock = threading.Lock()

    # ------------------------------------------------------------------ utils
    def _target_tokens(self, word: str) -> List[int]:
        rt = self.runtime
        ids = rt.tokenizer.encode(" " + (word or "crown").strip(), add_special_tokens=False)
        if not ids:
            ids = rt.tokenizer.encode("crown", add_special_tokens=False)
        return ids

    def _direction(self, prompt: str, negative: str, layer: int) -> torch.Tensor:
        rt = self.runtime
        d = rt.text_prompt_vector(prompt, layer) - rt.text_prompt_vector(negative or DEFAULT_NEGATIVE, layer)
        return d / (d.norm() + 1e-6)

    def _init_image(self, size: int, seed: int, init_b64: Optional[str]) -> torch.Tensor:
        g = torch.Generator(device="cuda").manual_seed(int(seed))
        if init_b64:
            raw = base64.b64decode(init_b64.split(",")[-1])
            im = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size), Image.LANCZOS)
            arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
            return arr.to("cuda")
        base = torch.rand((1, 3, size, size), generator=g, device="cuda") * 0.18 + 0.41
        return base

    # ------------------------------------------------------------------- run
    def run(
        self,
        prompt: str = DEFAULT_PROMPT,
        mode: str = "direction",
        layer: int = 24,
        vision_layer: int = 24,
        steps: int = DEFAULT_STEPS,
        size: int = DEFAULT_SIZE,
        lr: float = DEFAULT_LR,
        tv_weight: float = DEFAULT_TV,
        seed: int = 1,
        target_word: str = "crown",
        channel: int = 128,
        negative: str = DEFAULT_NEGATIVE,
        init_image_b64: Optional[str] = None,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            size = validate_size(size)
            steps = max(20, min(300, int(steps)))
            layer = max(1, min(int(layer), 28))
            vision_layer = max(1, min(int(vision_layer), 24))
            lr = max(0.005, min(0.12, float(lr)))
            tv_weight = max(0.0, min(8.0, float(tv_weight)))
            mode = mode if mode in ("direction", "logit", "neuron", "vision", "clip") else "direction"
            if mode == "clip":
                return self._run_clip(
                    prompt=prompt, steps=steps, size=size, lr=lr, tv_weight=tv_weight,
                    seed=seed, init_image_b64=init_image_b64, progress=progress,
                )

            if self.clip.loaded:
                self.clip.unload()
            rt = self.runtime
            rt.load()
            torch.manual_seed(int(seed))
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                if free < 4.0 * 2**30:
                    raise RuntimeError(f"可用显存仅 {free/2**30:.1f} GB，至少需要约 4 GB；请先释放其他 GPU 进程")

            inputs = rt.prepare_inputs(prompt, size)
            image = self._init_image(size, seed, init_image_b64).requires_grad_(True)
            opt = torch.optim.Adam([image], lr=lr)

            target_dir = None
            target_ids: List[int] = []
            if mode == "direction":
                target_dir = self._direction(prompt, negative, layer)
            elif mode == "logit":
                target_ids = self._target_tokens(target_word)
            elif mode == "neuron":
                channel = max(0, min(channel, 2047))

            vision_capture: Dict[str, Any] = {}
            handles = []
            if mode == "vision":
                idx = vision_layer - 1
                block = rt.visual_blocks[idx]

                def hook(_m, _a, output):
                    x = output[0] if isinstance(output, (tuple, list)) else output
                    vision_capture["x"] = x

                handles.append(block.register_forward_hook(hook))

            frames: List[str] = []
            frame_every = max(1, steps // 10)
            t0 = time.time()
            last_loss = None
            try:
                for step in range(steps):
                    opt.zero_grad(set_to_none=True)
                    aug = augment(image, step) if step > 0 else image
                    pix = patchify(aug)
                    out = rt.forward(pix, inputs, hidden=True)
                    if mode == "vision":
                        x = vision_capture["x"]
                        ch = max(0, min(channel, x.shape[-1] - 1))
                        loss = -x[..., ch].mean()
                    else:
                        pos = inputs["_image_positions"]
                        h = out.hidden_states[layer][0, pos].float()
                        if mode == "direction":
                            loss = -F.cosine_similarity(h, target_dir.expand_as(h), dim=-1).mean()
                        elif mode == "neuron":
                            loss = -h[:, channel].mean()
                        else:  # logit
                            dtype = next(rt.model.parameters()).dtype
                            norm = rt.model.model.language_model.norm
                            logits = rt.model.lm_head(norm(h.to(dtype))).mean(dim=0)
                            loss = -logits[target_ids].sum() * 0.02
                    loss = loss + tv_weight * tv_loss(aug)
                    loss.backward()
                    normalize_grad(image)
                    opt.step()
                    with torch.no_grad():
                        image.clamp_(0.0, 1.0)
                    last_loss = float(loss.detach())
                    if step % frame_every == 0 or step == steps - 1:
                        frames.append(tensor_to_data_url(image, "JPEG", 84))
                    if progress and (step % max(1, steps // 12) == 0 or step == steps - 1):
                        progress({
                            "step": step + 1,
                            "steps": steps,
                            "loss": round(last_loss, 4),
                            "elapsed": round(time.time() - t0, 1),
                            "preview": frames[-1] if frames else None,
                        })
                    del out, loss, pix, aug
            finally:
                for h in handles:
                    h.remove()

            elapsed = round(time.time() - t0, 2)
            rt.load()
            with torch.no_grad():
                rinputs = rt.prepare_inputs(prompt, size)
                rout = rt.forward(patchify(image), rinputs, hidden=True)
                hs = rout.hidden_states
                read_last = lens_for_layer(
                    rt.model, hs, rt.tokenizer, layer, positions=[rinputs["_prompt_pos"]]
                )
                read_img = lens_for_layer(
                    rt.model, hs, rt.tokenizer, layer, positions=rinputs["_image_positions"]
                )
                table = [
                    lens_for_layer(rt.model, hs, rt.tokenizer, l, positions=[rinputs["_prompt_pos"]], topk=3)
                    for l in range(1, 29)
                ]

            run_dir = OUTPUTS / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}_L{layer}"
            run_dir.mkdir(parents=True, exist_ok=True)
            final_path = save_tensor(image, run_dir / "final.png")
            for i, fr in enumerate(frames):
                raw = base64.b64decode(fr.split(",")[-1])
                (run_dir / f"frame_{i:02d}.jpg").write_bytes(raw)
            meta = {
                "prompt": prompt,
                "negative": negative,
                "mode": mode,
                "layer": layer,
                "vision_layer": vision_layer,
                "steps": steps,
                "size": size,
                "lr": lr,
                "tv_weight": tv_weight,
                "seed": seed,
                "target_word": target_word,
                "channel": channel,
                "loss": last_loss,
                "elapsed_seconds": elapsed,
                "lens": read_last,
                "lens_image": read_img,
                "model": "Qwen3-VL-2B-Instruct",
                "engine": "transformer_dream",
            }
            (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "ok": True,
                "run_dir": str(run_dir),
                "final": tensor_to_data_url(image, "PNG"),
                "frames": frames,
                "lens": read_last,
                "lens_image": read_img,
                "lens_table": table,
                "meta": meta,
                "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0,
            }

    # ------------------------------------------------------------- clip prior
    def _run_clip(self, prompt: str, steps: int, size: int, lr: float, tv_weight: float,
                  seed: int, init_image_b64: Optional[str],
                  progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        from .clip_runtime import ClipRuntime
        from .image_ops import jitter, pil_from_tensor, save_tensor, tensor_to_data_url, tv_loss

        if self.runtime.loaded:
            self.runtime.unload()
        clip = ClipRuntime.get()
        clip.load()
        torch.manual_seed(int(seed))
        image = self._init_image(size, seed, init_image_b64).requires_grad_(True)
        opt = torch.optim.Adam([image], lr=lr)
        target = clip.text_embed([prompt, f"a photo of {prompt}", f"{prompt}, highly detailed, centered"])
        frames: List[str] = []
        frame_every = max(1, steps // 10)
        t0 = time.time()
        last_loss = None
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            aug = augment(image, step) if step > 0 else image
            if step % 4 == 0:
                s = aug.shape[-1]
                c = int(s * (0.78 + 0.22 * random.random()))
                y = random.randint(0, s - c); x = random.randint(0, s - c)
                aug = F.interpolate(aug[:, :, y:y+c, x:x+c], size=(s, s), mode="bicubic", align_corners=False, antialias=True)
            emb = clip.image_embed(aug, 224)
            loss = -F.cosine_similarity(emb, target.expand_as(emb), dim=-1).mean()
            loss = loss + tv_weight * tv_loss(aug)
            loss.backward(); normalize_grad(image); opt.step()
            with torch.no_grad():
                image.clamp_(0.0, 1.0)
            last_loss = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
            if step % frame_every == 0 or step == steps - 1:
                frames.append(tensor_to_data_url(image, "JPEG", 84))
            if progress and (step % max(1, steps // 12) == 0 or step == steps - 1):
                progress({"step": step + 1, "steps": steps, "loss": round(last_loss, 4),
                          "elapsed": round(time.time() - t0, 1), "preview": frames[-1] if frames else None})
        elapsed = round(time.time() - t0, 2)
        run_dir = OUTPUTS / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_clip"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_tensor(image, run_dir / "final.png")
        for i, fr in enumerate(frames):
            (run_dir / f"frame_{i:02d}.jpg").write_bytes(base64.b64decode(fr.split(",")[-1]))
        meta = {
            "prompt": prompt, "mode": "clip", "steps": steps, "size": size,
            "lr": lr, "tv_weight": tv_weight, "seed": seed, "loss": last_loss,
            "elapsed_seconds": elapsed, "model": "CLIP ViT-L/14", "engine": "transformer_dream",
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "ok": True, "run_dir": str(run_dir), "final": tensor_to_data_url(image, "PNG"),
            "frames": frames, "lens": None, "lens_image": None, "lens_table": [], "meta": meta,
            "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }

    # --------------------------------------------------------------- extract
    def extract(self, prompt: str = DEFAULT_PROMPT, size: int = 256, vision_layers: Optional[List[int]] = None) -> Dict[str, Any]:
        size = validate_size(size)
        if self.clip.loaded:
            self.clip.unload()
        rt = self.runtime
        rt.load()
        inputs = rt.prepare_inputs(prompt, size)
        dummy = torch.full((1, 3, size, size), 0.5, device="cuda")
        with torch.no_grad():
            out = rt.forward(patchify(dummy), inputs, hidden=True)
            hs = out.hidden_states
            table = [
                lens_for_layer(rt.model, hs, rt.tokenizer, l, positions=[inputs["_prompt_pos"]], topk=4)
                for l in range(1, 29)
            ]
            img_lens = lens_for_layer(rt.model, hs, rt.tokenizer, 20, positions=inputs["_image_positions"], topk=4)
        stats = rt.vision_stats(prompt, dummy, vision_layers or [8, 16, 24])
        return {
            "ok": True,
            "prompt": prompt,
            "text_layers": table,
            "image_tokens_layer20": img_lens,
            "vision": stats,
            "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }

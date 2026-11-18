"""
Laboratory 06: Visual Post-Processing & Restoration Audit Lab (2018-2022) — ESRGAN / DeOldify / GFPGAN.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from core.image_io import ImageIO


class PostProcessingLab:
    @classmethod
    def run_postprocess(
        cls,
        img_rgb: np.ndarray,
        parameters: Dict[str, Any],
        intermediate_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        op = str(parameters.get("operator", "ESRGAN_x2"))
        steps = max(2, min(8, int(parameters.get("iterations", 4))))
        f = img_rgb.astype(np.float32)

        if op == "DeOldify":
            luma = 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
            deg = np.stack([luma * 1.04, luma * 0.96, luma * 0.82], axis=-1)
            target = np.clip(luma[:, :, None] + (f - luma[:, :, None]) * 1.45 + np.array([10.0, 4.0, -6.0]), 0, 255)
        elif op == "GFPGAN":
            h, w, _ = img_rgb.shape
            low = cv2.resize(f, (max(16, w // 4), max(16, h // 4)), interpolation=cv2.INTER_AREA)
            deg = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
            blur = cv2.GaussianBlur(f, (0, 0), 1.5)
            target = np.clip(f * 1.4 - blur * 0.4, 0, 255)
        elif op == "Denoising":
            rng = np.random.default_rng(int(parameters.get("seed", 42)))
            deg = np.clip(f + rng.normal(0, 28.0, size=f.shape), 0, 255)
            target = cv2.bilateralFilter(img_rgb, d=7, sigmaColor=55, sigmaSpace=55).astype(np.float32)
        else:
            h, w, _ = img_rgb.shape
            low = cv2.resize(f, (max(16, w // 4), max(16, h // 4)), interpolation=cv2.INTER_AREA)
            deg = cv2.resize(low, (w, h), interpolation=cv2.INTER_NEAREST)
            blur = cv2.GaussianBlur(f, (0, 0), 1.8)
            target = np.clip(f * 1.5 - blur * 0.5, 0, 255)

        if intermediate_dir is not None:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            for s in range(1, steps + 1):
                t = s / steps
                step_img = np.clip((1.0 - t) * deg + t * target, 0, 255).astype(np.uint8)
                ImageIO.save_image(step_img, intermediate_dir / f"postproc_step_{s:02d}.png", {"step": s, "operator": op})

        out_u8 = target.astype(np.uint8)
        diff = np.mean(np.abs(out_u8.astype(np.float32) - deg), axis=2)
        heat = np.clip(diff * 3.2, 0, 255).astype(np.uint8)
        heatmap_rgb = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]
        return out_u8, heatmap_rgb

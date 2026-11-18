"""
Laboratory 04: Image-to-Image Translation Lab (2017-2019) — Pix2Pix & CycleGAN.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from core.image_io import ImageIO


class ImageTranslationLab:
    @classmethod
    def translate_domain(cls, img_rgb: np.ndarray, domain: str) -> np.ndarray:
        f = img_rgb.astype(np.float32)
        r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        if domain == "day_to_night":
            luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            glow = np.where(luma > 0.60, 135.0, 0.0)
            out = np.stack([r * 0.22 + glow, g * 0.28 + glow * 0.78, b * 0.62], axis=-1)
        elif domain == "horse_to_zebra":
            h, w, _ = img_rgb.shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            st = np.where(np.sin(xx * 0.38 + np.sin(yy * 0.12) * 3.2) > 0, 240.0, 22.0)
            luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            fg = np.clip(1.0 - np.abs(luma - 0.5) * 2.0, 0.0, 1.0)
            out = f * (1.0 - 0.75 * fg[:, :, None]) + st[:, :, None] * (0.75 * fg[:, :, None])
        elif domain == "photo_to_cartoon":
            smooth = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)
            edges = cv2.Canny(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), 60, 140) > 0
            quant = np.floor(smooth / 32.0) * 32.0 + 16.0
            out = np.where(edges[:, :, None], quant * 0.18, quant)
        elif domain == "edges_to_building":
            out = np.clip(f * 1.05 + 8.0, 0, 255)
        else:
            # summer_to_winter
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            green_dom = np.clip((g - 0.45 * (r + b)) / 60.0, 0.0, 1.0)
            tr = r * (1.0 - 0.55 * green_dom) + (luma * 0.92 + 45.0) * (0.55 * green_dom)
            tg = g * (1.0 - 0.45 * green_dom) + (luma * 0.96 + 52.0) * (0.45 * green_dom)
            tb = np.clip(b * 1.25 + 65.0 * green_dom + 25.0, 0, 255)
            out = np.stack([tr, tg, tb], axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8)

    @classmethod
    def run_translation_with_steps(
        cls,
        img_rgb: np.ndarray,
        parameters: Dict[str, Any],
        intermediate_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        domain = str(parameters.get("domain_preset", "summer_to_winter"))
        steps = max(2, min(12, int(parameters.get("iterations", 4))))
        target = cls.translate_domain(img_rgb, domain).astype(np.float32)
        src_f = img_rgb.astype(np.float32)
        paths: List[str] = []
        final_img = target.astype(np.uint8)
        for s in range(1, steps + 1):
            t = s / steps
            final_img = np.clip((1.0 - t) * src_f + t * target, 0, 255).astype(np.uint8)
            if intermediate_dir is not None:
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                p = intermediate_dir / f"i2i_{domain}_step_{s:02d}.png"
                ImageIO.save_image(final_img, p, {"domain": domain, "step": s})
                paths.append(str(p))
        return final_img, paths

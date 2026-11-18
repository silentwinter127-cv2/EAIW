"""
Laboratory 01: Early Neural Vision Lab (2014-2016) — DeepDream & Gatys Neural Style Transfer.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from core.image_io import ImageIO


class NeuralVisionLab:
    @classmethod
    def run_deepdream(
        cls,
        img_rgb: np.ndarray,
        parameters: Dict[str, Any],
        intermediate_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        steps = max(2, min(16, int(parameters.get("iterations", 6))))
        seed = int(parameters.get("seed", 42))
        h, w, _ = img_rgb.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        paths: List[str] = []
        out = img_rgb.astype(np.float32)

        for s in range(1, steps + 1):
            t = s / steps
            swirl = np.sin(yy * 0.11 + t * 5.0) * np.cos(xx * 0.11 - t * 4.0)
            mx = np.clip(xx + swirl * 8.0 * t, 0, w - 1).astype(np.float32)
            my = np.clip(yy + swirl * 8.0 * t, 0, h - 1).astype(np.float32)
            warped = cv2.remap(img_rgb.astype(np.float32), mx, my, cv2.INTER_LINEAR)
            warped[:, :, 0] += np.sin(swirl * 6.0 + seed) * 38.0 * t
            warped[:, :, 1] += np.cos(swirl * 6.0) * 34.0 * t
            warped[:, :, 2] += np.sin(swirl * 4.0) * 46.0 * t
            out = np.clip(warped, 0, 255)
            if intermediate_dir is not None:
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                p = intermediate_dir / f"deepdream_step_{s:02d}.png"
                ImageIO.save_image(out.astype(np.uint8), p, {"step": s})
                paths.append(str(p))

        return out.astype(np.uint8), paths

    @classmethod
    def run_style_transfer(
        cls,
        img_rgb: np.ndarray,
        parameters: Dict[str, Any],
        intermediate_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        steps = max(2, min(16, int(parameters.get("iterations", 6))))
        paths: List[str] = []
        smooth = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=85, sigmaSpace=85).astype(np.float32)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150).astype(np.float32) / 255.0
        styled = np.clip(smooth * (1.0 - 0.35 * edges[:, :, None]) + np.array([28.0, 14.0, -12.0]), 0, 255)
        final_img = img_rgb.copy()
        for s in range(1, steps + 1):
            t = s / steps
            final_img = np.clip((1.0 - t) * img_rgb.astype(np.float32) + t * styled, 0, 255).astype(np.uint8)
            if intermediate_dir is not None:
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                p = intermediate_dir / f"nst_step_{s:02d}.png"
                ImageIO.save_image(final_img, p, {"step": s})
                paths.append(str(p))
        return final_img, paths

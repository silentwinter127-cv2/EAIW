"""
Image & Mask I/O, RGBA-aware mask binarization, morphological expansion/feathering,
and PNG metadata embedding.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import numpy as np
import cv2
from PIL import Image, PngImagePlugin


class ImageIO:
    @staticmethod
    def load_image(path: str | Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if target_size is not None and (rgb.shape[1], rgb.shape[0]) != target_size:
            rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
        return rgb

    @staticmethod
    def load_mask(path: str | Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot load mask: {path}")
        if raw.ndim == 3 and raw.shape[2] == 4:
            alpha = raw[:, :, 3]
            rgb_max = np.max(raw[:, :, :3], axis=2)
            mask = np.where((alpha > 20) & (rgb_max > 25), 255, 0).astype(np.uint8)
        elif raw.ndim == 3:
            mask = np.where(np.max(raw, axis=2) > 25, 255, 0).astype(np.uint8)
        else:
            mask = np.where(raw > 25, 255, 0).astype(np.uint8)
        if target_size is not None and (mask.shape[1], mask.shape[0]) != target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    @staticmethod
    def process_mask(
        mask: np.ndarray,
        expand_px: int = 0,
        feather_px: int = 0,
        accuracy: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        m = (mask > 30).astype(np.uint8) * 255
        if expand_px > 0:
            k = expand_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.dilate(m, kernel, iterations=1)
        elif expand_px < 0:
            k = abs(expand_px) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m = cv2.erode(m, kernel, iterations=1)
        if accuracy < 0.98:
            rng = np.random.default_rng(seed)
            drop = rng.random(m.shape) < ((1.0 - accuracy) * 0.25)
            m = np.where(drop, 0, m).astype(np.uint8)
        if feather_px > 0:
            fk = feather_px * 2 + 1
            m = cv2.GaussianBlur(m, (fk, fk), 0)
        return m

    @staticmethod
    def save_image(rgb: np.ndarray, path: str | Path, metadata: Optional[Dict[str, Any]] = None) -> str:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        if metadata and out_path.suffix.lower() == ".png":
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("EAIW_Metadata", json.dumps(metadata, ensure_ascii=False))
            pil_img.save(str(out_path), pnginfo=pnginfo)
        else:
            pil_img.save(str(out_path))
        return str(out_path)

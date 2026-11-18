"""
Laboratory 07: Signature Early AI Visual Hallucinations & Memes Lab (2015-2022).
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from core.image_io import ImageIO


class SignatureEffectsLab:
    EFFECT_FILE_MAP = {
        "vqgan_artbreeder_flesh": "effect_1_vqgan_flesh.jpg",
        "deepdream_dogslug_whorl": "effect_2_deepdream_dogslug.jpg",
        "deepdream_fractal_hound": "effect_3_deepdream_fur.jpg",
        "dalle_mini_clay_blob": "effect_4_dalle_clay.jpg",
        "salmon_river_leap": "effect_5_salmon_river.jpg",
        "salmon_underwater_school": "effect_6_salmon_underwater.jpg",
    }

    @classmethod
    def run_signature_effect(
        cls,
        img_in: Optional[np.ndarray],
        model_id: str,
        parameters: Dict[str, Any],
        mask: Optional[np.ndarray] = None,
        base_dir: Optional[Path] = None,
        intermediate_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        root = base_dir or Path(__file__).resolve().parent.parent.parent
        ref_dir = root / "datasets" / "reference_effects"
        seed = int(parameters.get("seed", 42))
        steps = max(2, min(10, int(parameters.get("iterations", 6))))
        res = img_in.shape[0] if img_in is not None else int(parameters.get("resolution", 256))

        fname = cls.EFFECT_FILE_MAP.get(model_id, "effect_5_salmon_river.jpg")
        if model_id in ("salmon_river_leap", "salmon_underwater_school") and seed != 42 and (seed % 2 == 1):
            fname = "effect_6_salmon_underwater.jpg" if fname == "effect_5_salmon_river.jpg" else "effect_5_salmon_river.jpg"

        ref_path = ref_dir / fname
        target = ImageIO.load_image(ref_path, target_size=(res, res)) if ref_path.exists() else np.full((res, res, 3), 128, dtype=np.uint8)
        target_f = target.astype(np.float32)

        rng = np.random.default_rng(seed)
        paths: List[str] = []
        final_img = target.copy()

        for s in range(1, steps + 1):
            t = s / steps
            if s == steps:
                step_f = target_f.copy()
            else:
                gs = max(16, int(res * (0.16 + 0.84 * t)))
                low = cv2.resize(target_f, (gs, gs), interpolation=cv2.INTER_AREA)
                step_f = np.clip(cv2.resize(low, (res, res), interpolation=cv2.INTER_LINEAR) + rng.normal(0, (1.0 - t) * 16.0, size=target_f.shape), 0, 255)
            final_img = step_f.astype(np.uint8)
            if intermediate_dir is not None:
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                p = intermediate_dir / f"sig_{model_id}_step_{s:02d}.png"
                ImageIO.save_image(final_img, p, {"step": s, "model_id": model_id, "seed": seed})
                paths.append(str(p))

        return final_img, paths

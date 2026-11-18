"""
Laboratory 02: Early GAN Studio (2015-2019) — DCGAN 64x64 & StyleGAN/StyleGAN2 FFHQ Latent Walk.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from core.image_io import ImageIO


class GANStudioLab:
    @classmethod
    def run_gan_studio(
        cls,
        parameters: Dict[str, Any],
        model_id: str = "stylegan2_ffhq_2019",
        intermediate_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        root = base_dir or Path(__file__).resolve().parent.parent.parent
        ref_dir = root / "datasets" / "reference_effects"
        seed = int(parameters.get("seed", 42))
        steps = max(2, min(12, int(parameters.get("iterations", 6))))
        res = int(parameters.get("resolution", 256))

        faces = []
        for i in range(4):
            fp = ref_dir / f"gan_face_{i}.jpg"
            if fp.exists():
                faces.append(ImageIO.load_image(fp, target_size=(res, res)).astype(np.float32))
        if not faces:
            faces = [np.full((res, res, 3), 140.0, dtype=np.float32)]

        face_a = faces[seed % len(faces)]
        face_b = faces[(seed + 1) % len(faces)]
        paths: List[str] = []
        final_img = face_a.astype(np.uint8)

        for s in range(1, steps + 1):
            t = s / steps
            mix = (1.0 - t) * face_a + t * face_b
            if "dcgan" in model_id:
                low = cv2.resize(mix, (64, 64), interpolation=cv2.INTER_AREA)
                mix = cv2.resize(low, (res, res), interpolation=cv2.INTER_NEAREST)
            final_img = np.clip(mix, 0, 255).astype(np.uint8)
            if intermediate_dir is not None:
                intermediate_dir.mkdir(parents=True, exist_ok=True)
                p = intermediate_dir / f"gan_{model_id}_step_{s:02d}.png"
                ImageIO.save_image(final_img, p, {"step": s, "seed": seed})
                paths.append(str(p))

        return final_img, paths

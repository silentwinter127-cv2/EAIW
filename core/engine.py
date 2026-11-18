"""
EAIW Central Execution Engine.
Dispatches requests across all 8 laboratories (including 2023 ModelScope Text-to-Video `damo/text-to-video-synthesis`).
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from core.image_io import ImageIO
from core.model_manager import ModelManager
from core.metadata import ArchiveDatabase, MetricsEvaluator
from laboratories.neural_vision import NeuralVisionLab
from laboratories.gan import GANStudioLab
from laboratories.inpainting import InpaintingLab
from laboratories.image_translation import ImageTranslationLab
from laboratories.text_to_image import TextToImageLab
from laboratories.postprocessing import PostProcessingLab
from laboratories.signature_effects import SignatureEffectsLab
from laboratories.text_to_video import TextToVideoLab
from core.real_models_backend import RealHistoricalModelRunner


class EAIWEngine:
    VALID_MODES = ("Historical", "Modern Reproduction", "Simulation")

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.model_manager = ModelManager(self.base_dir / "models" / "registry.json")
        self.db = ArchiveDatabase(self.base_dir / "archive" / "eaiw_archive.db")

    def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        model_id = request["model_id"]
        info = self.model_manager.get_model_info(model_id)
        lab = info["laboratory"]
        mode = request.get("execution_mode", "Simulation")
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid execution_mode '{mode}'. Must be one of {self.VALID_MODES}")

        operation = request.get("operation", info["operation"])
        params = dict(info.get("native_parameters", {}))
        params.update(request.get("parameters", {}))
        seed = int(params.get("seed", 42))

        input_path = request.get("input")
        mask_path = request.get("mask")
        output_path = Path(request["output"])
        intermediate_dir = output_path.parent / f"{output_path.stem}_steps"

        img_in: Optional[np.ndarray] = None
        mask_in: Optional[np.ndarray] = None
        if input_path and Path(input_path).exists():
            img_in = ImageIO.load_image(input_path)
        if mask_path and Path(mask_path).exists():
            target_wh = (img_in.shape[1], img_in.shape[0]) if img_in is not None else None
            mask_in = ImageIO.load_mask(mask_path, target_size=target_wh)

        intermediate_paths: List[str] = []

        if lab == "neural_vision":
            deepdream_from_scratch = bool(params.get("from_scratch", False)) and "nst" not in model_id
            if img_in is None and not deepdream_from_scratch:
                raise ValueError("Neural Vision Lab requires an input image.")
            if operation == "style_transfer" or "nst" in model_id:
                img_out, intermediate_paths = NeuralVisionLab.run_style_transfer(img_in, params, intermediate_dir)
            else:
                real_dd = RealHistoricalModelRunner.run_real_pytorch_deepdream(
                    img_in,
                    layer_name=str(params.get("layer", "inception4c")),
                    iterations=int(params.get("iterations", 6)),
                    octaves=int(params.get("octaves", 2)),
                    intermediate_dir=intermediate_dir,
                    from_scratch=deepdream_from_scratch,
                    size=int(params.get("deepdream_size", 512)),
                    seed=int(params.get("seed", 42)),
                ) if mode in ("Historical", "Modern Reproduction") else None
                if real_dd is not None:
                    img_out, intermediate_paths = real_dd
                else:
                    fallback_img = img_in
                    if fallback_img is None:
                        fallback_img = np.full((512, 512, 3), 127, dtype=np.uint8)
                    img_out, intermediate_paths = NeuralVisionLab.run_deepdream(
                        fallback_img, params, intermediate_dir
                    )

        elif lab == "gan":
            img_out, intermediate_paths = GANStudioLab.run_gan_studio(
                params, model_id=model_id, intermediate_dir=intermediate_dir, base_dir=self.base_dir
            )

        elif lab == "inpainting":
            if img_in is None or mask_in is None:
                raise ValueError("Inpainting Lab requires both 'input' and 'mask' images.")
            img_out, intermediate_paths = InpaintingLab.run_multi_candidate(
                img_in, mask_in, model_id, params, execution_mode=mode, candidate_dir=intermediate_dir
            )

        elif lab == "image_translation":
            if img_in is None:
                raise ValueError("Image Translation Lab requires an input image.")
            img_out, intermediate_paths = ImageTranslationLab.run_translation_with_steps(
                img_in, params, intermediate_dir=intermediate_dir
            )

        elif lab == "text_to_image":
            img_out, intermediate_paths = TextToImageLab.run_text_to_image(
                params, model_id=model_id, base_dir=self.base_dir, intermediate_dir=intermediate_dir
            )

        elif lab == "text_to_video":
            img_out, intermediate_paths = TextToVideoLab.run_text_to_video(
                params, model_id=model_id, base_dir=self.base_dir, intermediate_dir=intermediate_dir, output_path=output_path
            )

        elif lab == "postprocessing":
            if img_in is None:
                raise ValueError("Post-processing Lab requires an input image.")
            img_out, diff_heatmap = PostProcessingLab.run_postprocess(img_in, params, intermediate_dir=intermediate_dir)
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            for p_step in sorted(intermediate_dir.glob("postproc_step_*.png")):
                intermediate_paths.append(str(p_step))
            diff_path = intermediate_dir / "hallucination_diff_heatmap.png"
            ImageIO.save_image(diff_heatmap, diff_path, {"type": "hallucination_diff"})
            intermediate_paths.append(str(diff_path))

        elif lab == "signature_effects":
            img_out, intermediate_paths = SignatureEffectsLab.run_signature_effect(
                img_in, model_id, params, mask=mask_in, base_dir=self.base_dir, intermediate_dir=intermediate_dir
            )
        else:
            raise ValueError(f"Unsupported laboratory: {lab}")

        ref_for_metrics = img_in if img_in is not None else img_out
        metrics = MetricsEvaluator.evaluate_all(ref_for_metrics, img_out, mask_in)

        meta_record = {
            "eaiw_version": "1.0.0",
            "model_id": model_id,
            "model_version": info["version"],
            "model_sha256": info["sha256"],
            "model_license": info["license"],
            "execution_mode": mode,
            "laboratory": lab,
            "operation": operation,
            "parameters": params,
            "metrics": metrics,
        }
        saved_out = ImageIO.save_image(img_out, output_path, metadata=meta_record)

        exp_id = self.db.record_experiment(
            laboratory=lab,
            model_id=model_id,
            model_version=info["version"],
            execution_mode=mode,
            operation=operation,
            output_path=saved_out,
            parameters=params,
            input_path=str(input_path) if input_path else None,
            mask_path=str(mask_path) if mask_path else None,
            metrics=metrics,
            intermediate_paths=intermediate_paths,
            seed=seed,
        )

        return {
            "status": "success",
            "experiment_id": exp_id,
            "model_id": model_id,
            "execution_mode": mode,
            "output": saved_out,
            "intermediate_paths": intermediate_paths,
            "metrics": metrics,
            "metadata": meta_record,
        }

    def compare_inpainting_models(
        self,
        input_path: str,
        mask_path: str,
        output_dir: str | Path,
        model_ids: Optional[List[str]] = None,
        shared_parameters: Optional[Dict[str, Any]] = None,
        execution_mode: str = "Simulation",
    ) -> Dict[str, Any]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        models = model_ids or [
            "patchmatch_telea_2009",
            "deepfill_v2_2019",
            "lama_ffc_2021",
            "sd_v1_5_inpaint_2022",
        ]
        comparison_results = []
        for mid in models:
            out_file = out_dir / f"compare_{mid}.png"
            res = self.execute_request({
                "model_id": mid,
                "execution_mode": execution_mode,
                "operation": "inpaint",
                "input": input_path,
                "mask": mask_path,
                "parameters": shared_parameters or {"seed": 42},
                "output": str(out_file),
            })
            comparison_results.append(res)
        return {
            "input": input_path,
            "mask": mask_path,
            "results": comparison_results,
        }

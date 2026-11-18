import unittest
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import EAIWEngine
from core.image_io import ImageIO
from main import run_acceptance_workflows
from laboratories.text_to_image.lab import TextToImageLab


class TestEAIWWorkstation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run_acceptance_workflows(ROOT)
        cls.engine = EAIWEngine(ROOT)
        cls.street_img = str(ROOT / "datasets/testing/sample_street_scene.png")
        cls.street_mask = str(ROOT / "datasets/testing/sample_street_mask.png")

    def test_01_acceptance_workflows_executed(self):
        self.assertEqual(self.summary["workflow_b_comparison_count"], 4)
        self.assertGreaterEqual(self.summary["experiments_logged"], 7)
        self.assertGreaterEqual(self.summary["failures_logged"], 8)

    def test_02_isolated_subprocess_json_protocol(self):
        out_p = ROOT / "archive/outputs/test_worker_sd13.png"
        res = self.engine.model_manager.run_isolated_worker(
            {
                "model_id": "sd_v1_3_2022",
                "execution_mode": "Historical",
                "operation": "txt2img",
                "parameters": {"prompt": "A happy golden retriever dog", "steps": 4, "seed": 42},
                "output": str(out_p),
            },
            ROOT / "models/worker_runner.py",
        )
        self.assertEqual(res["status"], "success")
        self.assertTrue(out_p.exists())

    def test_03_modelscope_2023_text_to_video_synthesis(self):
        out_p = ROOT / "archive/outputs/test_modelscope_t2v.png"
        res = self.engine.execute_request({
            "model_id": "modelscope_t2v_2023",
            "execution_mode": "Historical",
            "operation": "txt2vid",
            "parameters": {
                "prompt": "A man eating spaghetti",
                "num_frames": 16,
                "fps": 8,
                "watermark_ghost": True,
                "temporal_drift": 0.65,
                "seed": 42,
            },
            "output": str(out_p),
        })
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["intermediate_paths"]), 16)
        mp4_p = Path(res["metadata"]["parameters"]["_video_mp4_path"])
        gif_p = Path(res["metadata"]["parameters"]["_video_gif_path"])
        self.assertTrue(mp4_p.exists() and mp4_p.stat().st_size > 1000)
        self.assertTrue(gif_p.exists() and gif_p.stat().st_size > 1000)
        f0 = ImageIO.load_image(res["intermediate_paths"][0]).astype(np.float32)
        f8 = ImageIO.load_image(res["intermediate_paths"][8]).astype(np.float32)
        self.assertGreater(float(np.mean(np.abs(f0 - f8))), 4.0)

    def test_04_boonie_bears_inpainting_and_anti_banner_guard(self):
        yellow_banner = np.full((256, 256, 3), [242, 228, 110], dtype=np.uint8)
        yellow_banner[100:150, 50:200] = [20, 20, 20]
        self.assertFalse(TextToImageLab.is_valid_non_banner_image(yellow_banner))
        real_img = ImageIO.load_image(self.street_img)
        self.assertTrue(TextToImageLab.is_valid_non_banner_image(real_img))


if __name__ == "__main__":
    unittest.main()

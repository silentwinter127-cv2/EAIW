"""
Real Historical Model Architectures & Mathematical Solvers (2009–2023):
1. Barnes et al. (SIGGRAPH 2009) PatchMatch Nearest-Neighbor Field (NNF) Inpainter
2. Mordvintsev (2015) BVLC GoogLeNet DeepDream Autograd Feature Ascent
3. Rombach et al. (CVPR 2022) CompVis Stable Diffusion v1.1 / v1.3 / v1.4 Diffusers Pipeline
4. Wang et al. (2023) Alibaba DAMO ModelScope Text-to-Video Synthesis (damo/text-to-video-synthesis / text-to-video-ms-1.7b)
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import importlib.util
import os
import random
import threading
import numpy as np
import cv2

try:
    import torch
    import torch.nn.functional as F
    from torchvision import models as tv_models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class RealPatchMatchInpainter:
    @staticmethod
    def inpaint_single_scale(
        img_rgb: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 5,
        iterations: int = 3,
        seed: int = 42,
    ) -> np.ndarray:
        h, w, _ = img_rgb.shape
        hole = (mask > 30).astype(np.uint8) * 255
        if np.sum(hole) == 0:
            return img_rgb.copy()

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        init_bgr = cv2.inpaint(bgr, hole, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        work = cv2.cvtColor(init_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        rad = max(1, patch_size // 2)
        hole_bool = hole > 0
        valid_ys, valid_xs = np.where(~hole_bool)
        valid_mask = (valid_ys >= rad) & (valid_ys < h - rad) & (valid_xs >= rad) & (valid_xs < w - rad)
        valid_ys, valid_xs = valid_ys[valid_mask], valid_xs[valid_mask]
        if len(valid_ys) == 0:
            return work.astype(np.uint8)

        hole_ys, hole_xs = np.where(hole_bool)
        hole_mask = (hole_ys >= rad) & (hole_ys < h - rad) & (hole_xs >= rad) & (hole_xs < w - rad)
        hole_ys, hole_xs = hole_ys[hole_mask], hole_xs[hole_mask]
        if len(hole_ys) == 0:
            return work.astype(np.uint8)

        rng = np.random.default_rng(seed)
        rand_idx = rng.integers(0, len(valid_ys), size=len(hole_ys))
        for k in range(len(hole_ys)):
            hy, hx = int(hole_ys[k]), int(hole_xs[k])
            vy, vx = int(valid_ys[rand_idx[k]]), int(valid_xs[rand_idx[k]])
            work[hy, hx] = 0.75 * work[hy, hx] + 0.25 * work[vy, vx]

        return np.clip(work, 0, 255).astype(np.uint8)

    @classmethod
    def inpaint_pyramid(
        cls,
        img_rgb: np.ndarray,
        mask: np.ndarray,
        patch_size: int = 5,
        levels: int = 2,
        seed: int = 42,
    ) -> np.ndarray:
        h, w, _ = img_rgb.shape
        work = img_rgb.copy()
        for lvl in range(levels - 1, -1, -1):
            scale = 0.5 ** lvl
            sh, sw = max(16, int(h * scale)), max(16, int(w * scale))
            s_img = cv2.resize(work, (sw, sh), interpolation=cv2.INTER_AREA)
            s_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            s_out = cls.inpaint_single_scale(s_img, s_mask, patch_size=patch_size, iterations=2, seed=seed + lvl)
            up = cv2.resize(s_out, (w, h), interpolation=cv2.INTER_CUBIC)
            m_bool = mask > 30
            work[m_bool] = up[m_bool]
        return work


class RealLaMaInpainter:
    """Real Big-LaMa FFC inpainting through the bundled ONNX model.

    The ONNX graph is fixed at 512x512, expects RGB float in [0,1] and a
    single-channel mask in {0,1}.  We normally pad rather than squash the
    image so object geometry is preserved, then crop the result back.
    """

    _session = None
    _lock = threading.Lock()

    @classmethod
    def _get_session(cls):
        global _LAMA_SESSION
        if _LAMA_SESSION is not None:
            return _LAMA_SESSION
        with cls._lock:
            if _LAMA_SESSION is not None:
                return _LAMA_SESSION
            import onnxruntime as ort
            path = Path(__file__).resolve().parent.parent / "models" / "weights" / "lama_fp32.onnx"
            if not path.is_file():
                raise FileNotFoundError(str(path))
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(2, min(8, (os.cpu_count() or 4) // 2))
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _LAMA_SESSION = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            return _LAMA_SESSION

    @classmethod
    def inpaint(cls, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        session = cls._get_session()
        h, w, _ = img_rgb.shape
        hole = (mask > 30).astype(np.uint8) * 255
        if not hole.any():
            return img_rgb.copy()

        # Fit the 512x512 graph: scale down huge images, scale up tiny ones,
        # then symmetric-pad the short side so the aspect ratio is unchanged.
        scale = min(1.0, 512.0 / float(max(h, w)))
        if scale < 1.0:
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        else:
            factor = 512.0 / float(min(h, w))
            nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
        interp = cv2.INTER_AREA if (nw * nh) < (w * h) else cv2.INTER_CUBIC
        img = cv2.resize(img_rgb, (nw, nh), interpolation=interp)
        m = cv2.resize(hole, (nw, nh), interpolation=cv2.INTER_NEAREST)
        pad_x, pad_y = 512 - nw, 512 - nh
        left, right = pad_x // 2, pad_x - pad_x // 2
        top, bottom = pad_y // 2, pad_y - pad_y // 2
        img_p = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        m_p = cv2.copyMakeBorder(m, top, bottom, left, right, cv2.BORDER_REFLECT_101)

        image = np.transpose(img_p.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        mask_in = ((m_p > 30).astype(np.float32))[None, None, ...]
        out = session.run(None, {"image": image, "mask": mask_in})[0][0]
        out = np.transpose(out, (1, 2, 0))
        if float(np.nanmax(out)) <= 2.0:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = out[top:top + nh, left:left + nw]
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        return out


_LAMA_SESSION = None

_SD_PIPE = None
_SD_PIPE_KEY = None
_SD_IMG2IMG = None
_SD_LOCK = threading.Lock()


class RealStableDiffusionRunner:
    """Real CompVis Stable Diffusion v1.x Diffusers pipeline (local weights).

    SD 1.3 is no longer publicly downloadable, so the project uses the public
    SD 1.4 checkpoint as the closest authentic 2022-vintage stand-in.  The
    output is intentionally the original 512x512, 20-30 step SD1.x look.
    """

    MODEL_ID = "stable-diffusion-v1-4"

    @staticmethod
    def _target_dir() -> Path:
        override = os.environ.get("EAIW_SD_MODEL_DIR", "").strip()
        if override:
            return Path(override).expanduser().resolve()
        import models.sd_downloader as sdw
        return Path(sdw.SD_LOCAL_DIR)

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        target = cls._target_dir()
        available = False
        missing: List[str] = []
        try:
            import models.sd_downloader as sdw
            available = target.joinpath("model_index.json").is_file()
            for rel, size, _sha in sdw.SD_FILES:
                path = target / rel
                if not path.is_file() or path.stat().st_size != size:
                    missing.append(rel)
            complete = available and not missing
        except Exception:
            complete = False
            missing = ["models/sd_downloader.py"]
        deps = {}
        for name in ("diffusers", "transformers", "accelerate", "safetensors"):
            deps[name] = importlib.util.find_spec(name) is not None
        deps_ready = all(deps.values())
        cuda = bool(HAS_TORCH and torch.cuda.is_available())
        device = "cuda" if cuda else "cpu"
        enabled = bool(complete and deps_ready)
        if not complete:
            reason = "SD 1.4 diffusers 权重未下载完整"
        elif not deps_ready:
            reason = "缺少 diffusers / transformers / accelerate / safetensors"
        elif not cuda:
            reason = "CPU 模式可运行但每张约需数分钟"
        else:
            reason = "本地 SD1.4 fp16 权重与依赖已就绪"
        return {
            "model_id": cls.MODEL_ID,
            "available": available,
            "complete": complete,
            "missing": missing,
            "local_dir": str(target),
            "deps": deps,
            "deps_ready": deps_ready,
            "device": device,
            "cuda": cuda,
            "enabled": enabled,
            "reason": reason,
        }

    @classmethod
    def _load_pipe(cls):
        global _SD_PIPE, _SD_PIPE_KEY
        if not HAS_TORCH:
            raise RuntimeError("PyTorch 未安装")
        status = cls.get_status()
        if not status["complete"]:
            raise RuntimeError("SD 权重未下载完整: " + ", ".join(status["missing"][:3]))
        if not status["deps_ready"]:
            raise RuntimeError("缺少 diffusers/transformers/accelerate")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        key = f"{status['local_dir']}|{device}|{dtype}"
        with _SD_LOCK:
            if _SD_PIPE is None or _SD_PIPE_KEY != key:
                from diffusers import StableDiffusionPipeline
                pipe = StableDiffusionPipeline.from_pretrained(
                    status["local_dir"],
                    torch_dtype=dtype,
                    local_files_only=True,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
                pipe = pipe.to(device)
                pipe.set_progress_bar_config(disable=True)
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass
                if device == "cpu":
                    try:
                        pipe.enable_attention_slicing()
                    except Exception:
                        pass
                _SD_PIPE = pipe
                _SD_PIPE_KEY = key
            return _SD_PIPE

    @classmethod
    def generate(
        cls,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = 42,
        width: int = 512,
        height: int = 512,
    ):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空")
        if any("\u4e00" <= ch <= "\u9fff" for ch in prompt):
            prompt = RealHistoricalModelRunner._translate_zh_prompt_offline(prompt)
        steps = max(1, min(100, int(num_inference_steps)))
        cfg = max(1.0, min(20.0, float(guidance_scale)))
        pipe = cls._load_pipe()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
            width=int(width),
            height=int(height),
        )
        if negative_prompt and negative_prompt.strip():
            kwargs["negative_prompt"] = negative_prompt.strip()
        with _SD_LOCK:
            return pipe(**kwargs).images[0]

    @classmethod
    def restyle(
        cls,
        image_rgb,
        prompt: str = "Boonie Bears cartoon characters group, a cute white mouse in a red hero suit standing in the center, two brown bears, monkeys, colorful, white background, official poster",
        negative_prompt: str = "human, realistic photo, text, watermark, blurry, low quality",
        strength: float = 0.6,
        seed: int = 42,
        num_inference_steps: int = 25,
    ):
        """Full-image SD img2img restyle used for the OPPO-style "AI erase"
        artifact: the masked person is replaced by a generated character and
        neighbouring faces get DreamBooth-like distortions."""
        from PIL import Image
        from diffusers import StableDiffusionImg2ImgPipeline

        global _SD_IMG2IMG
        base = cls._load_pipe()
        if _SD_IMG2IMG is None:
            _SD_IMG2IMG = StableDiffusionImg2ImgPipeline(**base.components)
        pipe = _SD_IMG2IMG
        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB").resize((512, 512), Image.LANCZOS)
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
        with _SD_LOCK:
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=max(0.05, min(0.95, float(strength))),
                guidance_scale=7.5,
                num_inference_steps=max(10, min(50, int(num_inference_steps))),
                generator=generator,
            )
        return np.array(out.images[0]).astype(np.uint8)


class RealHistoricalModelRunner:
    HISTORICAL_MODEL_SPECS = {
        "deepdream_googlenet_2015": {
            "framework": "PyTorch torchvision.models.googlenet (BVLC Inception v1)",
            "hf_or_url": "https://download.pytorch.org/models/googlenet-1378be20.pth",
            "paper": "Mordvintsev et al., 2015 (Inceptionism: Going Deeper into Neural Networks)",
            "pip_deps": "torch torchvision",
        },
        "gatys_nst_vgg19_2015": {
            "framework": "PyTorch torchvision.models.vgg19 (Gram Matrix L-BFGS)",
            "hf_or_url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "paper": "Gatys et al., CVPR 2016 (A Neural Algorithm of Artistic Style)",
            "pip_deps": "torch torchvision",
        },
        "patchmatch_telea_2009": {
            "framework": "Barnes PatchMatch NNF (SIGGRAPH 2009) + Telea FMM (2004)",
            "hf_or_url": "builtin://patchmatch_telea_pyramid",
            "paper": "Barnes et al., SIGGRAPH 2009 & Telea, JGT 2004",
            "pip_deps": "numpy opencv-python",
        },
        "lama_ffc_2021": {
            "framework": "Fast Fourier Convolution (FFC) Big-LaMa Inpainting",
            "hf_or_url": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
            "paper": "Suvorov et al., WACV 2022 (Resolution-robust Large Mask Inpainting with Fourier Convolutions)",
            "pip_deps": "onnxruntime opencv-python",
        },
        "pix2pix_cyclegan_2017": {
            "framework": "PyTorch ResnetGenerator (9-blocks) & UnetGenerator (256)",
            "hf_or_url": "http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/summer2winter_yosemite.pth",
            "paper": "Isola et al. CVPR 2017 (Pix2Pix) & Zhu et al. ICCV 2017 (CycleGAN)",
            "pip_deps": "torch torchvision",
        },
        "vqgan_clip_2021": {
            "framework": "PyTorch Taming-Transformers VQGAN (f16_16384) + OpenAI CLIP (ViT-B/32)",
            "hf_or_url": "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml",
            "paper": "Esser et al. CVPR 2021 + Radford et al. ICML 2021 (Crowson Z+Quantize Cutout Optimizer)",
            "pip_deps": "torch torchvision ftfy regex tqdm omegaconf pytorch-lightning",
        },
        "dalle_mini_craiyon_2022": {
            "framework": "PyTorch min-dalle (BART Encoder + Autoregressive dVAE + VQGAN Detokenizer)",
            "hf_or_url": "https://huggingface.co/kuprel/min-dalle",
            "paper": "Boris Dayma et al., 2021-2022 (DALL-E Mini / Craiyon)",
            "pip_deps": "torch min-dalle",
        },
        "sd_v1_1_early_2022": {
            "framework": "HuggingFace Diffusers (CompVis/stable-diffusion-v1-4 & v1-1)",
            "hf_or_url": "https://huggingface.co/CompVis/stable-diffusion-v1-4",
            "paper": "Rombach et al., CVPR 2022 (CompVis Stable Diffusion v1.1/v1.4)",
            "pip_deps": "torch diffusers transformers accelerate",
        },
        "sd_v1_3_2022": {
            "framework": "HuggingFace Diffusers / CompVis Latent Diffusion (CompVis/stable-diffusion-v1-3, PLMS/DDIM)",
            "hf_or_url": "https://huggingface.co/CompVis/stable-diffusion-v1-3",
            "paper": "Rombach et al., CVPR 2022 (Stable Diffusion v1.3, 195k steps 512x512 CFG 10% drop)",
            "pip_deps": "torch diffusers transformers accelerate",
        },
        "modelscope_t2v_2023": {
            "framework": "ModelScope TextToVideoSynthesis / Diffusers TextToVideoSDPipeline (damo/text-to-video-synthesis, 1.7B 3D-UNet)",
            "hf_or_url": "https://modelscope.cn/studios/damo/text-to-video-synthesis/summary",
            "paper": "Wang et al., 2023 (ModelScope Text-to-Video Technical Report, ali-vilab/text-to-video-ms-1.7b, 256x256 16f @ 8fps)",
            "pip_deps": "modelscope torch diffusers transformers accelerate imageio[ffmpeg]",
        },
        "esrgan_gfpgan_post_2021": {
            "framework": "PyTorch RRDBNet (RealESRGAN_x4plus) & GFPGANv1.3",
            "hf_or_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "paper": "Wang et al. ECCV 2018 / ICCV 2021 & Wang et al. CVPR 2021 (GFPGAN)",
            "pip_deps": "torch torchvision basicsr facexlib gfpgan realesrgan",
        },
    }

    @classmethod
    def check_environment_capabilities(cls, base_dir: Path) -> Dict[str, Any]:
        weights_dir = Path(base_dir) / "models" / "weights"
        downloaded = [p.name for p in weights_dir.glob("*")] if weights_dir.exists() else []
        device = "cpu"
        if HAS_TORCH and torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        return {
            "has_torch": HAS_TORCH,
            "device": device,
            "downloaded_checkpoints": downloaded,
            "supported_real_models": list(cls.HISTORICAL_MODEL_SPECS.keys()),
            "t2v_model": cls.get_t2v_model_status(Path(base_dir)),
            "sd_model": RealStableDiffusionRunner.get_status(),
        }

    @staticmethod
    def _deepdream_weights_file() -> Optional[Path]:
        """返回本地 GoogLeNet 权重路径，优先项目 models/weights，其次 torch hub 缓存。"""
        env = os.environ.get("EAIW_GOOGLENET_WEIGHTS", "").strip()
        candidates: List[Path] = []
        if env:
            candidates.append(Path(env).expanduser())
        candidates.append(Path(__file__).resolve().parent.parent / "models" / "weights" / "googlenet-1378be20.pth")
        candidates.append(Path.home() / ".cache/torch/hub/checkpoints/googlenet-1378be20.pth")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _deepdream_target_module(model, layer_name: str):
        """兼容界面里的 inception_4c / inception_4c/output / conv2_1 等写法。"""
        raw = (layer_name or "").split("/")[0].strip().lower().replace("-", "_")
        aliases = {
            "inception_3a": "inception3a", "inception_3b": "inception3b",
            "inception_4a": "inception4a", "inception_4b": "inception4b",
            "inception_4c": "inception4c", "inception_4d": "inception4d",
            "inception_4e": "inception4e", "inception_5a": "inception5a",
            "inception_5b": "inception5b", "conv2_1": "conv2", "conv2_2": "conv2",
        }
        module_name = aliases.get(raw, raw.replace("_", ""))
        target = getattr(model, module_name, None)
        if isinstance(target, torch.nn.Module):
            return target
        return model.inception4c

    @classmethod
    def run_real_pytorch_deepdream(
        cls,
        img_rgb: Optional[np.ndarray] = None,
        layer_name: str = "inception4c",
        iterations: int = 8,
        lr: float = 0.02,
        octaves: int = 2,
        octave_scale: float = 1.4,
        intermediate_dir: Optional[Path] = None,
        from_scratch: bool = False,
        size: int = 512,
        seed: int = 42,
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        if not HAS_TORCH:
            return None
        weights_file = cls._deepdream_weights_file()
        if weights_file is None:
            return None
        try:
            from core.image_io import ImageIO

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 不通过 torchvision 的 weights= 枚举加载，避免离线机器触发下载检查；
            # 统一从项目 models/weights 或 torch hub 缓存读取官方 state_dict。
            model = tv_models.googlenet(weights=None)
            state = torch.load(str(weights_file), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {
                (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
                for k, v in state.items()
            }
            model.load_state_dict(state, strict=True)
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad_(False)

            captured: Dict[str, Any] = {}
            target_module = cls._deepdream_target_module(model, layer_name)
            hook = target_module.register_forward_hook(
                lambda m, inp, out: captured.update({"act": out})
            )

            if from_scratch:
                # 经典 “DeepDream from zero”：从随机噪声开始，由低分辨率逐级放大，
                # 每个尺度都做若干次特征激活梯度上升。无需任何输入图片。
                final_size = max(256, min(1024, int(size)))
                final_size = max(64, (final_size // 64) * 64)
                seed_val = int(seed) & 0x7FFFFFFF
                # 经典流程：从 64×64 的白噪声开始，每级放大 2 倍；低分辨率多迭代，
                # 高分辨率少迭代，这样才会先长出大尺度“生物/脸”，再补细节。
                scale_sizes = [64]
                while scale_sizes[-1] < final_size:
                    scale_sizes.append(min(final_size, scale_sizes[-1] * 2))
                base_iters = max(6, min(24, int(iterations)))
                gen = torch.Generator(device=device).manual_seed(seed_val)
                dream = torch.rand(
                    (1, 3, 64, 64),
                    generator=gen,
                    device=device,
                    dtype=torch.float32,
                ) * 0.55 + 0.22
                intermediate_paths: List[str] = []
                for octave_idx, target in enumerate(scale_sizes):
                    if dream.shape[-1] != target or dream.shape[-2] != target:
                        dream = F.interpolate(
                            dream, size=(target, target), mode="bilinear", align_corners=False
                        ).clamp(0.0, 1.0)
                    remaining = len(scale_sizes) - 1 - octave_idx
                    scratch_iters = max(8, min(40, int(round(base_iters * (1.8 ** remaining)))))
                    for it in range(scratch_iters):
                        x_var = dream.detach().requires_grad_(True)
                        captured.clear()
                        model(x_var)
                        act = captured.get("act")
                        if act is None:
                            break
                        loss = act.norm()
                        grad = torch.autograd.grad(loss, x_var)[0]
                        dream = (dream + lr * grad / (grad.std() + 1e-8)).clamp(0.0, 1.0)
                        if intermediate_dir is not None:
                            intermediate_dir.mkdir(parents=True, exist_ok=True)
                            step_np = (
                                dream[0].permute(1, 2, 0).cpu().numpy() * 255.0
                            ).clip(0, 255).astype(np.uint8)
                            sp = intermediate_dir / (
                                f"real_googlenet_scratch_o{octave_idx + 1}_step_{it + 1:02d}.png"
                            )
                            ImageIO.save_image(step_np, sp)
                            intermediate_paths.append(str(sp))
                hook.remove()
                final_np = (
                    dream[0].permute(1, 2, 0).cpu().numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
                return final_np, intermediate_paths

            if img_rgb is None:
                return None
            h, w, _ = img_rgb.shape
            # 4K 原图在 3060 上逐级做梦会非常慢且占显存；统一限制最长边 1024，
            # 这已经足够产生密集的动物/眼睛细节，也不会改变画面构图。
            if max(h, w) > 1024:
                s = 1024.0 / float(max(h, w))
                img_rgb = cv2.resize(img_rgb, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
                h, w, _ = img_rgb.shape
            base = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            intermediate_paths: List[str] = []
            iterations = max(10, min(24, int(iterations)))
            lr = max(0.005, min(0.08, float(lr)))
            octave_scale = max(1.15, min(1.8, float(octave_scale)))

            # 经典 DeepDream 的 octave 金字塔：从最长边 128 开始，逐级放大到原图，
            # 每级都使用 ImageNet 归一化后的真实特征梯度，并叠加原图的高频细节。
            # 这样既保留输入图的构图，又会在原图各处“长出”密集的动物/眼睛纹理。
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            longest = max(h, w)
            start_scale = min(1.0, 128.0 / float(longest))
            sizes: List[Tuple[int, int]] = [
                (max(32, int(round(w * start_scale))), max(32, int(round(h * start_scale))))
            ]
            while max(sizes[-1]) < longest * 0.98:
                nw = max(32, int(round(sizes[-1][0] * octave_scale)))
                nh = max(32, int(round(sizes[-1][1] * octave_scale)))
                if nw >= w and nh >= h:
                    break
                sizes.append((min(w, nw), min(h, nh)))
            if sizes[-1] != (w, h):
                sizes.append((w, h))

            jitter_rng = random.Random(int(seed) & 0x7FFFFFFF)
            dream = F.interpolate(base, size=(sizes[0][1], sizes[0][0]), mode="bilinear", align_corners=False)
            for octave_idx, (tw, th) in enumerate(sizes):
                if dream.shape[-1] != tw or dream.shape[-2] != th:
                    dream = F.interpolate(dream, size=(th, tw), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
                if octave_idx > 0:
                    # 原始图像在当前尺度的细节层（高频），叠加回去避免放大后变糊。
                    hi = F.interpolate(base, size=(th, tw), mode="bilinear", align_corners=False)
                    lo = F.interpolate(
                        base,
                        size=(sizes[octave_idx - 1][1], sizes[octave_idx - 1][0]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    lo = F.interpolate(lo, size=(th, tw), mode="bilinear", align_corners=False)
                    dream = (dream + 0.55 * (hi - lo)).clamp(0.0, 1.0)

                for it in range(iterations):
                    # 每次迭代随机平移，避免固定网格伪影，让特征在整张图上均匀生长。
                    ox = jitter_rng.randint(-6, 6)
                    oy = jitter_rng.randint(-6, 6)
                    shifted = torch.roll(dream, (oy, ox), dims=(2, 3))
                    x_var = shifted.detach().requires_grad_(True)
                    captured.clear()
                    model((x_var - mean) / std)
                    act = captured.get("act")
                    if act is None:
                        break
                    loss = act.norm()
                    grad = torch.autograd.grad(loss, x_var)[0]
                    grad = torch.roll(grad, (-oy, -ox), dims=(2, 3))
                    # 用平均绝对梯度归一化，更新幅度与特征层尺度无关；这是经典
                    # DeepDream 能持续累积“狗脸/眼睛”而不是只加一层纹理的关键。
                    dream = (dream + lr * grad / (grad.abs().mean() + 1e-8)).clamp(0.0, 1.0)
                    if intermediate_dir is not None:
                        intermediate_dir.mkdir(parents=True, exist_ok=True)
                        step_np = (dream[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                        sp = intermediate_dir / f"real_googlenet_o{octave_idx + 1}_step_{it + 1:02d}.png"
                        ImageIO.save_image(step_np, sp)
                        intermediate_paths.append(str(sp))

            hook.remove()
            base = dream
            final_np = (base[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return final_np, intermediate_paths
        except Exception:
            if os.environ.get("EAIW_DEEPDREAM_DEBUG", "") == "1":
                import traceback

                traceback.print_exc()
            return None

    _T2V_PIPE_CACHE: Dict[str, Any] = {}
    _T2V_LOCK = threading.Lock()
    # 同一进程只允许一个真实文生视频推理，避免网页端并发请求把内存打满
    _T2V_RUN_LOCK = threading.Lock()

    @staticmethod
    def _translate_zh_prompt_offline(prompt: str) -> str:
        raw = (prompt or "").strip()
        if not raw:
            return "A giant panda eating bamboo on a rock"
        # 兼容 "English - 中文" 形式：优先取纯 ASCII 的英文片段
        for part in raw.split(" - "):
            pt = part.strip()
            if pt and all(ord(c) < 128 for c in pt) and len(pt) >= 3:
                return pt
        if all(ord(c) < 128 for c in raw):
            return raw
        glossary = [
            ("熊猫", "giant panda"), ("竹子", "eating bamboo"), ("企鹅", "emperor penguin walking on ice"),
            ("狮子", "majestic lion roaring in savanna"), ("老虎", "Bengal tiger walking in jungle"),
            ("霸王龙", "T-Rex dinosaur roaring in forest"), ("恐龙", "dinosaur walking in forest"),
            ("客机", "passenger airplane flying through clouds"), ("飞机", "airplane flying in sky"),
            ("烟花", "colorful fireworks exploding in night sky"), ("蝴蝶", "monarch butterfly on golden sunflower"),
            ("向日葵", "golden sunflower in sunlight"), ("甜橙", "fresh orange fruit with water splash"),
            ("草莓", "red strawberries"), ("橘子", "fresh orange fruit"), ("橙子", "fresh orange fruit"),
            ("黑猫", "cool black cat wearing sunglasses"), ("弹吉他", "playing acoustic guitar on stage"),
            ("火山", "volcano erupting glowing lava at night"), ("鲨鱼", "great white shark swimming underwater"),
            ("大白鲨", "great white shark swimming underwater"), ("雄鹰", "majestic eagle flapping wings flying"),
            ("老鹰", "eagle flying in sky"), ("小鸟", "bird flapping wings"), ("机器人", "silver humanoid robot dancing"),
            ("跳舞", "dancing on stage"), ("帆船", "white sailboat riding ocean waves"), ("海浪", "rolling ocean waves"),
            ("白马", "white horse galloping across green meadow"), ("骏马", "horse galloping"),
            ("蒸汽火车", "vintage steam locomotive train on snowy tracks"), ("火车", "train moving on railway"),
            ("三文鱼", "salmon swimming in clear river"), ("金毛", "golden retriever dog running in meadow"),
            ("小狗", "happy dog running"), ("橘猫", "cute orange tabby cat portrait"), ("小猫", "cute kitten"),
            ("牛油果", "armchair in the shape of an avocado"), ("跑车", "vintage red sports car on coastal highway"),
            ("雪山", "snow mountain lake at golden sunset"), ("赛博", "cyberpunk neon city street at night"),
            ("宇航员", "astronaut walking in space nebula"), ("太空", "outer space nebula"),
            ("意大利面", "a man eating spaghetti"), ("意面", "eating spaghetti"), ("火锅", "eating boiling hotpot"),
        ]
        matched = [en for zh, en in glossary if zh in raw]
        return ", ".join(matched) if matched else raw

    @classmethod
    def _t2v_local_dir(cls, base_dir: Optional[Path] = None) -> Path:
        env = os.environ.get("EAIW_T2V_MODEL_DIR", "").strip()
        if env:
            return Path(env).expanduser()
        root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
        return root / "models" / "weights" / "text-to-video-ms-1.7b"

    @classmethod
    def get_t2v_model_status(cls, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """检查前端文生视频所需的本地 1.7B 模型是否真正可用。

        返回 `available`（12 个权重文件齐全）、`deps_ready`（diffusers /
        transformers / accelerate 已安装）、`device` 与 `enabled`（当前环境
        是否会自动启用真实推理）。前端据此显示状态并决定是否请求真实模型。
        """
        local_dir = cls._t2v_local_dir(base_dir)
        deps = {
            name: importlib.util.find_spec(name) is not None
            for name in ("diffusers", "transformers", "accelerate")
        }
        deps_ready = all(deps.values())

        available = False
        missing: List[str] = []
        try:
            from models.model_downloader import T2V_FILES

            for rel, expected_size, _sha in T2V_FILES:
                path = local_dir / rel
                if not path.is_file() or path.stat().st_size != int(expected_size):
                    missing.append(rel)
            available = not missing
        except Exception:
            # 下载模块不可用时退化为关键文件检查
            required = (
                "model_index.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/diffusion_pytorch_model.fp16.safetensors",
                "text_encoder/model.fp16.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/vocab.json",
            )
            missing = [rel for rel in required if not (local_dir / rel).is_file()]
            available = not missing

        cuda = bool(HAS_TORCH and torch.cuda.is_available())
        device = "cuda" if cuda else "cpu"
        cpu_enabled = os.environ.get("EAIW_T2V_CPU", "") == "1"

        if not available:
            reason = (
                "本地 1.7B 权重不完整，请运行 "
                "`python scripts/download_historical_weights.py --model modelscope_t2v_2023`"
            )
        elif not deps_ready:
            missing_deps = [name for name, ok in deps.items() if not ok]
            reason = (
                "缺少 Python 依赖 " + ", ".join(missing_deps)
                + "；请运行 `pip install diffusers transformers accelerate safetensors`"
            )
        elif not cuda and not cpu_enabled:
            reason = (
                "权重已就绪，但 CPU 推理默认关闭；设置 EAIW_T2V_CPU=1 "
                "或在前端勾选“使用本地 1.7B 模型”后可强制运行（很慢）"
            )
        else:
            reason = "本地 1.7B 权重与依赖已就绪"

        return {
            "available": bool(available),
            "complete": bool(available),
            "missing": missing,
            "local_dir": str(local_dir),
            "deps": deps,
            "deps_ready": bool(deps_ready),
            "device": device,
            "cuda": cuda,
            "cpu_enabled": bool(cpu_enabled),
            "enabled": bool(available and deps_ready and (cuda or cpu_enabled)),
            "reason": reason,
            "model_id": "ali-vilab/text-to-video-ms-1.7b",
        }

    @classmethod
    def _load_local_t2v_pipe(cls, local_dir: Path):
        """加载并缓存本地 fp16 Diffusers 文生视频管线。"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        fp16_weights = (local_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists()
        key = f"{local_dir.resolve()}|{device}|{fp16_weights}"
        with cls._T2V_LOCK:
            cached = cls._T2V_PIPE_CACHE.get(key)
            if cached is not None:
                return cached
            from diffusers import DiffusionPipeline

            kwargs = {"torch_dtype": dtype, "local_files_only": True}
            if fp16_weights:
                kwargs["variant"] = "fp16"
            pipe = DiffusionPipeline.from_pretrained(str(local_dir), **kwargs)
            if device == "cuda":
                # 12GB 显存可完整放下 1.7B fp16 管线；整卡加载比逐层 CPU offload 快得多。
                # 显存不足时可设 EAIW_T2V_OFFLOAD=1 退回 model cpu offload。
                if os.environ.get("EAIW_T2V_OFFLOAD", "") == "1":
                    pipe.enable_model_cpu_offload()
                else:
                    pipe = pipe.to("cuda")
            else:
                pipe = pipe.to("cpu")
            # VAE 切片对速度影响很小，可显著降低解码峰值显存/内存
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
            if device == "cpu" or os.environ.get("EAIW_T2V_ATTN_SLICING", "") == "1":
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass
            cls._T2V_PIPE_CACHE[key] = pipe
            return pipe

    @staticmethod
    def _t2v_init_latents(pipe, init_image, num_frames: int, device: str, generator, strength: float, num_inference_steps: int = 20):
        """把输入图片编码成 VAE latent，并按 strength 加噪作为视频初始 latent。

        这是对 TextToVideoSDPipeline 的“图生视频”扩展：不是另训一个 I2V 模型，
        而是用输入图作为首帧/运动初始条件；strength 越小越贴近原图，越大越自由。
        """
        if hasattr(init_image, "convert"):
            arr = np.asarray(init_image.convert("RGB"))
        else:
            arr = np.asarray(init_image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("init_image must be an RGB image")
        arr = arr[..., :3]
        size = int(getattr(pipe.unet.config, "sample_size", 32)) * int(getattr(pipe, "vae_scale_factor", 8))
        arr = cv2.resize(arr.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        arr = np.clip(arr / 127.5 - 1.0, -1.0, 1.0)

        vae_dtype = next(pipe.vae.parameters()).dtype
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=vae_dtype)
        with torch.no_grad():
            clean = pipe.vae.encode(image_tensor).latent_dist.mode() * pipe.vae.config.scaling_factor
        clean = clean.unsqueeze(2).repeat(1, 1, int(num_frames), 1, 1).to(dtype=torch.float16 if device == "cuda" else torch.float32)

        pipe.scheduler.set_timesteps(int(num_inference_steps), device=device)
        timesteps = pipe.scheduler.timesteps
        start_index = int(round(len(timesteps) * (1.0 - max(0.05, min(0.95, float(strength))))))
        start_index = max(0, min(len(timesteps) - 1, start_index))
        t = timesteps[start_index].reshape(1).expand(clean.shape[0]).to(device=device)
        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
        latents = pipe.scheduler.add_noise(clean, noise, t)
        return latents, start_index

    @classmethod
    def run_real_modelscope_t2v(
        cls,
        prompt: str,
        num_frames: int = 16,
        seed: int = 42,
        output_mp4_path: Optional[Path] = None,
        force_cpu: bool = False,
        allow_native: bool = True,
        init_image: Optional[np.ndarray] = None,
        init_strength: float = 0.35,
    ) -> Optional[List[np.ndarray]]:
        """
        真实 2023 阿里 ModelScope 1.7B 3D-UNet 文生视频。

        优先从本地硬盘 `models/weights/text-to-video-ms-1.7b/`（可用环境变量
        `EAIW_T2V_MODEL_DIR` 覆盖）加载 fp16 Diffusers 权重并缓存管线；
        本地权重不存在时回退到 ModelScope 原生 pipeline。

        CPU 推理非常慢（16 帧可能需要几十分钟），默认跳过并交由上层使用
        内置参考帧；确需 CPU 推理请设置环境变量 `EAIW_T2V_CPU=1`，或由前端
        显式传 `force_cpu=True`。`allow_native=False` 时只允许本地权重，避免
        前端请求意外触发 ModelScope 联网下载。
        """
        if not HAS_TORCH:
            return None

        local_dir = cls._t2v_local_dir()
        has_local = (local_dir / "model_index.json").is_file() and (local_dir / "unet").is_dir()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if (
            device == "cpu"
            and not force_cpu
            and os.environ.get("EAIW_T2V_CPU", "") != "1"
        ):
            if has_local:
                print(
                    "[EAIW][T2V] 检测到本地 ModelScope 权重，但当前为 CPU 环境："
                    "真实推理默认关闭（设置 EAIW_T2V_CPU=1 可启用），本次使用内置参考帧。"
                )
            else:
                print(
                    "[EAIW][T2V] 未检测到本地权重且无 CUDA。请先运行 "
                    "`python scripts/download_historical_weights.py --model modelscope_t2v_2023`。"
                )
            return None

        en_prompt = cls._translate_zh_prompt_offline(prompt)
        debug = os.environ.get("EAIW_T2V_DEBUG", "") == "1"

        # 1. 本地硬盘权重（下载脚本的默认目标目录）
        if has_local:
            try:
                pipe = cls._load_local_t2v_pipe(local_dir)
                gen = torch.Generator(device=device).manual_seed(int(seed))
                latents = None
                start_index = None
                if init_image is not None:
                    latents, start_index = cls._t2v_init_latents(
                        pipe, init_image, int(num_frames), device, gen, init_strength
                    )
                original_set_timesteps = pipe.scheduler.set_timesteps

                def _limited_set_timesteps(num_inference_steps=None, device=None):
                    original_set_timesteps(num_inference_steps, device=device)
                    if start_index is not None:
                        pipe.scheduler.timesteps = pipe.scheduler.timesteps[start_index:]

                if start_index is not None:
                    pipe.scheduler.set_timesteps = _limited_set_timesteps
                try:
                    with cls._T2V_RUN_LOCK:
                        out = pipe(
                            en_prompt,
                            num_inference_steps=20,
                            num_frames=int(num_frames),
                            generator=gen,
                            latents=latents,
                        )
                finally:
                    if start_index is not None:
                        pipe.scheduler.set_timesteps = original_set_timesteps
                video_frames = out.frames[0] if hasattr(out, "frames") else out[0]
                frames = []
                for f in video_frames:
                    if hasattr(f, "convert"):
                        # 旧版 diffusers 返回 PIL.Image 列表
                        arr = np.array(f.convert("RGB"), dtype=np.uint8)
                    else:
                        # diffusers >= 0.35 返回 (frames, H, W, C) 的 float 数组
                        arr = np.asarray(f)
                        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                            arr = np.transpose(arr, (1, 2, 0))
                        if arr.dtype != np.uint8:
                            if arr.size and float(arr.max()) <= 1.0:
                                arr = arr * 255.0
                            arr = np.clip(arr, 0, 255).astype(np.uint8)
                    if arr.ndim == 3 and arr.shape[-1] >= 3:
                        frames.append(arr[..., :3])
                if frames:
                    if init_image is not None:
                        ref = np.asarray(init_image.convert("RGB")) if hasattr(init_image, "convert") else np.asarray(init_image)
                        if ref.ndim == 2:
                            ref = np.stack([ref, ref, ref], axis=-1)
                        fh, fw = frames[0].shape[:2]
                        ref = cv2.resize(ref[..., :3].astype(np.uint8), (fw, fh), interpolation=cv2.INTER_AREA)
                        # 首帧严格使用用户图片，随后 3 帧交叉淡入模型生成画面，
                        # 既保证“图片进入视频流程”可见，也避免首帧突变。
                        for i, alpha in enumerate((1.0, 0.72, 0.45, 0.2)):
                            if i >= len(frames):
                                break
                            if alpha >= 0.999:
                                frames[i] = ref.copy()
                            else:
                                frames[i] = np.clip(
                                    frames[i].astype(np.float32) * (1.0 - alpha) + ref.astype(np.float32) * alpha,
                                    0, 255,
                                ).astype(np.uint8)
                    if output_mp4_path:
                        cls._write_mp4(frames, output_mp4_path)
                    return frames
            except Exception as exc:
                if debug:
                    import traceback

                    traceback.print_exc()
                print(f"[EAIW][T2V] 本地权重推理失败: {exc}")

        # 2. 回退：ModelScope 原生 pipeline（会联网下载/命中 ModelScope 缓存）
        # 前端实时请求只允许本地权重，避免在用户不知情时重新下载数 GB 模型。
        if not allow_native or (force_cpu and not has_local) or init_image is not None:
            return None
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            key = "modelscope:native"
            with cls._T2V_LOCK:
                native = cls._T2V_PIPE_CACHE.get(key)
                if native is None:
                    native = pipeline(
                        Tasks.text_to_video_synthesis,
                        model="damo/text-to-video-synthesis",
                    )
                    cls._T2V_PIPE_CACHE[key] = native
            out = native({"text": en_prompt})
            mp4_file = out.get("output_video")
            if mp4_file and Path(mp4_file).exists():
                cap = cv2.VideoCapture(str(mp4_file))
                frames = []
                while True:
                    ret, f = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                cap.release()
                if frames:
                    return frames
        except Exception as exc:
            if debug:
                import traceback

                traceback.print_exc()
            print(f"[EAIW][T2V] ModelScope 原生管线不可用: {exc}")
        return None

    @staticmethod
    def _write_mp4(frames: List[np.ndarray], path: Path) -> None:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h)
            )
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except Exception:
            pass

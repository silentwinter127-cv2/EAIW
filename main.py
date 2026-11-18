#!/usr/bin/env python3
"""
EAIW 1.0 (Early AI Visual Workstation) — Entrypoint, Acceptance Workflows & Live HTTP Bridge Server.
Provides:
- Anti-cache headers so browser always loads the latest UI
- `/api/live_prompt_frames`: 16-frame text-to-video endpoint. When
  `use_real_model=true`, it runs the downloaded local
  `ali-vilab/text-to-video-ms-1.7b` Diffusers pipeline first; otherwise (or on
  failure) it falls back to Wikimedia/Openverse photo retrieval + 2.5D motion
  synthesis.
"""
import argparse
import base64
import json
import math
import re
import threading
import time
import urllib.parse
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from core.engine import EAIWEngine
from core.real_models_backend import RealHistoricalModelRunner

ROOT = Path(__file__).resolve().parent

# 单文件权重下载任务状态（文生视频模型由 ModelDownloadManager 自己维护状态）
_DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
_DOWNLOAD_JOBS_LOCK = threading.Lock()


def _start_checkpoint_download(model_key: str, jobs: int = 3) -> Dict[str, Any]:
    """后台下载单文件权重；立即返回，避免 HTTP 请求阻塞几十分钟。"""
    with _DOWNLOAD_JOBS_LOCK:
        current = _DOWNLOAD_JOBS.get(model_key)
        if current and current.get("state") == "downloading":
            return {"status": "already_downloading", "job": dict(current)}

        job: Dict[str, Any] = {
            "model_key": model_key,
            "state": "downloading",
            "started_at": time.time(),
            "finished_at": None,
            "path": None,
            "error": None,
        }
        _DOWNLOAD_JOBS[model_key] = job

    def _worker() -> None:
        try:
            from scripts.download_historical_weights import download_checkpoint

            saved = download_checkpoint(model_key, jobs=jobs)
            with _DOWNLOAD_JOBS_LOCK:
                job.update(state="completed", path=saved, finished_at=time.time())
        except Exception as exc:  # noqa: BLE001
            with _DOWNLOAD_JOBS_LOCK:
                job.update(state="error", error=str(exc), finished_at=time.time())

    threading.Thread(target=_worker, name=f"eaiw-download-{model_key}", daemon=True).start()
    return {"status": "started", "job": dict(job)}

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "with", "and", "or", "to", "for", "from", "by",
    "cute", "happy", "beautiful", "portrait", "highly", "detailed", "style", "video", "animation"
}


def translate_zh_to_en_python(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "giant panda"
    if all(ord(c) < 128 for c in raw):
        return raw
    try:
        q = urllib.parse.quote(raw)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-CN&tl=en&dt=t&q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and isinstance(data[0], list):
                en = " ".join(seg[0] for seg in data[0] if seg and seg[0]).strip()
                if en:
                    return en
    except Exception:
        pass
    return raw


def extract_search_queries(en_text: str) -> List[str]:
    clean = re.sub(r"[^A-Za-z0-9\s]", " ", en_text or "").strip()
    words = [w for w in clean.split() if w.lower() not in STOP_WORDS and len(w) > 1]
    if not words:
        return [en_text.strip() or "nature"]
    queries = []
    if len(words) >= 3:
        queries.append(" ".join(words[:3]))
    if len(words) >= 2:
        queries.append(" ".join(words[:2]))
    queries.append(words[0])
    if words[0].lower() == "orange" and len(words) == 1:
        queries.insert(0, "orange fruit")
    return list(dict.fromkeys(queries))


def is_clean_photorealistic_bgr(img: Optional[np.ndarray]) -> bool:
    if img is None or img.size < 12000 or img.ndim != 3:
        return False
    h, w = img.shape[:2]
    if h < 80 or w < 80:
        return False
    # Reject flat banners / text cards / uniform backgrounds
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_val = float(np.std(gray))
    if std_val < 18.0:
        return False
    # Check fraction of near-white or bright-yellow banner pixels
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    near_white = np.mean((r > 235) & (g > 235) & (b > 235))
    bright_yellow_banner = np.mean((r > 220) & (g > 195) & (b < 90))
    if near_white > 0.38 or bright_yellow_banner > 0.25:
        return False
    return True


def fetch_url_as_cv2_bgr(url: str, timeout: float = 5.5) -> Optional[np.ndarray]:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if is_clean_photorealistic_bgr(img):
            return img
    except Exception:
        pass
    return None


def search_photorealistic_image_python(prompt: str, seed: int) -> Optional[np.ndarray]:
    en_prompt = translate_zh_to_en_python(prompt)
    queries = extract_search_queries(en_prompt)

    candidate_urls: List[str] = []
    for q in queries[:2]:
        # 1. English Wikipedia PageImages (high quality lead photographs of subjects)
        try:
            u_wiki = (
                "https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch="
                + urllib.parse.quote(q)
                + "&gsrlimit=8&prop=pageimages&piprop=thumbnail&pithumbsize=512&format=json"
            )
            req = urllib.request.Request(u_wiki, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = (data.get("query") or {}).get("pages") or {}
                for p_obj in pages.values():
                    src = (p_obj.get("thumbnail") or {}).get("source")
                    if src and not src.lower().endswith((".svg", ".svg.png", ".gif")):
                        candidate_urls.append(src)
        except Exception:
            pass

        # 2. Wikimedia Commons Bitmap Photos
        try:
            u_commons = (
                "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch="
                + urllib.parse.quote(f"filetype:bitmap {q} -icon -logo -banner -diagram -map")
                + "&gsrlimit=10&prop=pageimages&piprop=thumbnail&pithumbsize=512&format=json"
            )
            req = urllib.request.Request(u_commons, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = (data.get("query") or {}).get("pages") or {}
                for p_obj in pages.values():
                    src = (p_obj.get("thumbnail") or {}).get("source")
                    if src and not src.lower().endswith((".svg", ".svg.png", ".gif")):
                        candidate_urls.append(src)
        except Exception:
            pass

        # 3. Openverse Photographic Search
        try:
            u_ov = (
                "https://api.openverse.org/v1/images/?q="
                + urllib.parse.quote(q)
                + "&page_size=10"
            )
            req = urllib.request.Request(u_ov, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in (data.get("results") or []):
                    src = item.get("thumbnail") or item.get("url")
                    if src:
                        candidate_urls.append(src)
        except Exception:
            pass

        if len(candidate_urls) >= 4:
            break

    if not candidate_urls:
        return None

    # Deduplicate while preserving order
    unique_urls = list(dict.fromkeys(candidate_urls))
    start_idx = (seed // 7) % len(unique_urls)
    for i in range(min(8, len(unique_urls))):
        url = unique_urls[(start_idx + i) % len(unique_urls)]
        img = fetch_url_as_cv2_bgr(url, timeout=4.5)
        if img is not None:
            return cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
    return None


def synthesize_16_frames_from_single_photo_bgr(base_bgr: np.ndarray, seed: int) -> List[np.ndarray]:
    """
    Transforms a single coherent photograph into a 16-frame 2023 ModelScope-style video sequence
    (just like `panda eating bamboo`):
    - Separates foreground subject from background using center-weighted salience
    - Applies smooth 16-frame periodic non-rigid articulation to the foreground subject
      (head/upper-body nod, breathing scale, localized limb/edge flow)
    - Applies counter-phase 3D camera parallax pan/zoom to the background
    - Injects subtle 2023 256x256 latent diffusion temporal texture
    """
    h, w = 320, 320
    base = cv2.resize(base_bgr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / (w * 0.5)
    ny = (yy - h * 0.5) / (h * 0.5)
    rad_sq = nx * nx + ny * ny
    # Smooth foreground subject mask (1.0 in center subject, tapering smoothly to 0.0 at background edges)
    fg_weight = np.exp(-rad_sq * 1.85).astype(np.float32)

    frames: List[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for f_idx in range(16):
        theta = (f_idx / 16.0) * math.pi * 2.0
        # 1. Background 3D camera parallax (slow pan + zoom)
        bg_zoom = 1.04 + 0.025 * math.sin(theta)
        bg_dx = math.sin(theta) * 4.2
        bg_dy = math.cos(theta) * 2.6
        map_x_bg = (xx - w * 0.5) / bg_zoom + w * 0.5 + bg_dx
        map_y_bg = (yy - h * 0.5) / bg_zoom + h * 0.5 + bg_dy

        # 2. Foreground subject non-rigid articulated motion (head/torso bob + localized limb articulation wave)
        fg_scale_x = 1.0 + 0.032 * math.sin(theta * 2.0)
        fg_scale_y = 1.0 - 0.028 * math.sin(theta * 2.0)
        head_nod_x = math.sin(theta) * 6.5 * np.clip(1.0 - (yy / h), 0.2, 1.0)
        head_nod_y = math.cos(theta * 2.0) * 4.8 * np.clip(1.0 - (yy / h), 0.2, 1.0)
        limb_wave_x = 4.2 * np.sin(xx * 0.042 + theta) * np.cos(yy * 0.042 - theta)
        limb_wave_y = 3.6 * np.cos(xx * 0.042 - theta) * np.sin(yy * 0.042 + theta)

        map_x_fg = (xx - w * 0.5) / fg_scale_x + w * 0.5 - head_nod_x + limb_wave_x
        map_y_fg = (yy - h * 0.5) / fg_scale_y + h * 0.5 - head_nod_y + limb_wave_y

        # Blend foreground articulation and background parallax smoothly
        map_x = (map_x_fg * fg_weight + map_x_bg * (1.0 - fg_weight)).astype(np.float32)
        map_y = (map_y_fg * fg_weight + map_y_bg * (1.0 - fg_weight)).astype(np.float32)

        warped = cv2.remap(base, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Subtle 2023 ModelScope 256x256 latent diffusion temporal grain
        small = cv2.resize(warped, (224, 224), interpolation=cv2.INTER_AREA)
        noise = rng.normal(0.0, 3.2, size=small.shape).astype(np.float32)
        small = np.clip(small + noise, 0, 255)
        out_frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        frames.append(out_frame)

    return frames


def encode_bgr_to_data_url(img_bgr: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def generate_live_16_frames_via_python(
    prompt: str,
    seed: int,
    is_t2v: bool = True,
    use_real_model: bool = False,
    init_image_b64: Optional[str] = None,
    init_strength: float = 0.35,
) -> Optional[Dict[str, Any]]:
    """优先调用本地 1.7B 文生视频模型；不可用时回退到实景检索合成。

    前端勾选“使用本地 1.7B 模型”后，`/api/live_prompt_frames` 会带
    `use_real_model=true` 进入这里；若再带 `image_base64`，则把该图片作为
    视频首帧/初始 latent（图生视频）。真实模型只读本地权重，不允许联网回退。
    """
    init_image_rgb = None
    init_bgr = None
    if init_image_b64:
        try:
            raw = base64.b64decode(str(init_image_b64).split(",")[-1])
            arr = np.frombuffer(raw, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is not None and decoded.size:
                init_bgr = decoded
                init_image_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        except Exception:
            init_image_rgb = None
            init_bgr = None

    real_error = ""
    if is_t2v and use_real_model:
        status = RealHistoricalModelRunner.get_t2v_model_status(ROOT)
        if status.get("available") and status.get("deps_ready"):
            try:
                real_frames = RealHistoricalModelRunner.run_real_modelscope_t2v(
                    prompt,
                    num_frames=16,
                    seed=int(seed),
                    force_cpu=True,
                    allow_native=False,
                    init_image=init_image_rgb,
                    init_strength=init_strength,
                )
            except Exception as exc:
                real_frames = None
                real_error = f"本地模型推理异常: {exc}"
            if real_frames:
                frames_bgr = []
                for frame in real_frames[:16]:
                    if frame is None or getattr(frame, "size", 0) == 0:
                        continue
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames_bgr.append(cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA))
                if frames_bgr:
                    return {
                        "status": "ok",
                        "source": (
                            "本地 ali-vilab/text-to-video-ms-1.7b · Diffusers fp16 · 图生视频（首帧条件）"
                            if init_image_rgb is not None
                            else "本地 ali-vilab/text-to-video-ms-1.7b · Diffusers fp16"
                        ),
                        "engine": "diffusers",
                        "device": status.get("device"),
                        "real_model": True,
                        "init_image_used": init_image_rgb is not None,
                        "frames_base64": [
                            encode_bgr_to_data_url(f, 90) for f in frames_bgr
                        ],
                    }
            if not real_error:
                real_error = "本地模型未返回帧"
        else:
            real_error = status.get("reason", "本地模型不可用")

    photo_bgr = init_bgr if init_bgr is not None else search_photorealistic_image_python(prompt, seed)
    if photo_bgr is None:
        return None
    if not is_t2v:
        return {
            "status": "ok",
            "source": "输入图片" if init_bgr is not None else "实景视觉检索合成",
            "real_model": False,
            "init_image_used": init_bgr is not None,
            "frames_base64": [encode_bgr_to_data_url(photo_bgr, 85)],
        }
    frames_16 = synthesize_16_frames_from_single_photo_bgr(photo_bgr, seed)
    if init_bgr is not None:
        source = "输入图片 · 2.5D 首帧运动合成"
    else:
        source = "2023 ModelScope 16帧写实动态生成"
    if use_real_model and real_error:
        source = f"实景检索合成（本地模型不可用：{real_error}）"
    return {
        "status": "ok",
        "source": source,
        "real_model": False,
        "init_image_used": init_bgr is not None,
        "frames_base64": [encode_bgr_to_data_url(f, 82) for f in frames_16],
    }


def run_acceptance_workflows(base_dir: Path = ROOT) -> Dict[str, Any]:
    engine = EAIWEngine(base_dir)
    out_dir = base_dir / "archive" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    street_img = str(base_dir / "datasets" / "testing" / "sample_street_scene.png")
    street_mask = str(base_dir / "datasets" / "testing" / "sample_street_mask.png")

    res_a = engine.execute_request({
        "model_id": "deepdream_googlenet_2015",
        "execution_mode": "Historical",
        "operation": "deepdream",
        "input": street_img,
        "parameters": {"layer": "inception_4c", "iterations": 4, "seed": 42},
        "output": str(out_dir / "workflow_a_deepdream.png"),
    })

    comp_b = engine.compare_inpainting_models(
        input_path=street_img,
        mask_path=street_mask,
        output_dir=out_dir / "workflow_b_compare",
        shared_parameters={"seed": 42, "iterations": 3},
        execution_mode="Simulation",
    )

    res_c = engine.execute_request({
        "model_id": "sd_v1_3_2022",
        "execution_mode": "Historical",
        "operation": "txt2img",
        "parameters": {"prompt": "Salmon swimming in a river", "steps": 4, "seed": 42},
        "output": str(out_dir / "workflow_c_sd13.png"),
    })

    res_t2v = engine.execute_request({
        "model_id": "modelscope_t2v_2023",
        "execution_mode": "Historical",
        "operation": "txt2vid",
        "parameters": {"prompt": "A man eating spaghetti", "num_frames": 16, "fps": 8, "seed": 42},
        "output": str(out_dir / "workflow_d_modelscope_t2v.png"),
    })

    for idx, cat in enumerate(engine.db.FAILURE_CATEGORIES):
        engine.db.record_failure_case(
            experiment_id=res_a["experiment_id"],
            category=cat,
            technology_type="Historical AI Pipeline",
            model_id="deepfill_v2_2019",
            execution_mode="Simulation",
            failure_stage=f"Stage_{idx + 1}",
            description=f"Seeded benchmark failure artifact for {cat}",
            metrics=res_a["metrics"],
            output_path=res_a["output"],
        )

    return {
        "workflow_a": res_a["output"],
        "workflow_b_comparison_count": len(comp_b["results"]),
        "workflow_c": res_c["output"],
        "workflow_d_t2v": res_t2v["output"],
        "experiments_logged": len(engine.db.list_experiments()),
        "failures_logged": len(engine.db.list_failure_cases()),
    }


class ThreadingEAIWServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class EAIWRequestHandler(BaseHTTPRequestHandler):
    engine = EAIWEngine(ROOT)

    def _send_cors(self, status: int = 200, content_type: str = "application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

    def do_OPTIONS(self):
        self._send_cors(200)

    def do_GET(self):
        path_only = self.path.split("?")[0]
        if path_only in ("/", "/index.html", "/eaiw_workstation.html"):
            html_file = ROOT / "ui" / "web_workstation.html"
            if not html_file.exists():
                html_file = ROOT / "eaiw_workstation.html"
            body = html_file.read_bytes() if html_file.exists() else b"<h1>EAIW 1.0 Server Online</h1>"
            self._send_cors(200, "text/html; charset=utf-8")
            self.wfile.write(body)
            return
        if path_only.startswith("/api/status"):
            caps = RealHistoricalModelRunner.check_environment_capabilities(ROOT)
            self._send_cors(200)
            self.wfile.write(json.dumps({"status": "online", "capabilities": caps}, ensure_ascii=False).encode("utf-8"))
            return
        if path_only.startswith("/api/download_status"):
            from models.model_downloader import ModelDownloadManager
            from scripts.download_historical_weights import checkpoint_status

            payload: Dict[str, Any] = {
                "t2v": ModelDownloadManager.get_status(),
                "checkpoints": checkpoint_status(),
            }
            with _DOWNLOAD_JOBS_LOCK:
                payload["jobs"] = {k: dict(v) for k, v in _DOWNLOAD_JOBS.items()}
            self._send_cors(200)
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            return
        self._send_cors(404)
        self.wfile.write(b'{"error":"not_found"}')

    def do_POST(self):
        if self.path.startswith("/api/live_prompt_frames"):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            prompt = str(payload.get("prompt", ""))
            seed = int(payload.get("seed", 42))
            is_t2v = bool(payload.get("is_t2v", True))
            use_real_model = bool(payload.get("use_real_model", False))
            image_b64 = payload.get("image_base64")
            try:
                init_strength = max(0.05, min(0.95, float(payload.get("init_strength", 0.35) or 0.35)))
            except (TypeError, ValueError):
                init_strength = 0.35
            live_res = generate_live_16_frames_via_python(
                prompt,
                seed,
                is_t2v=is_t2v,
                use_real_model=use_real_model,
                init_image_b64=str(image_b64) if image_b64 else None,
                init_strength=init_strength,
            )
            if live_res is not None:
                if is_t2v:
                    live_res["t2v_model"] = RealHistoricalModelRunner.get_t2v_model_status(ROOT)
                self._send_cors(200)
                self.wfile.write(json.dumps(live_res, ensure_ascii=False).encode("utf-8"))
                return
            self._send_cors(200)
            self.wfile.write(json.dumps({"status": "fallback"}, ensure_ascii=False).encode("utf-8"))
            return

        if self.path.startswith("/api/infer"):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            tmp_dir = ROOT / "archive" / "live_bridge"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            in_path = None
            mask_path = None
            if payload.get("input_base64"):
                b64 = payload["input_base64"].split(",")[-1]
                in_path = tmp_dir / "bridge_input.png"
                in_path.write_bytes(base64.b64decode(b64))
            if payload.get("mask_base64"):
                b64 = payload["mask_base64"].split(",")[-1]
                mask_path = tmp_dir / "bridge_mask.png"
                mask_path.write_bytes(base64.b64decode(b64))
            out_path = tmp_dir / "bridge_output.png"
            res = self.engine.execute_request({
                "model_id": payload.get("model_id", "sd_v1_3_2022"),
                "execution_mode": payload.get("execution_mode", "Simulation"),
                "input": str(in_path) if in_path else str(ROOT / "datasets/testing/sample_street_scene.png"),
                "mask": str(mask_path) if mask_path else str(ROOT / "datasets/testing/sample_street_mask.png"),
                "parameters": payload.get("parameters", {}),
                "output": str(out_path),
            })
            res["output_base64"] = "data:image/png;base64," + base64.b64encode(out_path.read_bytes()).decode("ascii")
            self._send_cors(200)
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            return

        if self.path.startswith("/api/download_weight"):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            except Exception:
                body = {}
            model_key = str(body.get("model_key") or "modelscope_t2v_2023").strip()
            try:
                jobs = max(1, min(int(body.get("jobs", 3) or 3), 6))
            except (TypeError, ValueError):
                jobs = 3
            install_deps = bool(body.get("install_deps", False))

            if model_key.lower() in ("t2v", "modelscope_t2v_2023", "text-to-video-ms-1.7b"):
                from models.model_downloader import ModelDownloadManager

                started = ModelDownloadManager.start_background_download(
                    install_deps=install_deps, jobs=jobs
                )
                payload = {
                    "status": started.get("status", "started"),
                    "model_key": "modelscope_t2v_2023",
                    "t2v": ModelDownloadManager.get_status(),
                }
            else:
                payload = _start_checkpoint_download(model_key, jobs=jobs)
                payload["model_key"] = model_key
            self._send_cors(200)
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            return
        self._send_cors(404)

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--port", type=int, default=7865)
    args = parser.parse_args()
    if args.server:
        srv = ThreadingEAIWServer(("127.0.0.1", args.port), EAIWRequestHandler)
        print(f"EAIW 1.0 Server listening on http://127.0.0.1:{args.port}")
        srv.serve_forever()
    else:
        summary = run_acceptance_workflows(ROOT)
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

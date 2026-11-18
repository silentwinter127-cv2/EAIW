"""
EAIW 1.0 — Multi-Threaded Local Real Model HTTP Bridge & Web Workstation Server (http://127.0.0.1:7865).
Uses ThreadingHTTPServer + explicit Content-Length + Cache-Control: no-store so repeated
or rapid clicks on 'Execute Workstation Node' never block, hang, or return stale cache.
"""
import base64
import json
import time
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from core.engine import EAIWEngine
from core.real_models_backend import RealHistoricalModelRunner

_REQ_COUNTER = 0
_REQ_LOCK = threading.Lock()

# 单文件下载任务状态；文生视频模型的状态由 ModelDownloadManager 维护。
_DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}
_DOWNLOAD_LOCK = threading.Lock()


def _start_checkpoint_job(model_key: str, jobs: int = 3) -> Dict[str, Any]:
    with _DOWNLOAD_LOCK:
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
            with _DOWNLOAD_LOCK:
                job.update(state="completed", path=saved, finished_at=time.time())
        except Exception as exc:  # noqa: BLE001
            with _DOWNLOAD_LOCK:
                job.update(state="error", error=str(exc), finished_at=time.time())

    threading.Thread(target=_worker, name=f"eaiw-download-{model_key}", daemon=True).start()
    return {"status": "started", "job": dict(job)}


def _next_req_id() -> int:
    global _REQ_COUNTER
    with _REQ_LOCK:
        _REQ_COUNTER += 1
        return _REQ_COUNTER


def make_handler(engine: EAIWEngine, base_dir: Path):
    class EAIWBridgeHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, status: int, content_type: str, payload: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: int, data: dict):
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def do_OPTIONS(self):
            self._send_bytes(200, "text/plain; charset=utf-8", b"OK")

        def do_GET(self):
            clean_path = self.path.split("?")[0]
            if clean_path in ("/", "/index.html", "/workstation", "/eaiw_workstation.html"):
                ui_file = base_dir / "ui" / "web_workstation.html"
                if not ui_file.exists():
                    ui_file = base_dir / "eaiw_workstation.html"
                if ui_file.exists():
                    self._send_bytes(200, "text/html; charset=utf-8", ui_file.read_bytes())
                    return

            if clean_path.startswith("/api/status"):
                caps = RealHistoricalModelRunner.check_environment_capabilities(base_dir)
                self._send_json(200, {
                    "status": "online",
                    "server": "EAIW 1.0 Multi-Threaded Local Historical Model Server",
                    "capabilities": caps,
                    "experiments_count": len(engine.db.list_experiments(limit=100)),
                    "failures_count": len(engine.db.list_artifact_failures()),
                })
                return

            if clean_path.startswith("/api/experiments"):
                self._send_json(200, {
                    "experiments": engine.db.list_experiments(limit=30),
                    "failures": engine.db.list_artifact_failures(),
                })
                return

            if clean_path.startswith("/api/download_status"):
                from models.model_downloader import ModelDownloadManager
                from scripts.download_historical_weights import checkpoint_status

                with _DOWNLOAD_LOCK:
                    jobs = {k: dict(v) for k, v in _DOWNLOAD_JOBS.items()}
                self._send_json(200, {
                    "t2v": ModelDownloadManager.get_status(),
                    "checkpoints": checkpoint_status(),
                    "jobs": jobs,
                })
                return

            self._send_json(404, {"error": "Not found", "hint": "Visit http://127.0.0.1:7865/ for EAIW Workstation UI"})

        def do_POST(self):
            clean_path = self.path.split("?")[0]
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            body = json.loads(raw_body)

            if clean_path.startswith("/api/infer"):
                req_id = _next_req_id()
                tmp_dir = base_dir / "archive" / "live_bridge"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                if body.get("input_base64"):
                    b64 = body["input_base64"].split(",")[-1]
                    in_p = tmp_dir / f"bridge_input_{req_id}.png"
                    in_p.write_bytes(base64.b64decode(b64))
                    body["input"] = str(in_p)

                if body.get("mask_base64"):
                    b64 = body["mask_base64"].split(",")[-1]
                    m_p = tmp_dir / f"bridge_mask_{req_id}.png"
                    m_p.write_bytes(base64.b64decode(b64))
                    body["mask"] = str(m_p)

                model_id = body.get("model_id", "model")
                body["output"] = str(tmp_dir / f"bridge_out_{model_id}_{req_id}.png")
                res = engine.execute_request(body)
                out_bytes = Path(res["output"]).read_bytes()
                res["output_base64"] = "data:image/png;base64," + base64.b64encode(out_bytes).decode("ascii")
                res["req_id"] = req_id

                inter_b64 = []
                for ip in res.get("intermediate_paths", [])[:8]:
                    p_obj = Path(ip)
                    if p_obj.exists():
                        inter_b64.append("data:image/png;base64," + base64.b64encode(p_obj.read_bytes()).decode("ascii"))
                res["intermediate_base64_list"] = inter_b64

                self._send_json(200, res)
                return

            if clean_path.startswith("/api/download_weight"):
                from models.model_downloader import ModelDownloadManager

                model_key = str(body.get("model_key") or "modelscope_t2v_2023").strip()
                try:
                    jobs = max(1, min(int(body.get("jobs", 3) or 3), 6))
                except (TypeError, ValueError):
                    jobs = 3
                if model_key.lower() in ("t2v", "modelscope_t2v_2023", "text-to-video-ms-1.7b"):
                    started = ModelDownloadManager.start_background_download(
                        install_deps=bool(body.get("install_deps", False)), jobs=jobs
                    )
                    self._send_json(200, {
                        "status": started.get("status", "started"),
                        "model_key": "modelscope_t2v_2023",
                        "t2v": ModelDownloadManager.get_status(),
                    })
                else:
                    self._send_json(200, _start_checkpoint_job(model_key, jobs=jobs))
                return

            self._send_json(404, {"error": "Unknown POST endpoint"})

        def log_message(self, fmt, *args):
            return

    return EAIWBridgeHandler


def run_bridge_server(base_dir: Path, host: str = "127.0.0.1", port: int = 7865, open_browser: bool = True):
    engine = EAIWEngine(base_dir)
    handler = make_handler(engine, base_dir)
    srv = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print("=" * 68)
    print(f"  EAIW 1.0 — Early AI Visual Workstation Local Server is RUNNING!")
    print(f"  -> Open in Browser: {url}")
    print(f"  -> API Status URL : {url}api/status")
    print("=" * 68)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    srv.serve_forever()

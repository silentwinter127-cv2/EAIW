from __future__ import annotations

from . import bootstrap  # noqa: F401  (must precede torch)

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from .config import DEFAULT_PROMPT, M_LAYERS, OUTPUTS, V_LAYERS
from .engine import DreamEngine
from .clip_runtime import ClipRuntime


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ENGINE = DreamEngine()
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
ACTIVE: Dict[str, Any] = {"id": None}


def _new_job(payload: Dict[str, Any]) -> str:
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {
            "id": jid,
            "state": "queued",
            "created": time.time(),
            "step": 0,
            "steps": int(payload.get("steps", 120)),
            "preview": None,
            "result": None,
            "error": None,
            "prompt": payload.get("prompt", DEFAULT_PROMPT),
        }
        ACTIVE["id"] = jid

    def worker() -> None:
        with JOBS_LOCK:
            JOBS[jid]["state"] = "running"
        try:
            def progress(p: Dict[str, Any]) -> None:
                with JOBS_LOCK:
                    JOBS[jid].update({
                        "step": p.get("step", 0),
                        "steps": p.get("steps", 0),
                        "preview": p.get("preview"),
                        "loss": p.get("loss"),
                        "elapsed": p.get("elapsed"),
                    })

            result = ENGINE.run(progress=progress, **payload)
            with JOBS_LOCK:
                JOBS[jid].update({"state": "completed", "result": result, "preview": result.get("final")})
        except Exception as exc:  # noqa: BLE001
            with JOBS_LOCK:
                JOBS[jid].update({"state": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            with JOBS_LOCK:
                ACTIVE["id"] = None

    threading.Thread(target=worker, name=f"dream-{jid}", daemon=True).start()
    return jid


class Handler(BaseHTTPRequestHandler):
    server_version = "TransformerDream/0.1"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(200, "text/plain", b"OK")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            f = WEB / "index.html"
            if not f.exists():
                return self._json(500, {"ok": False, "error": "web/index.html missing"})
            return self._send(200, "text/html; charset=utf-8", f.read_bytes())
        if path == "/api/status":
            info = ENGINE.runtime.status()
            info.update(ClipRuntime.get().status())
            with JOBS_LOCK:
                info["active_job"] = ACTIVE["id"]
            return self._json(200, {"ok": True, **info, "outputs": str(OUTPUTS)})
        if path.startswith("/api/job/"):
            jid = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(jid)
            if not job:
                return self._json(404, {"ok": False, "error": "job not found"})
            return self._json(200, {"ok": True, **job})
        return self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception as exc:  # noqa: BLE001
            return self._json(400, {"ok": False, "error": f"bad json: {exc}"})

        if path == "/api/run":
            with JOBS_LOCK:
                if ACTIVE["id"]:
                    return self._json(409, {"ok": False, "error": "another dream job is running", "active": ACTIVE["id"]})
            payload = {
                "prompt": str(data.get("prompt") or DEFAULT_PROMPT),
                "mode": str(data.get("mode") or "direction"),
                "layer": int(data.get("layer", 24)),
                "vision_layer": int(data.get("vision_layer", 16)),
                "steps": int(data.get("steps", 240)),
                "size": int(data.get("size", 256)),
                "lr": float(data.get("lr", 0.02)),
                "tv_weight": float(data.get("tv_weight", 2.0)),
                "seed": int(data.get("seed", 1)),
                "target_word": str(data.get("target_word") or "crown"),
                "channel": int(data.get("channel", 128)),
                "negative": str(data.get("negative") or "a plain background"),
                "init_image_b64": data.get("init_image_b64"),
            }
            jid = _new_job(payload)
            return self._json(200, {"ok": True, "job": jid})

        if path == "/api/extract":
            try:
                out = ENGINE.extract(
                    str(data.get("prompt") or DEFAULT_PROMPT),
                    int(data.get("size", 256)),
                    [int(x) for x in (data.get("vision_layers") or [8, 16, 24])],
                )
                return self._json(200, out)
            except Exception as exc:  # noqa: BLE001
                return self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        if path == "/api/unload":
            with JOBS_LOCK:
                if ACTIVE["id"]:
                    return self._json(409, {"ok": False, "error": "cannot unload while a job is running"})
            ENGINE.runtime.unload()
            ClipRuntime.get().unload()
            info = ENGINE.runtime.status()
            info.update(ClipRuntime.get().status())
            return self._json(200, {"ok": True, **info})

        return self._json(404, {"ok": False, "error": "not found"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[dream] {self.address_string()} {fmt % args}")


def main(port: int = 7866) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    print(f"Transformer Dream Engine: http://127.0.0.1:{port}")
    print(f"Qwen dir: {ENGINE.runtime.status()['model_dir']}")
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7866)
    main(ap.parse_args().port)

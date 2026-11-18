"""EAIW 1.0 — 2023 ModelScope 文生视频权重下载器（多镜像、断点续传、SHA-256 校验）。

目标模型：``ali-vilab/text-to-video-ms-1.7b``（ModelScope 镜像
``AI-ModelScope/text-to-video-ms-1.7b``），Diffusers fp16 格式，共 12 个文件、
约 3.6 GB。下载目录固定为 ``models/weights/text-to-video-ms-1.7b/``，下载完成后
``DiffusionPipeline.from_pretrained(..., variant="fp16")`` 可直接离线加载。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models.download_utils import (
    DownloadCancelled,
    DownloadError,
    download_file,
    human_bytes,
    human_seconds,
    sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "ali-vilab/text-to-video-ms-1.7b"
MODELSCOPE_MODEL_ID = "AI-ModelScope/text-to-video-ms-1.7b"
MODEL_LOCAL_DIR = ROOT / "models" / "weights" / "text-to-video-ms-1.7b"
MANIFEST_NAME = "download_manifest.json"

# (相对路径, 精确字节数, SHA-256) —— 与 HuggingFace / ModelScope 上游逐字节一致。
# 这些数值用于校验；不要再凭“约等于”判断文件是否完整。
T2V_FILES: List[Tuple[str, int, str]] = [
    ("model_index.json", 384, "3b800626383c7f9794d5c97ccac5ff006a523806550162279b2b76455872dd47"),
    ("scheduler/scheduler_config.json", 465, "35442fb22160d0123e0d2f8ee345c5afe08c2601ab6ca7dff870f8befc228e0d"),
    ("tokenizer/tokenizer_config.json", 755, "fdd524218f0554acb77cdecf36807c01f4d1ad2574fd6345d0bdef48873c3d14"),
    ("tokenizer/special_tokens_map.json", 460, "f118ab3a983206e4f32583448de6bd6aae4ee21869135cef1f5848a753cdaab6"),
    ("tokenizer/merges.txt", 524619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
    ("tokenizer/vocab.json", 1059962, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
    ("text_encoder/config.json", 644, "57fb3a8ed312c54cdea7fa3598ff4ef8b5777b0b5bc8ba07b3ddf17ad30aacbe"),
    ("vae/config.json", 657, "28e3c55b3265fdb5138e6330b71b372a499a91afe836bfdb5b58e2e2a43ca15a"),
    ("unet/config.json", 787, "baf717dc4b75250d12e2b60df5af8339b3373b16760ae687e35fc388c3aa9329"),
    ("vae/diffusion_pytorch_model.fp16.safetensors", 167335350, "c8b011e5a18c53888d51a81aa28223ddec87b450c14dc9650d9c3ebbcd17624e"),
    ("text_encoder/model.fp16.safetensors", 680821102, "6ab158327d06ce861b5c78843672c78433ea82fcf03f142097ba204b81251cd2"),
    ("unet/diffusion_pytorch_model.fp16.safetensors", 2822650226, "f972e78d39735baa09aab1cf60dd19f213e32f1c5a89382385e1a7f647b2e537"),
]

# 旧脚本/旧接口使用的 (路径, 大小) 视图，保留以兼容。
T2V_MS_1_7B_FILES: List[Tuple[str, int]] = [(rel, size) for rel, size, _ in T2V_FILES]

T2V_TOTAL_BYTES = sum(size for _, size, _ in T2V_FILES)
PARALLEL_FILE_THRESHOLD = 16 << 20  # 小于 16 MB 的配置/分词器文件顺序下载，大文件并行

MIRRORS: List[Tuple[str, str]] = [
    # ModelScope 的 CDN 在国内通常最快（实测 6+ MB/s），优先使用。
    ("modelscope", f"https://modelscope.cn/models/{MODELSCOPE_MODEL_ID}/resolve/master"),
    ("hf-mirror", f"https://hf-mirror.com/{MODEL_ID}/resolve/main"),
    ("huggingface", f"https://huggingface.co/{MODEL_ID}/resolve/main"),
]


def model_dir() -> Path:
    """返回权重目录（可通过环境变量 EAIW_T2V_MODEL_DIR 覆盖，便于测试/迁移）。"""
    override = os.environ.get("EAIW_T2V_MODEL_DIR")
    return Path(override).expanduser() if override else MODEL_LOCAL_DIR


class ModelDownloadManager:
    """全局单例式下载管理器：一个后台线程，状态可被 HTTP 接口轮询。"""

    _lock = threading.Lock()
    _cancel = threading.Event()
    _thread: Optional[threading.Thread] = None
    _last_bytes = 0
    _last_time = 0.0
    _state: Dict[str, Any] = {
        "status": "idle",  # idle | downloading | completed | error | cancelled
        "target_dir": str(MODEL_LOCAL_DIR),
        "model_id": MODEL_ID,
        "current_file": "",
        "current_files": [],
        "file_index": 0,
        "total_files": len(T2V_FILES),
        "downloaded_bytes": 0,
        "total_bytes": T2V_TOTAL_BYTES,
        "percent": 0.0,
        "speed_bps": 0.0,
        "speed_mbps": 0.0,
        "eta_seconds": None,
        "message": "尚未下载。运行 python scripts/download_historical_weights.py --model modelscope_t2v_2023 开始下载。",
        "error": "",
        "logs": [],
        "files": {},
    }

    # ------------------------------------------------------------------ 状态
    @classmethod
    def _reset_state_locked(cls) -> None:
        cls._state.update(
            {
                "status": "downloading",
                "current_file": "",
                "current_files": [],
                "file_index": 0,
                "downloaded_bytes": 0,
                "total_bytes": T2V_TOTAL_BYTES,
                "percent": 0.0,
                "speed_bps": 0.0,
                "speed_mbps": 0.0,
                "eta_seconds": None,
                "error": "",
                "logs": [],
                "files": {
                    rel: {
                        "path": rel,
                        "size": size,
                        "sha256": sha,
                        "downloaded": 0,
                        "status": "pending",
                        "speed_bps": 0.0,
                    }
                    for rel, size, sha in T2V_FILES
                },
            }
        )
        cls._last_bytes = 0
        cls._last_time = time.time()

    @classmethod
    def is_model_completely_downloaded(cls) -> bool:
        """仅按精确字节数判断（不做 SHA-256，避免每次轮询都读 3.6 GB）。"""
        target = model_dir()
        for rel, size, _sha in T2V_FILES:
            path = target / rel
            try:
                if not path.is_file() or path.stat().st_size != size:
                    return False
            except OSError:
                return False
        return True

    @classmethod
    def verify_model(cls, deep: bool = True) -> Dict[str, Any]:
        """校验本地 12 个文件；``deep=True`` 时逐个计算 SHA-256（约 1–2 分钟）。"""
        target = model_dir()
        missing: List[str] = []
        size_mismatch: List[Dict[str, Any]] = []
        sha_mismatch: List[str] = []
        for rel, size, sha in T2V_FILES:
            path = target / rel
            if not path.is_file():
                missing.append(rel)
                continue
            actual_size = path.stat().st_size
            if actual_size != size:
                size_mismatch.append({"path": rel, "expected": size, "actual": actual_size})
                continue
            if deep:
                actual_sha = sha256_file(path)
                if actual_sha.lower() != sha.lower():
                    sha_mismatch.append(rel)
        ok = not missing and not size_mismatch and not sha_mismatch
        return {
            "ok": ok,
            "target_dir": str(target),
            "deep": deep,
            "missing": missing,
            "size_mismatch": size_mismatch,
            "sha_mismatch": sha_mismatch,
            "total_bytes": T2V_TOTAL_BYTES,
            "checked_files": len(T2V_FILES),
        }

    @classmethod
    def get_manifest(cls) -> Optional[Dict[str, Any]]:
        manifest = model_dir() / MANIFEST_NAME
        if not manifest.is_file():
            return None
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        with cls._lock:
            complete = cls.is_model_completely_downloaded()
            if complete and cls._state["status"] not in ("downloading",):
                cls._state["status"] = "completed"
                cls._state["downloaded_bytes"] = T2V_TOTAL_BYTES
                cls._state["percent"] = 100.0
                cls._state["speed_bps"] = 0.0
                cls._state["speed_mbps"] = 0.0
                cls._state["eta_seconds"] = 0
                cls._state["file_index"] = len(T2V_FILES)
                cls._state["current_file"] = ""
                cls._state["current_files"] = []
                cls._state["message"] = f"✅ 2023 ModelScope 1.7B 权重已完整保存在本地: {model_dir()}"
            snapshot = copy.deepcopy(cls._state)
        snapshot["target_dir"] = str(model_dir())
        snapshot["model_id"] = MODEL_ID
        snapshot["total_files"] = len(T2V_FILES)
        snapshot["total_bytes"] = T2V_TOTAL_BYTES
        snapshot["completed"] = complete
        snapshot["files"] = [
            snapshot["files"].get(rel, {"path": rel, "size": size, "downloaded": 0, "status": "pending"})
            for rel, size, _sha in T2V_FILES
        ]
        snapshot["percent"] = round(float(snapshot.get("percent") or 0.0), 1)
        snapshot["eta_text"] = human_seconds(snapshot.get("eta_seconds"))
        snapshot["downloaded_text"] = human_bytes(snapshot.get("downloaded_bytes"))
        snapshot["total_text"] = human_bytes(T2V_TOTAL_BYTES)
        return snapshot

    @classmethod
    def _log(cls, message: str) -> None:
        text = f"[EAIW][T2V] {message}"
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"), flush=True)
        with cls._lock:
            cls._state["message"] = message
            logs = cls._state.setdefault("logs", [])
            logs.append(message)
            if len(logs) > 40:
                del logs[:-40]

    # ------------------------------------------------------------- 进度刷新
    @classmethod
    def _refresh_progress_locked(cls, now: Optional[float] = None) -> None:
        files = cls._state.get("files") or {}
        done = sum(int(item.get("downloaded") or 0) for item in files.values())
        now = now or time.time()
        elapsed = max(now - cls._last_time, 1e-6)
        delta = done - cls._last_bytes
        if elapsed >= 0.5 and delta >= 0:
            instant = delta / elapsed
            previous = float(cls._state.get("speed_bps") or 0.0)
            cls._state["speed_bps"] = instant if previous <= 0 else 0.65 * previous + 0.35 * instant
            cls._last_bytes = done
            cls._last_time = now
        speed = float(cls._state.get("speed_bps") or 0.0)
        remaining = max(T2V_TOTAL_BYTES - done, 0)
        cls._state["downloaded_bytes"] = done
        cls._state["percent"] = round(done * 100.0 / T2V_TOTAL_BYTES, 2) if T2V_TOTAL_BYTES else 100.0
        cls._state["speed_mbps"] = round(speed / (1024 * 1024), 2)
        cls._state["eta_seconds"] = int(remaining / speed) if speed > 16 * 1024 else None

    @classmethod
    def _update_file_progress(cls, rel: str, downloaded: int, speed_bps: float) -> None:
        if cls._cancel.is_set():
            raise DownloadCancelled("用户已取消下载")
        with cls._lock:
            item = cls._state["files"].get(rel)
            if item is not None:
                item["downloaded"] = downloaded
                item["speed_bps"] = round(speed_bps, 1)
            cls._state["current_file"] = rel
            cls._refresh_progress_locked()

    # ------------------------------------------------------------- 依赖安装
    @classmethod
    def ensure_pip_dependencies(cls) -> Tuple[bool, str]:
        """可选：安装推理依赖（下载本身不需要它们）。返回 (是否成功, 说明)。"""
        required = ["diffusers", "transformers", "accelerate", "safetensors"]
        missing = []
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        if not missing:
            return True, "推理依赖已安装"
        message = f"正在安装推理依赖: {' '.join(missing)}（使用清华 PyPI 镜像）..."
        cls._log(message)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "-i",
                    "https://pypi.tuna.tsinghua.edu.cn/simple",
                    "--trusted-host",
                    "pypi.tuna.tsinghua.edu.cn",
                    *missing,
                ],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            cls._log(f"依赖安装失败（可稍后手动 pip install {' '.join(missing)}）: {exc}")
            return False, str(exc)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
            cls._log(f"依赖安装失败（返回码 {proc.returncode}）。可手动执行: pip install {' '.join(missing)}")
            if tail:
                cls._log(tail)
            return False, tail
        cls._log("推理依赖安装完成")
        return True, "安装完成"

    # --------------------------------------------------------------- 下载实现
    @classmethod
    def _file_urls(cls, rel_path: str) -> List[str]:
        # 路径只含 ASCII 安全字符；仍做一次防御性检查，避免清单被篡改后写出目录。
        if ".." in Path(rel_path).parts or Path(rel_path).is_absolute():
            raise DownloadError(f"非法模型文件路径: {rel_path}")
        return [f"{base}/{rel_path}" for _label, base in MIRRORS]

    @classmethod
    def _download_one(cls, rel_path: str, size: int, sha256: str) -> bool:
        target = model_dir()
        dest = target / rel_path
        try:
            dest.resolve().relative_to(target.resolve())
        except (OSError, ValueError):
            raise DownloadError(f"模型文件路径越界: {rel_path}")

        with cls._lock:
            item = cls._state["files"].setdefault(
                rel_path,
                {"path": rel_path, "size": size, "sha256": sha256, "downloaded": 0, "status": "pending"},
            )
            item["status"] = "downloading"
            if rel_path not in cls._state["current_files"]:
                cls._state["current_files"].append(rel_path)
            cls._state["current_file"] = rel_path

        def on_progress(downloaded: int, total: Optional[int], speed: float) -> None:
            cls._update_file_progress(rel_path, downloaded, speed)

        try:
            download_file(
                cls._file_urls(rel_path),
                dest,
                expected_size=size,
                expected_sha256=sha256,
                progress=on_progress,
                label=rel_path,
                should_stop=cls._cancel.is_set,
                timeout=60.0,
                attempts_per_url=3,
            )
        except DownloadCancelled:
            with cls._lock:
                item["status"] = "pending"
                if rel_path in cls._state["current_files"]:
                    cls._state["current_files"].remove(rel_path)
            raise
        except Exception as exc:  # noqa: BLE001 - 单文件失败不应中断其它文件
            with cls._lock:
                item["status"] = "failed"
                item["error"] = str(exc)
                if rel_path in cls._state["current_files"]:
                    cls._state["current_files"].remove(rel_path)
            cls._log(f"❌ {rel_path} 下载失败: {exc}")
            return False

        with cls._lock:
            item["status"] = "completed"
            item["downloaded"] = size
            item["speed_bps"] = 0.0
            if rel_path in cls._state["current_files"]:
                cls._state["current_files"].remove(rel_path)
            cls._state["file_index"] = sum(
                1 for f in cls._state["files"].values() if f.get("status") == "completed"
            )
            cls._refresh_progress_locked()
        cls._log(f"✅ {rel_path} 完成（{human_bytes(size)}）")
        return True

    @classmethod
    def _write_manifest(cls) -> None:
        target = model_dir()
        manifest = {
            "model_id": MODEL_ID,
            "modelscope_model_id": MODELSCOPE_MODEL_ID,
            "format": "diffusers",
            "variant": "fp16",
            "completed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "total_bytes": T2V_TOTAL_BYTES,
            "files": {rel: {"size": size, "sha256": sha} for rel, size, sha in T2V_FILES},
        }
        tmp = target / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target / MANIFEST_NAME)

    @classmethod
    def _run_download(cls, jobs: int = 3, install_deps: bool = False, verify_existing: bool = False) -> bool:
        jobs = max(1, min(6, int(jobs)))
        target = model_dir()
        try:
            cls._log(f"🚀 目标: {MODEL_ID}（{human_bytes(T2V_TOTAL_BYTES)}，12 个文件）")
            cls._log(f"📁 保存到: {target}")
            cls._log("🔗 镜像顺序: " + " → ".join(label for label, _ in MIRRORS))

            if install_deps:
                cls.ensure_pip_dependencies()

            target.mkdir(parents=True, exist_ok=True)

            pending: List[Tuple[str, int, str]] = []
            corrupt: List[str] = []
            with cls._lock:
                for rel, size, sha in T2V_FILES:
                    path = target / rel
                    item = cls._state["files"][rel]
                    if path.is_file() and path.stat().st_size == size:
                        if verify_existing:
                            actual = sha256_file(path)
                            if actual.lower() != sha.lower():
                                # 不能在持锁时调用 _log（同一把非可重入锁会死锁）。
                                corrupt.append(rel)
                                path.unlink()
                                pending.append((rel, size, sha))
                                continue
                        item["status"] = "completed"
                        item["downloaded"] = size
                        continue
                    part = path.with_name(path.name + ".part")
                    if part.is_file():
                        item["downloaded"] = min(part.stat().st_size, size)
                    pending.append((rel, size, sha))
                cls._last_bytes = sum(item["downloaded"] for item in cls._state["files"].values())
                cls._last_time = time.time()
                cls._state["file_index"] = len(T2V_FILES) - len(pending)
                cls._refresh_progress_locked()

            for rel in corrupt:
                cls._log(f"⚠️ 本地 {rel} 校验失败，需要重新下载")

            if not pending:
                cls._write_manifest()
                with cls._lock:
                    cls._state["status"] = "completed"
                    cls._state["percent"] = 100.0
                    cls._state["downloaded_bytes"] = T2V_TOTAL_BYTES
                    cls._state["eta_seconds"] = 0
                    cls._state["message"] = f"✅ 12 个文件此前已全部下载完成: {target}"
                cls._log(f"🎉 全部 12 个模型文件已存在且大小正确: {target}")
                return True

            small = [item for item in pending if item[1] < PARALLEL_FILE_THRESHOLD]
            large = [item for item in pending if item[1] >= PARALLEL_FILE_THRESHOLD]

            for rel, size, sha in small:
                if cls._cancel.is_set():
                    raise DownloadCancelled("用户已取消下载")
                if not cls._download_one(rel, size, sha):
                    raise DownloadError(f"{rel} 下载失败")

            failures: List[str] = []
            if large:
                workers = min(jobs, len(large))
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eaiw-t2v") as pool:
                    futures = {pool.submit(cls._download_one, rel, size, sha): rel for rel, size, sha in large}
                    for future in as_completed(futures):
                        rel = futures[future]
                        try:
                            ok = future.result()
                        except DownloadCancelled:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            ok = False
                            cls._log(f"❌ {rel} 异常: {exc}")
                        if not ok:
                            failures.append(rel)

            if cls._cancel.is_set():
                raise DownloadCancelled("用户已取消下载")
            if failures:
                raise DownloadError("以下文件下载失败: " + ", ".join(sorted(failures)))

            # 最终以磁盘上的真实大小为准再核对一次。
            bad = [
                rel
                for rel, size, _sha in T2V_FILES
                if not (target / rel).is_file() or (target / rel).stat().st_size != size
            ]
            if bad:
                raise DownloadError("下载后大小校验失败: " + ", ".join(bad))

            cls._write_manifest()
            with cls._lock:
                cls._state["status"] = "completed"
                cls._state["percent"] = 100.0
                cls._state["downloaded_bytes"] = T2V_TOTAL_BYTES
                cls._state["file_index"] = len(T2V_FILES)
                cls._state["speed_bps"] = 0.0
                cls._state["speed_mbps"] = 0.0
                cls._state["eta_seconds"] = 0
                cls._state["current_file"] = ""
                cls._state["current_files"] = []
                cls._state["message"] = f"✅ 2023 ModelScope 1.7B 全部 12 个模型文件已下载至本地: {target}"
            cls._log(f"🎉 全部 12 个模型文件下载完成: {target}")
            return True
        except DownloadCancelled:
            with cls._lock:
                cls._state["status"] = "cancelled"
                cls._state["current_files"] = []
                cls._state["message"] = "⚠️ 下载已取消，已下载分片会保留，下次启动可断点续传"
            cls._log("下载已取消，分片已保留")
            return False
        except Exception as exc:  # noqa: BLE001
            with cls._lock:
                cls._state["status"] = "error"
                cls._state["error"] = str(exc)
                cls._state["current_files"] = []
                cls._state["message"] = f"❌ 下载失败: {exc}"
            cls._log(f"下载失败: {exc}")
            return False

    # --------------------------------------------------------------- 公共入口
    @classmethod
    def download_all_files_sync(
        cls,
        install_deps: bool = False,
        jobs: int = 3,
        verify_existing: bool = False,
    ) -> bool:
        """阻塞下载全部 12 个文件；返回是否成功。"""
        with cls._lock:
            if cls._state["status"] == "downloading":
                busy = True
            else:
                busy = False
                cls._reset_state_locked()
        if busy:
            cls._log("已有下载任务在运行，忽略本次请求")
            return False
        cls._cancel.clear()
        return cls._run_download(jobs=jobs, install_deps=install_deps, verify_existing=verify_existing)

    @classmethod
    def start_background_download(cls, install_deps: bool = False, jobs: int = 3) -> Dict[str, Any]:
        """启动后台下载线程并立即返回状态（供 HTTP 接口使用）。"""
        with cls._lock:
            if cls._state["status"] != "downloading":
                cls._reset_state_locked()
                cls._state["message"] = "正在准备下载 2023 ModelScope 1.7B 文生视频权重..."
                cls._cancel.clear()
                thread = threading.Thread(
                    target=cls._run_download,
                    kwargs={"jobs": jobs, "install_deps": install_deps},
                    name="eaiw-t2v-download",
                    daemon=True,
                )
                cls._thread = thread
                thread.start()
        return cls.get_status()

    @classmethod
    def cancel_download(cls) -> Dict[str, Any]:
        cls._cancel.set()
        cls._log("收到取消请求，正在停止下载...")
        return cls.get_status()


__all__ = [
    "MODEL_ID",
    "MODELSCOPE_MODEL_ID",
    "MODEL_LOCAL_DIR",
    "T2V_FILES",
    "T2V_MS_1_7B_FILES",
    "T2V_TOTAL_BYTES",
    "MIRRORS",
    "ModelDownloadManager",
    "model_dir",
]

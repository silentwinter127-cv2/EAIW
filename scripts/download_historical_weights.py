#!/usr/bin/env python3
"""EAIW 1.0 — 历史模型权重下载与校验工具（2014–2023）。

最常用命令::

    python scripts/download_historical_weights.py --model modelscope_t2v_2023

它会把 2023 ModelScope 文生视频模型 ``ali-vilab/text-to-video-ms-1.7b`` 的
12 个 Diffusers fp16 文件（约 3.6 GB）下载到
``models/weights/text-to-video-ms-1.7b/``，支持断点续传、多镜像回退和
SHA-256 校验。下载完成后 ``core/real_models_backend.py`` 会直接离线加载它。

其它命令::

    --list                         查看全部模型及本地状态
    --model KEY                    下载单个模型（KEY 见 --list）
    --all                          下载全部可自动获取的权重（含 1.7B 文生视频）
    --verify [KEY]                 校验本地文件（默认校验文生视频模型）
    --jobs N                       大文件并行下载数（默认 3，最大 6）
    --install-deps                 顺带安装 diffusers/transformers/accelerate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认 cp936，统一改成 UTF-8，避免中文/emoji 触发 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from models.download_utils import (  # noqa: E402
    download_file,
    human_bytes,
    human_seconds,
    sha256_file,
)
from models.model_downloader import (  # noqa: E402
    MODEL_ID,
    MODEL_LOCAL_DIR,
    T2V_FILES,
    T2V_TOTAL_BYTES,
    ModelDownloadManager,
    model_dir,
)

WEIGHTS_DIR = ROOT / "models" / "weights"

# 单文件历史权重清单。sha256 优先使用完整值；只有前缀时按前缀校验
# （torchvision 的文件名后缀就是官方 SHA-256 前 8 位）。
OFFICIAL_CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "deepdream_googlenet_2015": {
        "filename": "googlenet-1378be20.pth",
        "urls": ["https://download.pytorch.org/models/googlenet-1378be20.pth"],
        "size": 52147035,
        "sha256": "1378be20a8e875cf1568b8a71654e704449655e34711a959a38b04fb34905cef",
        "sha256_prefix": "1378be20",
        "description": "BVLC GoogLeNet (Inception v1) ImageNet-1K 官方权重 — 2015 DeepDream",
    },
    "gatys_nst_vgg19_2015": {
        "filename": "vgg19-dcbb9e9d.pth",
        "urls": ["https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"],
        "size": 574673361,
        "sha256": "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44",
        "sha256_prefix": "dcbb9e9d",
        "description": "Gatys et al. VGG-19 Gram 矩阵神经风格迁移官方权重 — 2015",
    },
    "lama_ffc_2021": {
        "filename": "lama_fp32.onnx",
        "urls": [
            "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
            "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        ],
        "size": 208044816,
        "sha256": "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
        "sha256_prefix": "",
        "description": "Suvorov et al. Big-LaMa 快速傅里叶卷积图像修复 ONNX — 2021",
    },
    "pix2pix_cyclegan_2017": {
        "filename": "summer2winter_yosemite.pth",
        "urls": [
            "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/summer2winter_yosemite.pth",
        ],
        "size": 45575747,
        "sha256": "600d267609742d22dee8a518f40eec9f4008400ba6abe1f3a8b957b23d4925d5",
        "sha256_prefix": "",
        "description": "UC Berkeley CycleGAN summer2winter_yosemite 官方权重 — 2017",
    },
    "pix2pix_cyclegan_horse2zebra_2017": {
        "filename": "horse2zebra.pth",
        "urls": [
            "https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/horse2zebra.pth",
        ],
        "size": 45575747,
        "sha256": "5bfc34b154035be1cd592cd273fdf328810bee4756044c9cf7deb8d4b6d2fadb",
        "sha256_prefix": "",
        "description": "UC Berkeley CycleGAN horse2zebra 官方权重 — 2017",
    },
    "esrgan_realesrgan_x4plus_2021": {
        "filename": "RealESRGAN_x4plus.pth",
        "urls": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        ],
        "size": 67040989,
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "sha256_prefix": "",
        "description": "Wang et al. Real-ESRGAN RRDBNet x4 官方权重 — 2021",
    },
    "vqgan_clip_2021": {
        "filename": "vqgan_imagenet_f16_16384.ckpt",
        "urls": [
            "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
        ],
        "size": 980092370,
        "sha256": "845a68805098cb666420d5db93df53f3a3b6dd443e6dd85c05759c5b998cd663",
        "sha256_prefix": "",
        "description": "Taming-Transformers VQGAN ImageNet f16-16384 官方 last.ckpt — 2021",
    },
    "sd_v1_3_2022": {
        "filename": "sd-v1-3.ckpt",
        "urls": [
            "https://huggingface.co/CompVis/stable-diffusion-v1-3-original/resolve/main/sd-v1-3.ckpt",
        ],
        "size": None,
        "sha256": "",
        "sha256_prefix": "54632c6e",
        "gated": True,
        "description": "CompVis Stable Diffusion v1.3 官方 ckpt — 2022（HF 需登录授权）",
    },
    "modelscope_t2v_2023": {
        "filename": "text-to-video-ms-1.7b/（12 个 Diffusers 文件）",
        "urls": [f"https://hf-mirror.com/{MODEL_ID}/resolve/main"],
        "size": T2V_TOTAL_BYTES,
        "sha256": "",
        "sha256_prefix": "",
        "multi": True,
        "description": "Alibaba ModelScope 1.7B 3D-UNet 文生视频 — 2023（Diffusers fp16，12 文件）",
    },
}

BUILTIN_CHECKPOINTS = {
    "patchmatch_telea_2009": "builtin://patchmatch_telea_pyramid（纯 Python/OpenCV 实现，无需下载）",
}


# --------------------------------------------------------------------- 工具函数
def _format_size(size: Optional[int]) -> str:
    return human_bytes(size) if size else "未知"


def _file_is_good(spec: Dict[str, Any]) -> bool:
    path = WEIGHTS_DIR / spec["filename"]
    if not path.is_file():
        return False
    if spec.get("size") and path.stat().st_size != spec["size"]:
        return False
    return True


def _verify_single(spec: Dict[str, Any]) -> Dict[str, Any]:
    path = WEIGHTS_DIR / spec["filename"]
    result = {"path": str(path), "ok": False, "problems": []}
    if not path.is_file():
        result["problems"].append("文件不存在")
        return result
    actual_size = path.stat().st_size
    if spec.get("size") and actual_size != spec["size"]:
        result["problems"].append(f"大小不符（期望 {spec['size']}，实际 {actual_size}）")
        return result
    expected_sha = (spec.get("sha256") or "").lower()
    prefix = (spec.get("sha256_prefix") or "").lower()
    if expected_sha or prefix:
        actual_sha = sha256_file(path)
        result["sha256"] = actual_sha
        if expected_sha and actual_sha != expected_sha:
            result["problems"].append("SHA-256 不符")
        elif prefix and not actual_sha.startswith(prefix):
            result["problems"].append(f"SHA-256 前缀不符（期望 {prefix}…）")
    result["ok"] = not result["problems"]
    return result


def _print_list(as_json: bool = False) -> None:
    if as_json:
        payload = {}
        for key, spec in OFFICIAL_CHECKPOINTS.items():
            if spec.get("multi"):
                payload[key] = {
                    "kind": "multi",
                    "target_dir": str(model_dir()),
                    "files": [{"path": rel, "bytes": size, "sha256": sha} for rel, size, sha in T2V_FILES],
                    "total_bytes": T2V_TOTAL_BYTES,
                    "downloaded": ModelDownloadManager.is_model_completely_downloaded(),
                }
            else:
                payload[key] = {
                    "kind": "single",
                    "filename": spec["filename"],
                    "bytes": spec.get("size"),
                    "urls": spec.get("urls", []),
                    "downloaded": _file_is_good(spec),
                    "gated": bool(spec.get("gated")),
                    "manual": bool(spec.get("manual")),
                }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("=" * 86)
    print("EAIW 历史模型权重清单（下载目录: models/weights/）")
    print("=" * 86)
    print(f"{'KEY':42s} {'大小':>10s}  状态  说明")
    print("-" * 86)
    for key, spec in OFFICIAL_CHECKPOINTS.items():
        if spec.get("multi"):
            state = "✅ 已完整" if ModelDownloadManager.is_model_completely_downloaded() else "— 未下载"
            size_text = human_bytes(T2V_TOTAL_BYTES)
        else:
            state = "✅ 已存在" if _file_is_good(spec) else "— 未下载"
            if spec.get("manual") and not _file_is_good(spec):
                state = "— 需手动"
            size_text = _format_size(spec.get("size"))
        print(f"{key:42s} {size_text:>10s}  {state:6s} {spec['description']}")
    for key, text in BUILTIN_CHECKPOINTS.items():
        print(f"{key:42s} {'内置':>10s}  ✅ 可用  {text}")
    print("-" * 86)
    print("下载文生视频模型: python scripts/download_historical_weights.py --model modelscope_t2v_2023")
    print("校验文生视频模型: python scripts/download_historical_weights.py --verify")
    print(f"文生视频目标目录: {model_dir()}")


def _progress_printer(filename: str):
    state = {"last": 0.0, "start": time.time()}

    def callback(done: int, total: Optional[int], speed: float) -> None:
        now = time.time()
        if now - state["last"] < 2.0 and (total is None or done < total):
            return
        state["last"] = now
        if total:
            pct = done * 100.0 / total
            eta = (total - done) / speed if speed > 1024 else None
            print(
                f"  {filename}: {pct:5.1f}%  {human_bytes(done)} / {human_bytes(total)}"
                f"  @ {human_bytes(speed)}/s  剩余 {human_seconds(eta)}",
                flush=True,
            )
        else:
            print(f"  {filename}: {human_bytes(done)}  @ {human_bytes(speed)}/s", flush=True)

    return callback


# --------------------------------------------------------------------- 下载实现
def _download_single(model_key: str, spec: Dict[str, Any]) -> str:
    if spec.get("manual"):
        raise RuntimeError(f"{model_key} 需要手动下载并放到 {WEIGHTS_DIR / spec['filename']}")
    urls = spec.get("urls") or []
    if not urls:
        raise RuntimeError(f"{model_key} 没有可用的下载地址")

    dest = WEIGHTS_DIR / spec["filename"]
    if _file_is_good(spec):
        print(f"✅ 已存在且大小正确，跳过: {dest}（{_format_size(spec.get('size'))}）")
        return str(dest)

    print(f"⬇️  下载 {model_key}")
    print(f"   文件: {spec['filename']}  大小: {_format_size(spec.get('size'))}")
    for url in urls:
        print(f"   镜像: {url}")
    download_file(
        urls,
        dest,
        expected_size=spec.get("size"),
        expected_sha256=spec.get("sha256") or None,
        progress=_progress_printer(spec["filename"]),
        label=spec["filename"],
    )

    prefix = (spec.get("sha256_prefix") or "").lower()
    if prefix:
        actual = sha256_file(dest)
        if not actual.startswith(prefix):
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"{spec['filename']} SHA-256 前缀校验失败（期望 {prefix}…，实际 {actual[:8]}…），已删除"
            )
        print(f"✅ SHA-256 前缀校验通过: {actual[:16]}…")
    print(f"🎉 已保存: {dest}（{human_bytes(dest.stat().st_size)}）")
    return str(dest)


def download_checkpoint(model_key: str, install_deps: bool = False, jobs: int = 3) -> str:
    """下载单个模型并返回本地路径（供 CLI 与 HTTP 接口调用）。"""
    key = (model_key or "").strip()
    if key in BUILTIN_CHECKPOINTS:
        return BUILTIN_CHECKPOINTS[key].split("（")[0]
    spec = OFFICIAL_CHECKPOINTS.get(key)
    if spec is None:
        raise KeyError(f"未知模型: {key}；可用: {', '.join(OFFICIAL_CHECKPOINTS)}")

    if spec.get("multi"):
        print(f"⬇️  开始下载 2023 ModelScope 1.7B 文生视频模型（{human_bytes(T2V_TOTAL_BYTES)}）")
        print(f"   目标目录: {model_dir()}")
        ok = ModelDownloadManager.download_all_files_sync(install_deps=install_deps, jobs=jobs)
        if not ok:
            status = ModelDownloadManager.get_status()
            raise RuntimeError(status.get("message") or "下载失败")
        verify = ModelDownloadManager.verify_model(deep=False)
        if not verify["ok"]:
            raise RuntimeError(f"下载后校验失败: {verify}")
        return str(model_dir())

    return _download_single(key, spec)


def _verify(model_key: str) -> bool:
    key = (model_key or "modelscope_t2v_2023").strip()
    if key in BUILTIN_CHECKPOINTS:
        print(f"✅ {key}: {BUILTIN_CHECKPOINTS[key]}")
        return True
    spec = OFFICIAL_CHECKPOINTS.get(key)
    if spec is None:
        print(f"❌ 未知模型: {key}")
        return False
    if spec.get("multi"):
        print(f"🔍 校验 2023 ModelScope 1.7B（{len(T2V_FILES)} 个文件，计算 SHA-256 约需 1–2 分钟）...")
        result = ModelDownloadManager.verify_model(deep=True)
        if result["ok"]:
            print(f"✅ 全部 {len(T2V_FILES)} 个文件大小与 SHA-256 均正确: {result['target_dir']}")
            return True
        for rel in result["missing"]:
            print(f"❌ 缺少: {rel}")
        for item in result["size_mismatch"]:
            print(f"❌ 大小不符: {item['path']}（期望 {item['expected']}，实际 {item['actual']}）")
        for rel in result["sha_mismatch"]:
            print(f"❌ SHA-256 不符: {rel}")
        return False

    print(f"🔍 校验 {key}: {WEIGHTS_DIR / spec['filename']}")
    result = _verify_single(spec)
    if result["ok"]:
        print(f"✅ 校验通过（{human_bytes((WEIGHTS_DIR / spec['filename']).stat().st_size)}）")
        return True
    for problem in result["problems"]:
        print(f"❌ {problem}")
    return False


def _download_all(install_deps: bool, jobs: int) -> int:
    failures: List[str] = []
    # 先下文生视频模型（3.6 GB，用户主要目标）。
    try:
        download_checkpoint("modelscope_t2v_2023", install_deps=install_deps, jobs=jobs)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"modelscope_t2v_2023: {exc}")

    for key, spec in OFFICIAL_CHECKPOINTS.items():
        if spec.get("multi"):
            continue
        if spec.get("manual"):
            print(f"⏭️  跳过 {key}（需手动放置: {WEIGHTS_DIR / spec['filename']}）")
            continue
        if spec.get("gated"):
            print(f"⏭️  跳过 {key}（需要 HuggingFace 登录授权，可手动下载）")
            continue
        try:
            download_checkpoint(key, jobs=jobs)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{key}: {exc}")

    if failures:
        print("\n❌ 以下模型下载失败:")
        for item in failures:
            print(f"   - {item}")
        return 1
    print("\n🎉 全部可自动下载的历史权重已完成")
    return 0


def checkpoint_status() -> List[Dict[str, Any]]:
    """返回全部权重的本地状态，供 /api/download_status 使用。"""
    items: List[Dict[str, Any]] = []
    t2v_complete = ModelDownloadManager.is_model_completely_downloaded()
    for key, spec in OFFICIAL_CHECKPOINTS.items():
        entry: Dict[str, Any] = {
            "key": key,
            "kind": "multi" if spec.get("multi") else "single",
            "filename": spec.get("filename", ""),
            "size": spec.get("size"),
            "description": spec.get("description", ""),
            "manual": bool(spec.get("manual")),
            "gated": bool(spec.get("gated")),
        }
        if spec.get("multi"):
            entry["downloaded"] = t2v_complete
            entry["target"] = str(model_dir())
        else:
            dest = WEIGHTS_DIR / spec["filename"]
            entry["downloaded"] = _file_is_good(spec)
            entry["path"] = str(dest)
        items.append(entry)
    return items


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="EAIW 历史模型权重下载器（支持断点续传 / 多镜像 / SHA-256 校验）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/download_historical_weights.py --model modelscope_t2v_2023\n"
            "  python scripts/download_historical_weights.py --verify\n"
            "  python scripts/download_historical_weights.py --all --jobs 4\n"
        ),
    )
    parser.add_argument("--list", action="store_true", help="列出模型及本地状态，不下载")
    parser.add_argument("--model", type=str, help="下载指定模型（KEY 见 --list）")
    parser.add_argument("--all", action="store_true", help="下载全部可自动获取的权重")
    parser.add_argument("--t2v", action="store_true", help="等价于 --model modelscope_t2v_2023")
    parser.add_argument("--verify", nargs="?", const="modelscope_t2v_2023", help="校验本地权重（默认文生视频模型）")
    parser.add_argument("--jobs", type=int, default=3, help="大文件并行下载数（默认 3，最大 6）")
    parser.add_argument("--install-deps", action="store_true", help="下载后安装 diffusers/transformers/accelerate")
    parser.add_argument("--json", action="store_true", help="--list 输出 JSON")
    args = parser.parse_args(argv)

    if args.verify is not None:
        return 0 if _verify(args.verify) else 1

    if args.list:
        _print_list(as_json=args.json)
        return 0

    if args.model:
        try:
            download_checkpoint(args.model, install_deps=args.install_deps, jobs=args.jobs)
            return 0
        except KeyError as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 下载失败: {exc}", file=sys.stderr)
            return 1

    if args.all:
        return _download_all(install_deps=args.install_deps, jobs=args.jobs)

    if args.t2v:
        try:
            download_checkpoint("modelscope_t2v_2023", install_deps=args.install_deps, jobs=args.jobs)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 下载失败: {exc}", file=sys.stderr)
            return 1

    _print_list(as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

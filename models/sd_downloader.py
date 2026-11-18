"""Resumable downloader for the SD-1.4 diffusers weights used by lab 05.

The original CompVis SD-1.3 repository is gated on Hugging Face (and has no
public ModelScope mirror), so lab 05 uses the closest public checkpoint,
CompVis SD-1.4, with the same 512x512 / CLIP text encoder / 8-channel VAE
architecture.  All files are SHA-256 checked before being moved into place.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

from models.download_utils import (
    DownloadError,
    download_file,
    human_bytes,
    sha256_file,
)

ROOT = Path(__file__).resolve().parent.parent
SD_LOCAL_DIR = ROOT / "models" / "weights" / "stable-diffusion-v1-4"

SD_FILES: List[Tuple[str, int, str]] = [
    ("model_index.json", 543, "03039784faa9ac77f813833046820f4800cb205988edadd4d3e0477e307d532d"),
    ("scheduler/scheduler_config.json", 313, "8335df29b53da12e9aeca35b1c2e60d88b23eae4cc5ad3d07efdcd7f003cc343"),
    ("text_encoder/config.json", 592, "aa210e6d6abd239a466af57a1dadb62d5ced63295064d295a549e08a6c4a2020"),
    ("text_encoder/model.safetensors", 492265879, "7b3a12df205cb3c74dd4eae4354d93f606ae6b3bc29d5d06fd97921cb9ad8a81"),
    ("tokenizer/merges.txt", 524619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
    ("tokenizer/special_tokens_map.json", 472, "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac"),
    ("tokenizer/tokenizer_config.json", 806, "00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882"),
    ("tokenizer/vocab.json", 1059962, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
    ("unet/config.json", 743, "d6749baad843798f21da6c2d3dd9d642e5ebb73455ade4763158f5c6d3e95b03"),
    ("unet/diffusion_pytorch_model.safetensors", 3438167534, "145a07e0f05ec5bbe6e2e9faf608bdb311caf708895cac8c8ed713c59864e1e8"),
    ("vae/config.json", 551, "0e5966d4d52c8077648754ffd2c14ab5e0b26a4bdfa6af1d207c27550fa6d2af"),
    ("vae/diffusion_pytorch_model.safetensors", 334643276, "a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c"),
]

SD_TOTAL_BYTES = sum(item[1] for item in SD_FILES)
_MIRRORS = (
    "https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-4/resolve/master",
    "https://hf-mirror.com/CompVis/stable-diffusion-v1-4/resolve/main",
)


def _logged(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def is_complete(target: Path = SD_LOCAL_DIR) -> bool:
    return all((target / rel).is_file() and (target / rel).stat().st_size == size for rel, size, _ in SD_FILES)


def ensure_sd_weights(target: Path = SD_LOCAL_DIR, jobs: int = 2) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    if is_complete(target):
        return target
    pending = []
    for rel, size, sha in SD_FILES:
        dest = target / rel
        if dest.is_file() and dest.stat().st_size == size:
            continue
        pending.append((rel, size, sha))
    _logged(f"[SD] target: {target}")
    _logged(f"[SD] pending {len(pending)} file(s), {human_bytes(sum(x[1] for x in pending))}")
    errors: List[str] = []

    def work(item: Tuple[str, int, str]) -> None:
        rel, size, sha = item
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        urls = [f"{base}/{rel}" for base in _MIRRORS]
        _logged(f"[SD] downloading {rel} ({human_bytes(size)})")
        download_file(urls, dest, expected_size=size, expected_sha256=sha, label=rel, attempts_per_url=3)
        _logged(f"[SD] done {rel}")

    # The three large shards benefit from separate connections; small configs
    # are cheap either way.
    large = [x for x in pending if x[1] >= 16 << 20]
    small = [x for x in pending if x[1] < 16 << 20]
    for item in small:
        try:
            work(item)
        except Exception as exc:  # noqa: BLE001 - report and continue to cleanup
            errors.append(f"{item[0]}: {exc}")
    if large:
        with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(large)))) as pool:
            futures = {pool.submit(work, item): item for item in large}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{item[0]}: {exc}")
    if errors:
        raise DownloadError("; ".join(errors))
    missing = [rel for rel, size, _ in SD_FILES if not (target / rel).is_file() or (target / rel).stat().st_size != size]
    if missing:
        raise DownloadError("incomplete SD download: " + ", ".join(missing))
    return target


if __name__ == "__main__":
    try:
        ensure_sd_weights()
    except Exception as exc:
        _logged(f"[SD] FAILED: {exc}")
        sys.exit(1)
    _logged(f"[SD] ready: {SD_LOCAL_DIR}")

"""EAIW 下载工具：支持断点续传、多镜像回退、原子落盘与 SHA-256 校验。

仅供本项目下载公开的模型权重使用。所有下载先写入 ``*.part``，校验通过后
才用 :func:`os.replace` 原子替换为正式文件，因此中途断电/断网不会留下
“看起来完整、实际被截断”的权重文件。
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CHUNK_SIZE = 1 << 20  # 1 MiB
USER_AGENT = "EAIW-downloader/1.1 (Windows; +https://modelscope.cn)"

ProgressCallback = Callable[[int, Optional[int], float], None]


class DownloadError(RuntimeError):
    """下载失败（网络、HTTP 状态、大小或校验值不符）。"""


class DownloadCancelled(DownloadError):
    """用户主动取消下载。"""


def human_bytes(num: Optional[float]) -> str:
    if num is None:
        return "?"
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def human_seconds(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN 检查
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """流式计算文件 SHA-256，避免把 2.8 GB 的 UNet 读进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _log(message: str) -> None:
    text = f"[EAIW][下载] {message}"
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        # Windows 控制台可能是 cp936，无法输出 emoji；退化为 ASCII 安全文本。
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)


def _prime_hasher(hasher: "hashlib._Hash", path: Path) -> None:
    """把已存在的分片内容喂给 hasher，用于断点续传后的整体 SHA-256。"""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher


def _parse_content_range(value: str) -> Optional[tuple[int, int, Optional[int]]]:
    match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value or "")
    if not match:
        return None
    total = None if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


def _finalize(part: Path, dest: Path, expected_sha256: Optional[str], label: str) -> None:
    if expected_sha256:
        actual = sha256_file(part)
        if actual.lower() != expected_sha256.lower():
            try:
                part.unlink()
            except OSError:
                pass
            raise DownloadError(
                f"{label} SHA-256 校验失败（期望 {expected_sha256[:12]}…，实际 {actual[:12]}…），已删除损坏分片"
            )
    os.replace(part, dest)


def download_file(
    urls: Sequence[str],
    dest: Path,
    expected_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    label: Optional[str] = None,
    timeout: float = 60.0,
    attempts_per_url: int = 3,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Path:
    """下载 ``urls`` 中的任意一个到 ``dest``。

    - 若 ``dest`` 已存在且大小（及可选的 SHA-256）匹配，则直接返回；
    - 使用 ``dest + '.part'`` 断点续传，服务端返回 200 而不是 206 时自动从头重下；
    - 只有完整大小与 SHA-256 都通过后才原子替换为 ``dest``。

    成功返回 ``dest``；失败抛出 :class:`DownloadError`。
    """
    if not urls:
        raise DownloadError("未提供任何下载镜像 URL")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    label = label or dest.name

    def report(done: int, total: Optional[int], speed: float) -> None:
        if progress is not None:
            progress(done, total, speed)

    def cancelled() -> bool:
        return bool(should_stop and should_stop())

    # 1) 已存在且可信的正式文件：直接复用，避免 3.6 GB 重复下载。
    if dest.exists():
        size = dest.stat().st_size
        if expected_size is None:
            report(size, None, 0.0)
            return dest
        if size == expected_size:
            if expected_sha256:
                actual = sha256_file(dest)
                if actual.lower() == expected_sha256.lower():
                    report(size, expected_size, 0.0)
                    return dest
                _log(f"{label} 已存在但 SHA-256 不匹配，重新下载")
                dest.unlink()
            else:
                report(size, expected_size, 0.0)
                return dest
        else:
            _log(f"{label} 已存在但大小 {size} != {expected_size}，重新下载")
            dest.unlink()

    # 2) 分片恰好完整（上次已下完但没来得及改名）：直接校验落盘。
    if part.exists() and expected_size is not None and part.stat().st_size == expected_size:
        try:
            _finalize(part, dest, expected_sha256, label)
            report(expected_size, expected_size, 0.0)
            _log(f"{label} 已校验完成")
            return dest
        except DownloadError:
            _log(f"{label} 完整分片校验失败，将重新下载")

    last_error: Optional[BaseException] = None
    for url_index, url in enumerate(urls):
        if cancelled():
            raise DownloadCancelled(f"{label} 下载已取消")
        if url_index:
            _log(f"{label} 切换镜像: {url.split('/')[2]}")
        for attempt in range(1, attempts_per_url + 1):
            if cancelled():
                raise DownloadCancelled(f"{label} 下载已取消")
            offset = part.stat().st_size if part.exists() else 0
            if expected_size is not None and offset > expected_size:
                _log(f"{label} 分片大小异常（{offset} > {expected_size}），丢弃重下")
                part.unlink()
                offset = 0

            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"

            started = time.time()
            transferred = 0
            last_report = 0.0
            try:
                request = Request(url, headers=headers)
                with urlopen(request, timeout=timeout) as response:
                    status = getattr(response, "status", 200)
                    if status == 206:
                        content_range = _parse_content_range(response.headers.get("Content-Range", ""))
                        if offset == 0:
                            mode = "wb"
                        elif content_range is None or content_range[0] != offset:
                            raise DownloadError(
                                f"服务端 Range 响应异常（请求偏移 {offset}，Content-Range={content_range}）"
                            )
                        else:
                            mode = "ab"
                        if content_range and content_range[2] is not None and expected_size is not None:
                            if content_range[2] != expected_size:
                                _log(
                                    f"{label} 警告: 服务端声明总大小 {content_range[2]}，"
                                    f"清单为 {expected_size}，以清单校验为准"
                                )
                        hasher = hashlib.sha256()
                        if mode == "ab" and part.exists():
                            _prime_hasher(hasher, part)
                    elif status == 200:
                        if offset:
                            _log(f"{label} 服务端未支持 Range，改为从头下载")
                        mode = "wb"
                        offset = 0
                        hasher = hashlib.sha256()
                    else:
                        raise DownloadError(f"HTTP {status}")

                    downloaded = offset
                    with open(part, mode) as fh:
                        while True:
                            if cancelled():
                                raise DownloadCancelled(f"{label} 下载已取消")
                            chunk = response.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            fh.write(chunk)
                            hasher.update(chunk)
                            downloaded += len(chunk)
                            transferred += len(chunk)
                            if expected_size is not None and downloaded > expected_size:
                                raise DownloadError(
                                    f"{label} 收到超出预期大小的数据（{downloaded} > {expected_size}）"
                                )
                            now = time.time()
                            if progress is not None and now - last_report >= 1.0:
                                elapsed = max(now - started, 1e-6)
                                report(downloaded, expected_size, transferred / elapsed)
                                last_report = now

                elapsed = max(time.time() - started, 1e-6)
                report(downloaded, expected_size, transferred / elapsed)
                if expected_size is not None and downloaded != expected_size:
                    raise DownloadError(f"{label} 传输不完整（{downloaded}/{expected_size} 字节）")

                actual_sha = hasher.hexdigest()
                if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
                    try:
                        part.unlink()
                    except OSError:
                        pass
                    raise DownloadError(
                        f"{label} SHA-256 校验失败（期望 {expected_sha256[:12]}…，实际 {actual_sha[:12]}…）"
                    )
                os.replace(part, dest)
                _log(f"{label} 下载完成（{human_bytes(dest.stat().st_size)}，SHA-256 已校验）")
                return dest

            except DownloadCancelled:
                raise
            except HTTPError as exc:
                last_error = exc
                if exc.code == 416:
                    # 分片已经完整但最后一次写入未完成，或服务端认为越界。
                    if part.exists() and expected_size is not None and part.stat().st_size == expected_size:
                        try:
                            _finalize(part, dest, expected_sha256, label)
                            _log(f"{label} 下载完成（HTTP 416 后确认分片完整）")
                            return dest
                        except DownloadError as verify_exc:
                            last_error = verify_exc
                    _log(f"{label} HTTP 416，重置分片后重试")
                    try:
                        part.unlink()
                    except OSError:
                        pass
                elif exc.code in (401, 403, 404):
                    _log(f"{label} HTTP {exc.code}（{url.split('/')[2]}），尝试下一个镜像")
                    break
                else:
                    _log(f"{label} HTTP {exc.code}，第 {attempt}/{attempts_per_url} 次重试")
            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                _log(f"{label} 网络错误: {exc}；第 {attempt}/{attempts_per_url} 次重试")
            except DownloadError as exc:
                last_error = exc
                _log(f"{label} {exc}；第 {attempt}/{attempts_per_url} 次重试")
            except Exception as exc:  # 防御性兜底：任何解析/磁盘异常都不应让进程崩溃
                last_error = exc
                _log(f"{label} 未预期错误: {exc!r}")
                try:
                    if part.exists() and expected_size is not None and part.stat().st_size > expected_size:
                        part.unlink()
                except OSError:
                    pass

            if cancelled():
                raise DownloadCancelled(f"{label} 下载已取消")
            time.sleep(min(2.0 * attempt, 8.0))

    raise DownloadError(f"{label} 所有镜像均下载失败：{last_error}")


def download_files_parallel(
    items: Iterable[tuple[Sequence[str], Path, Optional[int], Optional[str]]],
    workers: int = 3,
) -> list[Path]:
    """简单并行下载封装（下载清单本身在 model_downloader 中维护）。"""
    from concurrent.futures import ThreadPoolExecutor

    tasks = list(items)
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(download_file, urls, dest, size, sha, None, dest.name)
            for urls, dest, size, sha in tasks
        ]
        for future in futures:
            results.append(future.result())
    return results

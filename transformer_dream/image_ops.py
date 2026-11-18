from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from .config import CHANNELS, MERGE_SIZE, PATCH_SIZE, TEMPORAL_PATCH_SIZE


def validate_size(size: int) -> int:
    size = int(size)
    if size % (PATCH_SIZE * MERGE_SIZE) != 0:
        raise ValueError("size must be a multiple of 32")
    if not (256 <= size <= 448):
        raise ValueError("size must be between 256 and 448")
    return size


def patchify(img: torch.Tensor) -> torch.Tensor:
    """Differentiable equivalent of Qwen2VLImageProcessor.patchify.

    img: [B, 3, S, S] in [0, 1]; returns [B, grid_h*grid_w, 3*2*16*16].
    """
    b, c, h, w = img.shape
    p, m, t = PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH_SIZE
    if h % (p * m) or w % (p * m):
        raise ValueError("image dimensions must be multiples of 32")
    gh, gw = h // p, w // p
    x = img * 2.0 - 1.0
    patches = x.reshape(b, c, gh // m, m, p, gw // m, m, p)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    patches = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, t, -1, -1)
    return patches.reshape(b, gh * gw, c * t * p * p)


def decode_patches(patches: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """Inverse of patchify for one image.  Returns [1, 3, H, W] in [0, 1]."""
    p, m, t = PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH_SIZE
    gh2, gw2 = grid_h // m, grid_w // m
    x = patches.reshape(gh2, gw2, m, m, CHANNELS, t, p, p)
    x = x[:, :, :, :, :, 0]
    x = x.permute(4, 0, 2, 5, 1, 3, 6).reshape(1, CHANNELS, grid_h * p, grid_w * p)
    return ((x + 1.0) * 0.5).clamp(0, 1)


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    return (
        (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
        + (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    )


def jitter(img: torch.Tensor, max_shift: int = 3) -> torch.Tensor:
    """Small differentiable translation so optimization does not lock to a grid."""
    pad = max_shift
    x = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    dy = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    dx = int(torch.randint(0, 2 * pad + 1, (1,)).item())
    h, w = img.shape[-2:]
    return x[:, :, dy : dy + h, dx : dx + w]


def augment(img: torch.Tensor, step: int = 0, scale: float = 0.12, rotate_deg: float = 7.0,
            translate: float = 0.06) -> torch.Tensor:
    """Random scale/rotation/translation/flip + mild blur/color jitter.

    Classic feature visualization needs this augmentation; without it the
    optimizer writes adversarial high-frequency patterns (letters/noise)
    instead of multi-scale structure.
    """
    b, c, h, w = img.shape
    device = img.device
    ang = (torch.rand(1, device=device).item() * 2 - 1) * rotate_deg * math.pi / 180.0
    sc = 1.0 + (torch.rand(1, device=device).item() * 2 - 1) * scale
    tx = (torch.rand(1, device=device).item() * 2 - 1) * translate
    ty = (torch.rand(1, device=device).item() * 2 - 1) * translate
    ca, sa = math.cos(ang), math.sin(ang)
    theta = torch.tensor([[[ca / sc, -sa / sc, tx], [sa / sc, ca / sc, ty]]], device=device, dtype=img.dtype)
    grid = F.affine_grid(theta, (b, c, h, w), align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if torch.rand(1, device=device).item() < 0.5:
        out = torch.flip(out, dims=[3])

    # random color decorrelation near identity, keeps images from collapsing to
    # one channel while still exploring color combinations.
    if torch.rand(1, device=device).item() < 0.7:
        m = torch.eye(3, device=device, dtype=img.dtype)
        m = m + (torch.rand(3, 3, device=device, dtype=img.dtype) - 0.5) * 0.18
        out = torch.einsum("ij,bjhw->bihw", m, out)
        out = out + (torch.rand(1, 1, 1, 1, device=device) - 0.5) * 0.08
        out = out.clamp(0, 1)

    # periodic 3x3 blur suppresses pixel-grid adversarial noise.
    if step % 3 == 0:
        k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device=device, dtype=img.dtype) / 16.0
        k = k.view(1, 1, 3, 3).expand(c, 1, 3, 3)
        out = F.conv2d(F.pad(out, (1, 1, 1, 1), mode="reflect"), k, groups=c)
    return out.clamp(0, 1)


def normalize_grad(img: torch.Tensor) -> None:
    if img.grad is None:
        return
    g = img.grad
    img.grad = g / (g.abs().mean() + 1e-8)


def pil_from_tensor(img: torch.Tensor) -> Image.Image:
    arr = (img.detach()[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def tensor_to_data_url(img: torch.Tensor, fmt: str = "PNG", quality: int = 92) -> str:
    buf = io.BytesIO()
    im = pil_from_tensor(img)
    if fmt.upper() == "JPEG":
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def save_tensor(img: torch.Tensor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_from_tensor(img).save(path)
    return str(path)

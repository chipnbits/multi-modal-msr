"""Blend-weight kernels for stitching overlapping super-resolved patches."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

BlendKind = Literal["auto", "linear", "cosine", "gaussian", "ones"]


def linear_2d(patch_px: int) -> np.ndarray:
    """Separable triangular window: 1 at the centre, 0 at the edges."""
    if patch_px < 1:
        raise ValueError(f"patch_px must be >= 1, got {patch_px}")
    centre = (patch_px - 1) / 2.0
    idx = np.arange(patch_px, dtype=np.float32)
    w_1d = 1.0 - np.abs(idx - centre) / centre if patch_px > 1 else np.ones(1, dtype=np.float32)
    w_1d = np.clip(w_1d, 0.0, 1.0).astype(np.float32)
    return np.outer(w_1d, w_1d).astype(np.float32)


def cosine_2d(patch_px: int) -> np.ndarray:
    """Separable Hann (squared-sine) window: 1 at the centre, 0 at the edges.

    Smoother than `linear_2d` at the boundary (continuous first derivative), which
    suppresses the high-frequency seam artefacts you can see at patch borders with
    a triangular ramp.
    """
    if patch_px < 1:
        raise ValueError(f"patch_px must be >= 1, got {patch_px}")
    if patch_px == 1:
        return np.ones((1, 1), dtype=np.float32)
    idx = np.arange(patch_px, dtype=np.float32)
    w_1d = np.sin(np.pi * idx / (patch_px - 1)) ** 2
    return np.outer(w_1d, w_1d).astype(np.float32)


def gaussian_2d(patch_px: int, *, sigma: float | None = None) -> np.ndarray:
    """Gaussian decay from centre — strongest reported PSNR/SSIM in 4K SR. STUB."""
    raise NotImplementedError("Gaussian blending — fill in (exponential decay from centre).")


def ones_2d(patch_px: int) -> np.ndarray:
    """Uniform weight — for non-overlapping strides where blending is a no-op."""
    return np.ones((patch_px, patch_px), dtype=np.float32)


BLENDS: dict[str, Callable[[int], np.ndarray]] = {
    "linear": linear_2d,
    "cosine": cosine_2d,
    "gaussian": gaussian_2d,
    "ones": ones_2d,
}


def make_blend_weight(patch_px: int, stride_px: int, kind: BlendKind = "auto") -> np.ndarray:
    """Return a `(patch_px, patch_px)` weight array, picking the kernel for the chosen stride."""
    if kind == "auto":
        kind = "ones" if stride_px >= patch_px else "linear"
    if kind not in BLENDS:
        raise ValueError(f"unknown blend kind {kind!r}; choose from {sorted(BLENDS)} or 'auto'")
    return BLENDS[kind](patch_px)

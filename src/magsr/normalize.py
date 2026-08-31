"""`[vmin, vmax] <-> [0, 1]` normalization shared across the repo.

Single home for the clip-scale convention used by the datasets, the reconstruction /
evaluation flows, and the dataset-build QA figures. The stats consumed here are written by
`scripts/build_ksa_dataset/04_compute_ksa_aligned_normalization.py` (KSA layout:
`{"global": {...}, "blocks": {"1": {...}, ...}}`) and by `generate_wa_patch_pairs`
(WA layout: top-level `vmin`/`vmax`).

`clip_scale01` / `center_scale01` are array-generic: they accept numpy arrays and torch
tensors alike (both expose `.clip`), with scalar or broadcastable bounds — the KSA dataset
passes batched per-block `(B, 1, 1, 1)` bound tensors. `Normalizer` is the scalar-bounds
object handed around the numpy reconstruction pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TypeVar

import numpy as np

# np.ndarray or torch.Tensor — anything exposing .clip and arithmetic operators.
ArrayT = TypeVar("ArrayT")


def clip_scale01(x: ArrayT, vmin: Any, vmax: Any) -> ArrayT:
    """Clip `x` to `[vmin, vmax]` and linearly scale to `[0, 1]`. NaN propagates.

    Works on numpy arrays and torch tensors; `vmin`/`vmax` may be scalars or arrays/tensors
    broadcastable to `x` (e.g. per-block batched bounds shaped `(B, 1, 1, 1)`).
    """
    return (x.clip(vmin, vmax) - vmin) / (vmax - vmin)


def denorm01(x: ArrayT, vmin: Any, vmax: Any) -> ArrayT:
    """Invert the `[0, 1]` scaling back to physical units: `x * (vmax - vmin) + vmin`.

    Inverse of `clip_scale01` (up to the clip). Same genericity: numpy or torch, scalar or
    broadcastable bounds.
    """
    return x * (vmax - vmin) + vmin


def center_scale01(v: ArrayT, half_range: float) -> ArrayT:
    """Map a signed, zero-centred `v` to `[0, 1]`: `0 -> 0.5`, `±half_range -> 0/1` (clipped).

    The convention for gradient/relief channels (DEM slope, mean-removed elevation).
    Works on numpy arrays and torch tensors. NaN propagates.
    """
    return (v / (2.0 * half_range) + 0.5).clip(0.0, 1.0)


def to_pm1(x: ArrayT) -> ArrayT:
    """`[0, 1] -> [-1, 1]` — the U-Net input convention. Inverse of `from_pm1`.

    The datasets emit `[0, 1]` channels; the U-Net corrector consumes `[-1, 1]`.
    Works on numpy arrays and torch tensors. NaN propagates.
    """
    return x * 2.0 - 1.0


def from_pm1(x: ArrayT) -> ArrayT:
    """`[-1, 1] -> [0, 1]` — invert `to_pm1` on the U-Net output."""
    return (x + 1.0) * 0.5


class Normalizer:
    """`[vmin, vmax]` ↔ `[0, 1]` clip+scale. Source of vmin/vmax is the caller's problem.

    `eps` is added to the denominator (`vmax - vmin + eps`) — the WA loader's historical
    degenerate-range guard, kept as an option so its trained checkpoints see byte-identical
    inputs. The KSA pipeline uses the default `eps=0.0`.
    """

    def __init__(self, vmin: float, vmax: float, *, eps: float = 0.0):
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.eps = float(eps)

    @classmethod
    def from_stats(cls, stats: Mapping[str, Any], *, eps: float = 0.0) -> "Normalizer":
        """Build from any mapping carrying `vmin`/`vmax` keys (a stats-JSON section)."""
        return cls(vmin=stats["vmin"], vmax=stats["vmax"], eps=eps)

    @classmethod
    def from_json(cls, path: Path | str, *, block: int | None = None) -> "Normalizer":
        """Load a KSA-layout `normalization.json`: the `global` section, or one block's.

        Pass `block=<id>` (1-indexed survey block) for the per-block bounds used by
        `norm_mode="blockwise"` models.
        """
        stats = json.loads(Path(path).read_text())
        section = stats["global"] if block is None else stats["blocks"][str(block)]
        return cls.from_stats(section)

    @property
    def data_range(self) -> float:
        """`vmax - vmin` — the span handed to SSIM/MS-SSIM and RMSE denormalization."""
        return self.vmax - self.vmin

    def normalize(self, arr: np.ndarray, *, nan_fill: float | None = None) -> np.ndarray:
        clipped = np.clip(arr, self.vmin, self.vmax)
        out = ((clipped - self.vmin) / (self.vmax - self.vmin + self.eps)).astype(np.float32, copy=False)
        return out if nan_fill is None else np.nan_to_num(out, nan=nan_fill)

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        return denorm01(arr, self.vmin, self.vmax).astype(np.float32, copy=False)

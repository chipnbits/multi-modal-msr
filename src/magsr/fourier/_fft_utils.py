"""FFT-domain utilities for potential-field operators.

Pure-torch helpers used by `equivalent_layer.py`. Each function is batch-safe
(leading batch dimensions are passed through unchanged) and device/dtype
follows the input tensors.

Sign / wavenumber conventions:
- `make_wavenumbers` returns `kx, ky, k` in radians per metre with the same
  layout as `torch.fft.fft2` output (i.e. `fftfreq` ordering, DC at `[0, 0]`).
- `k = sqrt(kx**2 + ky**2)`; we never need `|k|` in any other shape.

Caching: the wavenumber and Tukey-window functions cache their outputs by
``(shape, sample-spacing, alpha, device, dtype)`` so repeated calls with the
same arguments — typical inside a training loop — are O(1) lookups.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Wavenumbers (cached) — invariant given (shape, dx, dy, device, dtype).
# ---------------------------------------------------------------------------
def _make_wavenumbers_impl(h, w, dx, dy, device, dtype, *, rfft):
    freq_w = torch.fft.rfftfreq if rfft else torch.fft.fftfreq
    fx = freq_w(w, d=dx, device=device, dtype=dtype) * (2.0 * math.pi)
    fy = torch.fft.fftfreq(h, d=dy, device=device, dtype=dtype) * (2.0 * math.pi)
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    return kx, ky, torch.sqrt(kx * kx + ky * ky)


@lru_cache(maxsize=32)
def _wavenumbers_full_cached(h, w, dx, dy, device_str, dtype):
    device = torch.device(device_str)
    return _make_wavenumbers_impl(h, w, dx, dy, device, dtype, rfft=False)


@lru_cache(maxsize=32)
def _wavenumbers_rfft_cached(h, w, dx, dy, device_str, dtype):
    device = torch.device(device_str)
    return _make_wavenumbers_impl(h, w, dx, dy, device, dtype, rfft=True)


def _device_key(device) -> str:
    if device is None:
        return "cpu"
    return str(torch.device(device))


def make_wavenumbers(
    h: int, w: int, dx: float, dy: float, *, device=None, dtype=torch.float32
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (kx, ky, k) on an (h, w) grid with `fftfreq` ordering.

    `kx` varies along the last axis (width / x), `ky` along the second-to-last
    (height / y). Units: radians per metre. Cached by argument tuple.
    """
    return _wavenumbers_full_cached(h, w, dx, dy, _device_key(device), dtype)


def make_wavenumbers_rfft(
    h: int, w: int, dx: float, dy: float, *, device=None, dtype=torch.float32
) -> tuple[Tensor, Tensor, Tensor]:
    """Wavenumbers on the ``rfft2`` half-spectrum layout: ``(H, W // 2 + 1)``.

    ``rfft2`` exploits Hermitian symmetry of real-valued FFTs to return only
    half the spectrum (the ``+`` kx half). Pairing this with ``irfft2(s=…)``
    cuts FFT cost ~2x for real-valued inputs without losing any information.

    Cached by argument tuple — repeated calls for the same tile geometry
    return the same tensor.
    """
    return _wavenumbers_rfft_cached(h, w, dx, dy, _device_key(device), dtype)


# ---------------------------------------------------------------------------
# Tukey window (cached) — invariant given (shape, alpha, device, dtype).
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _tukey1d_cached(n, alpha, device_str, dtype):
    device = torch.device(device_str)
    if alpha <= 0.0:
        return torch.ones(n, device=device, dtype=dtype)
    alpha = min(alpha, 1.0)
    x = torch.arange(n, device=device, dtype=dtype)
    w = torch.ones(n, device=device, dtype=dtype)
    edge = alpha * (n - 1) / 2.0
    left = x < edge
    w = torch.where(left, 0.5 * (1.0 + torch.cos(math.pi * (x / edge - 1.0))), w)
    right = x > (n - 1) - edge
    w = torch.where(
        right,
        0.5 * (1.0 + torch.cos(math.pi * ((x - (n - 1) + edge) / edge))),
        w,
    )
    return w


@lru_cache(maxsize=32)
def _tukey2d_cached(h, w, alpha, device_str, dtype):
    wy = _tukey1d_cached(h, alpha, device_str, dtype)
    wx = _tukey1d_cached(w, alpha, device_str, dtype)
    return wy.unsqueeze(-1) * wx.unsqueeze(0)


def tukey1d(n: int, alpha: float, *, device=None, dtype=torch.float32) -> Tensor:
    """1-D Tukey (cosine-tapered) window of length ``n``.

    ``alpha`` is the fraction of the window that is tapered (0 = rectangular,
    1 = Hann). ``alpha=0.15`` puts a 7.5 % cosine ramp on each end. Cached.
    """
    return _tukey1d_cached(n, float(alpha), _device_key(device), dtype)


def tukey2d(h: int, w: int, alpha: float, *, device=None, dtype=torch.float32) -> Tensor:
    """Outer product 2-D Tukey window. Cached."""
    return _tukey2d_cached(h, w, float(alpha), _device_key(device), dtype)


def bilinear_fit(field: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Least-squares planar fit ``f(x, y) = a · x + b · y + c``.

    `field` has shape `(..., H, W)`. Returns `(a, b, c)` each with shape
    `(...)`. Uses pixel coordinates centred at zero, so the plane reconstructs
    via :func:`apply_plane`.

    Closed-form: with centred coordinates ``⟨x⟩ = ⟨y⟩ = ⟨xy⟩ = 0`` so the
    normal equations decouple into three scalar inversions::

        c = mean(field)
        a = sum(x · field) / sum(x²)
        b = sum(y · field) / sum(y²)

    Roughly 5x faster than ``torch.linalg.lstsq`` on a 132×132 grid and
    avoids allocating the (HW, 3) design matrix.
    """
    *_, h, w = field.shape
    device, dtype = field.device, field.dtype
    y = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2.0
    x = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2.0
    # On the (H, W) grid each unique x_j appears H times and each y_i
    # appears W times, so the design-matrix diagonal entries are
    # H · Σ_j x_j² and W · Σ_i y_i².
    sum_x2_full = (x * x).sum() * h
    sum_y2_full = (y * y).sum() * w
    # Reductions over spatial dims only; leading batch dims pass through.
    c = field.mean(dim=(-2, -1))
    a = (field * x).sum(dim=(-2, -1)) / sum_x2_full
    b = (field * y.view(-1, 1)).sum(dim=(-2, -1)) / sum_y2_full
    return a, b, c


def apply_plane(
    plane: tuple[Tensor, Tensor, Tensor], h: int, w: int, *, device=None, dtype=torch.float32
) -> Tensor:
    """Reconstruct a plane from `(a, b, c)` on an `(H, W)` grid.

    Coordinates are pixel-centred (matching `bilinear_fit`), so the same plane
    that was subtracted can be added back exactly.
    """
    a, b, c = plane
    if device is None:
        device = a.device
    if dtype is None or not torch.is_floating_point(a):
        dtype = torch.float32
    else:
        dtype = a.dtype
    y = torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2.0
    x = torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    a = a.unsqueeze(-1).unsqueeze(-1) if a.ndim else a
    b = b.unsqueeze(-1).unsqueeze(-1) if b.ndim else b
    c = c.unsqueeze(-1).unsqueeze(-1) if c.ndim else c
    return a * xx + b * yy + c


@lru_cache(maxsize=32)
def _extension_window_cached(h, w, r, device_str, dtype):
    device = torch.device(device_str)

    def profile(n):
        p = torch.ones(n + 2 * r, device=device, dtype=dtype)
        ramp = torch.cos(torch.linspace(0.0, math.pi / 2.0, r, device=device, dtype=dtype)) ** 2
        p[:r] = ramp.flip(0)  # 0 -> 1 ascending into the data window
        p[-r:] = ramp  # 1 -> 0 descending into the zero pad
        return p

    return profile(h).unsqueeze(-1) * profile(w).unsqueeze(0)


def extend_feathered(field: Tensor, r: int) -> Tensor:
    """Replicate-extend an ``(…, H, W)`` grid by ``r`` px/side, feathered to zero.

    The ring carries plausible (edge-replicated) values that decay smoothly
    (cosine²) into the zero pad. Used by the *inversion*: with a bare zero
    pad the misfit treats the guard ring as real zero-valued data, so the
    fitted source must explain an abrupt data→0 step at the tile boundary
    and rings there; the feathered extension gives it a smooth story
    instead. Differentiable — ring gradients accumulate onto edge pixels.
    """
    if r <= 0:
        return field
    *batch, h, w = field.shape
    x = field.reshape(-1, 1, h, w)
    x = F.pad(x, (r, r, r, r), mode="replicate").reshape(*batch, h + 2 * r, w + 2 * r)
    win = _extension_window_cached(h, w, int(r), _device_key(field.device), field.dtype)
    return x * win


def _pad_amounts(h: int, w: int, pad_to: int) -> tuple[int, int, int, int]:
    """Return (left, right, top, bottom) for symmetric centre-pad to `pad_to`."""
    if pad_to < max(h, w):
        raise ValueError(f"pad_to={pad_to} must be >= max(h={h}, w={w})")
    pad_h = pad_to - h
    pad_w = pad_to - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return left, right, top, bottom


def pad_centered(field: Tensor, pad_to: int, mode: str = "constant", value: float = 0.0) -> Tensor:
    """Symmetric pad of an `(..., H, W)` tensor to `(pad_to, pad_to)`.

    `mode='constant'` (zero-pad) is the default; `'replicate'` is useful for
    extending DEM-style data over the guard ring.
    """
    *_, h, w = field.shape
    left, right, top, bottom = _pad_amounts(h, w, pad_to)
    if mode == "constant":
        return F.pad(field, (left, right, top, bottom), mode="constant", value=value)
    # `replicate` requires the input to have a leading batch dim; `F.pad` rejects
    # bare 2D tensors. Add/remove a batch dim around the call as needed.
    if field.ndim == 2:
        return F.pad(field.unsqueeze(0), (left, right, top, bottom), mode=mode).squeeze(0)
    return F.pad(field, (left, right, top, bottom), mode=mode)


def crop_centered(field: Tensor, h: int, w: int) -> Tensor:
    """Inverse of `pad_centered` — extract the central `(h, w)` window."""
    *_, ph, pw = field.shape
    if ph < h or pw < w:
        raise ValueError(f"cannot crop ({ph},{pw}) -> ({h},{w})")
    top = (ph - h) // 2
    left = (pw - w) // 2
    return field[..., top : top + h, left : left + w]

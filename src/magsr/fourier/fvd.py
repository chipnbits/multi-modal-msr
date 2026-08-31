"""First-vertical-derivative (1VD/FVD) operator + training loss.

Shares the package's FFT helpers (``_fft_utils``) with the continuation
operators; only the spectral kernel differs:

    upward continuation = exp(-|k|·dz)        (equivalent_layer.py)
    first vertical deriv = |k|                (Blakely) — this file

Detrend on, NO plane re-add (∂z of a linear regional plane ≈ 0, and |k| nulls DC).
Kernel sign is irrelevant: the loss is ``|FVD(pred) − FVD(target)|`` with both fields
through the SAME operator, so it is self-consistent and exactly 0 at ``pred == target``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from magsr.fourier._fft_utils import (
    apply_plane,
    bilinear_fit,
    crop_centered,
    make_wavenumbers_rfft,
    pad_centered,
    tukey2d,
)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def first_vertical_derivative(
    field: Tensor,
    dx: float,
    dy: float,
    *,
    pad_to: int | None = None,
    taper_frac: float = 0.15,
) -> Tensor:
    """First vertical derivative (|k| kernel) of a potential-field grid.

    ``field``: ``(H, W)`` or ``(B, H, W)`` real grid; ``dx``/``dy`` pixel size in metres.
    Mirrors ``inversion.upward_continue_field`` (bilinear detrend → Tukey → zero-pad →
    rfft2) but multiplies by ``|k|`` and does NOT restore the bilinear plane.
    """
    squeeze = field.ndim == 2
    if squeeze:
        field = field.unsqueeze(0)
    *_, h, w = field.shape
    device, dtype = field.device, field.dtype
    if pad_to is None:
        pad_to = _next_pow2(int(1.5 * max(h, w)))

    plane = bilinear_fit(field)  # detrend (NOT restored — vertical deriv of a plane ≈ 0)
    f = field - apply_plane(plane, h, w, device=device, dtype=dtype)
    if taper_frac > 0:
        f = f * tukey2d(h, w, taper_frac, device=device, dtype=dtype)
    f_pad = pad_centered(f, pad_to)

    _, _, k = make_wavenumbers_rfft(pad_to, pad_to, dx, dy, device=device, dtype=dtype)
    out_pad = torch.fft.irfft2(torch.fft.rfft2(f_pad) * k, s=(pad_to, pad_to))  # H = |k|
    out = crop_centered(out_pad, h, w)
    return out.squeeze(0) if squeeze else out


class FVDLoss(nn.Module):
    """L1 between the 1VD of prediction and target.

    Computed only over FULLY-VALID patches (the FFT cannot take NaN; any patch with an
    invalid HR pixel is skipped so hole-fill ringing never pollutes the penalty).
    ``dx``/``dy`` = HR pixel size in metres. Fields are upcast to float32 (rfft needs it).
    """

    def __init__(self, dx: float = 60.0, dy: float = 60.0, taper_frac: float = 0.15) -> None:
        super().__init__()
        self.dx, self.dy, self.taper_frac = dx, dy, taper_frac

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        valid = torch.isfinite(target).flatten(1).all(dim=1)  # (B,) fully-valid patches
        if not bool(valid.any()):
            return pred.sum() * 0.0
        p = pred[valid, 0].float()
        t = target[valid, 0].float()
        fp = first_vertical_derivative(p, self.dx, self.dy, taper_frac=self.taper_frac)
        ft = first_vertical_derivative(t, self.dx, self.dy, taper_frac=self.taper_frac)
        return (fp - ft).abs().mean()

"""Synthetic operator-consistent SR pairs: HR context → (HR target, degraded LR).

Uses the calibrated forward degradation (upward continuation by Δz in the band
measured by the fitting study, then 3×3 block-mean decimation, plus optional
survey noise) to build LR tiles that are EXACTLY consistent with the operator.
Training on these isolates the pure SR difficulty of the task from the real
HR↔LR survey inconsistency (calibration gain, LR compilation noise floor,
leveling) that caps performance on real pairs.

The continuation runs on an oversized real-data context (e.g. 264 px for a
132 px target) so the target's spectrum carries no synthetic edge effects:
at Δz ≤ 250 m the kernel's spatial footprint is a few hundred metres — far
inside the 66-px (≈4 km) discarded ring.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from magsr.fourier._fft_utils import (
    apply_plane,
    bilinear_fit,
    crop_centered,
    extend_feathered,
    make_wavenumbers_rfft,
    pad_centered,
)
from magsr.fourier.equivalent_layer import fit_equivalent_layer, upward_continue


def synth_uc_pair(
    hr_full: Tensor,
    dz: Tensor,
    *,
    noise_nt: float = 1.0,
    out_px: int = 132,
    scale: int = 3,
    dx: float = 60.0,
    pad_to: int = 384,
    feather: int = 26,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """``(B,1,P,P)`` HR context in nT → (HR target ``(B,1,out,out)``, synthetic LR).

    - ``dz``: per-sample continuation heights (m), ``(B,)``.
    - The HR target is the untouched centre crop (NaNs preserved for masked loss).
    - The LR is UC(context, dz) → centre crop → ``scale``× block mean → + white
      noise of ``noise_nt`` (nT RMS). NaNs in the context are filled with the
      patch mean for the FFT only (the plane/mean carries no high-|k| content).
    - The bilinear plane is detrended before the FFT and re-added after — it is
      invariant under continuation, so this is exact, and it keeps the feather
      ring from biting into a strong regional gradient.
    """
    B, _, P, _ = hr_full.shape
    if P < out_px + 2 * (feather // 2):
        raise ValueError(f"context {P}px too small for target {out_px}px")
    x = hr_full[:, 0].float()
    finite = torch.isfinite(x)
    fill = torch.nan_to_num(x, nan=0.0).sum(dim=(-2, -1)) / finite.sum(dim=(-2, -1)).clamp_min(1.0)
    x = torch.where(finite, x, fill.view(-1, 1, 1))

    plane = bilinear_fit(x)
    plane_grid = apply_plane(plane, P, P, device=x.device, dtype=x.dtype)
    detr = x - plane_grid

    _, _, k = make_wavenumbers_rfft(pad_to, pad_to, dx, dx, device=x.device, dtype=x.dtype)
    spec = torch.fft.rfft2(pad_centered(extend_feathered(detr, feather), pad_to))
    kern = torch.exp(-k.unsqueeze(0) * dz.to(x.device, x.dtype).view(-1, 1, 1))
    u = torch.fft.irfft2(spec * kern, s=(pad_to, pad_to))
    u = crop_centered(u, P, P) + plane_grid

    off = (P - out_px) // 2
    uc_center = u[..., off : off + out_px, off : off + out_px]
    lr = F.avg_pool2d(uc_center.unsqueeze(1), scale, scale)
    if noise_nt > 0:
        noise = (
            torch.randn(lr.shape, generator=generator, device=lr.device, dtype=lr.dtype)
            if generator is not None
            else torch.randn_like(lr)
        )
        lr = lr + noise_nt * noise

    hr_center = hr_full[..., off : off + out_px, off : off + out_px]
    return hr_center, lr


def synth_uc_pair_drape(
    hr_full: Tensor,
    dem_hr: Tensor,
    dz: Tensor,
    *,
    noise_nt: float = 1.0,
    out_px: int = 132,
    scale: int = 3,
    dx: float = 60.0,
    pad_to: int = 384,
    clearance_m: float = 60.0,
    layer_depth_m: float = 300.0,
    n_layers: int = 24,
    cg_iters: int = 10,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Drape-aware synthetic pair: EL fit on the survey drape, forward to drape+dz.

    Uses the full equivalent-layer fit σ on ``z_obs = −(clearance + DEM)``
    and forward to the rigid shifted drape ``z_obs − dz``. The generated LR is
    terrain-coupled component of continuation, making DEM a genuinely informative
    input channel for the inverse task.

    ``dem_hr``: DEM on the HR grid of the context, ``(B, P, P)`` metres (+up).
    ``n_layers`` is higher than the single-tile default because one batched
    chessboard spans the whole batch's absolute-altitude range.
    """
    B, _, P, _ = hr_full.shape
    x = hr_full[:, 0].float()
    finite = torch.isfinite(x)
    fill = torch.nan_to_num(x, nan=0.0).sum(dim=(-2, -1)) / finite.sum(dim=(-2, -1)).clamp_min(1.0)
    x = torch.where(finite, x, fill.view(-1, 1, 1))
    # DEM NaN -> patch mean (0-fill would fabricate cliffs in the drape).
    dem = dem_hr.float()
    dfin = torch.isfinite(dem)
    dmean = torch.nan_to_num(dem, nan=0.0).sum(dim=(-2, -1)) / dfin.sum(dim=(-2, -1)).clamp_min(1.0)
    dem = torch.where(dfin, dem, dmean.view(-1, 1, 1))

    with torch.no_grad():
        z_obs = -(clearance_m + dem)  # z-down survey drape
        z_layer = float(z_obs.max()) + layer_depth_m
        layer = fit_equivalent_layer(
            x,
            dx=dx,
            dy=dx,
            z_obs=z_obs,
            z_layer=z_layer,
            pad_to=pad_to,
            cg_iters=cg_iters,
        )
        z_t = z_obs - dz.to(x.device, x.dtype).view(-1, 1, 1)  # rigid shift up
        u = upward_continue(
            layer, z_target=z_t, n_layers=n_layers, z_min=float(z_t.min()), z_max=float(z_t.max())
        )

    off = (P - out_px) // 2
    uc_center = u[..., off : off + out_px, off : off + out_px]
    lr = F.avg_pool2d(uc_center.unsqueeze(1), scale, scale)
    if noise_nt > 0:
        noise = (
            torch.randn(lr.shape, generator=generator, device=lr.device, dtype=lr.dtype)
            if generator is not None
            else torch.randn_like(lr)
        )
        lr = lr + noise_nt * noise
    hr_center = hr_full[..., off : off + out_px, off : off + out_px]
    return hr_center, lr

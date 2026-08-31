"""Validation metrics for magnetic super-resolution.

Pools the per-metric factories and helpers used by WA / KSA training and
evaluation scripts. NaN-masking convention (HR NaNs -> invalid pixels,
zeroed in both tensors before accumulation) is centralized in
:func:`apply_hr_nan_mask` so every call site uses the same rule.
"""

from __future__ import annotations

import math

import torch
from torchmetrics import MeanSquaredError
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure,
    structural_similarity_index_measure,
)
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)

MSSSIM_4SCALE_BETAS = (0.05168, 0.32949, 0.34622, 0.27262)


def apply_hr_nan_mask(
    sr: torch.Tensor, hr: torch.Tensor, nan_value: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero HR-NaN pixels in both tensors; return masked (sr, hr) pair."""
    mask = hr.isfinite().to(sr.dtype)
    return sr * mask, hr.nan_to_num(nan_value) * mask


def rmse_to_psnr(rmse: float, data_range: float = 1.0) -> float:
    """PSNR in dB from a scalar RMSE, assuming inputs scaled to `data_range`."""
    return 20.0 * math.log10(data_range / (rmse + 1e-12))


def make_rmse() -> MeanSquaredError:
    """Torchmetric that accumulates squared errors and returns RMSE on `.compute()`."""
    return MeanSquaredError(squared=False)


def make_ssim(data_range: float = 1.0) -> StructuralSimilarityIndexMeasure:
    return StructuralSimilarityIndexMeasure(data_range=data_range)


def _msssim_kwargs(patch_size: int, data_range: float, betas, *, functional: bool) -> dict:
    """Shared MS-SSIM kwargs: auto 4-scale betas below 161 px (else torchmetrics' 5-scale default);
    `functional=True` adds reduction='none' for the per-patch functional call."""
    if betas is None and patch_size < 161:
        betas = MSSSIM_4SCALE_BETAS
    kwargs: dict = {"data_range": data_range}
    if functional:
        kwargs["reduction"] = "none"
    if betas is not None:
        kwargs["betas"] = betas
    return kwargs


def make_msssim(
    patch_size: int,
    data_range: float = 1.0,
    betas: tuple[float, ...] | None = None,
) -> MultiScaleStructuralSimilarityIndexMeasure:
    """MS-SSIM with auto-picked scale count: 4-scale below 161 px, else default 5-scale."""
    return MultiScaleStructuralSimilarityIndexMeasure(
        **_msssim_kwargs(patch_size, data_range, betas, functional=False)
    )


def per_patch_rmse(sr: torch.Tensor, hr: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """RMSE over valid pixels, one scalar per patch. Inputs `(B, C, H, W)`; returns `(B,)`.

    Assumes `sr` / `hr` are already masked (invalid pixels zeroed via
    :func:`apply_hr_nan_mask`); `mask` supplies the valid-pixel count for the
    denominator.
    """
    sq = (sr - hr) ** 2
    num = sq.flatten(1).sum(dim=1)
    if mask is None:
        den = torch.prod(torch.tensor(sr.shape[-2:], device=sr.device))
    else:
        den = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (num / den).sqrt()


def per_patch_ssim(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-patch SSIM (functional, `reduction='none'`). Returns `(B,)`."""
    return structural_similarity_index_measure(sr, hr, data_range=data_range, reduction="none")


def per_patch_msssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    patch_size: int,
    data_range: float = 1.0,
    betas: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Per-patch MS-SSIM calculated on CPU to avoid NVRTC/GPU issues."""
    kwargs = _msssim_kwargs(patch_size, data_range, betas, functional=True)
    return multiscale_structural_similarity_index_measure(sr.cpu().float(), hr.cpu().float(), **kwargs)


def mean_std(values: torch.Tensor) -> tuple[float, float]:
    """`(mean, unbiased std)` of a 1-D tensor of per-patch metric values.

    Returns `(nan, nan)` for an empty tensor and `(mean, 0.0)` for a single value.
    """
    if values.numel() == 0:
        return float("nan"), float("nan")
    mean = values.mean().item()
    std = values.std(unbiased=True).item() if values.numel() > 1 else 0.0
    return mean, std

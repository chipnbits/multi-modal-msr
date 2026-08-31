"""Loss functions for multi-modal MSR training."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """L1 over HR-valid pixels. Mask is derived from NaNs in `target`."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        diff = (pred - target.nan_to_num(0.0)).abs()
        return diff[mask].mean() if mask.any() else diff.sum() * 0.0

"""Parameter-free bicubic upsampler matching the SR model interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicModel(nn.Module):
    """Bicubic interpolation following the `LR -> SR` model interface."""

    def __init__(self, upscale_factor: int = 4):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.build_kwargs = {"upscale_factor": upscale_factor}

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False)

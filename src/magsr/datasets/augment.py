"""Train-time data augmentation for paired SR tensors."""

from __future__ import annotations

import torch


def random_d4(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Apply the same random D4 transform (rot90^k + optional h/v flips) to all tensors."""
    k = int(torch.randint(0, 4, ()).item())
    hflip = torch.rand(()).item() < 0.5
    vflip = torch.rand(()).item() < 0.5
    out = [torch.rot90(t, k, dims=(-2, -1)) for t in tensors]
    if hflip:
        out = [torch.flip(t, dims=(-1,)) for t in out]
    if vflip:
        out = [torch.flip(t, dims=(-2,)) for t in out]
    return tuple(t.contiguous() for t in out)

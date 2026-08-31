"""RDN† / RDN++ — Residual Dense Network variant from Smith et al. 2022.

Non-GAN variant of ESRGAN+ described in Smith et al., "Magnetic grid
resolution enhancement using machine learning," Ore Geology Reviews 150
(2022), https://doi.org/10.1016/j.oregeorev.2022.105119 (the paper names
this model "RDN†"; file named `rdnpp.py` to avoid the dagger glyph).

Backbone is the ESRGAN+ generator from Rakotonirina & Rasoanaivo (2020),
reference at https://github.com/ncarraz/ESRGANplus. Identical architecture
minus the Gaussian noise injection (γ=0 in Fig. 3). Concretely:

  - 23 stacked `RRDB` Basic Blocks, each containing 3 `ResidualDenseBlock`
    units with β=0.2 residual scaling (Fig. 3).
  - `ResidualDenseBlock` has 5 convs with dense concat + the ESRGAN+
    extra residuals (`conv1x1` skip, `x4 += x2`) shown in Fig. 4.
  - LeakyReLU(0.2) activations throughout.
  - Upsampling is nearest-neighbour + 3×3 conv per stage (Fig. 2), not
    pixel-shuffle.
  - Single-channel I/O for TMI grids (`in_channels=1, out_channels=1`).

Training in the paper: L1 pixel loss only; Adam + OneCycleLR; inputs
# min-max scaled to [0, 1] with training-set global stts.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "MultiDomainAdapter",
    "RDNpp",
    "ResidualDenseBlock",
    "RRDB",
    "rdnpp_default_x4",
    "rdnpp_small_x4",
    "rdnpp_large_x4",
]


class MultiDomainAdapter(nn.Module):
    """Per-domain residual 1×1 conv adapter (Rebuffi et al. NeurIPS 2017).

    Weights and bias are zero-init so the adapter starts as the identity
    (`out = x + 0`); training opens it as needed. Per-sample weight selection
    via `idx` lets a single forward handle mixed-domain batches.
    """

    def __init__(self, channels: int, num_domains: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_domains, channels, channels))
        self.bias = nn.Parameter(torch.zeros(num_domains, channels))

    def forward(self, x: Tensor, idx: Tensor) -> Tensor:
        B, C, H, W = x.shape
        w = self.weight[idx]  # [B, C, C]
        b = self.bias[idx]  # [B, C]
        out = torch.bmm(w, x.reshape(B, C, H * W)).reshape(B, C, H, W)
        return x + out + b[:, :, None, None]


class ResidualDenseBlock(nn.Module):
    """
    5-conv dense block with ESRGAN+ residual connections (no noise injection).
    Matches Fig 4. from (Smith et. al., 2022) and `ResidualDenseBlock` in the ESRGAN+ reference.

    Matches `ResidualDenseBlock_5C` in the ESRGAN+ reference. Dense concat
    across 5 convs; the extra `conv1x1` projection and `x4 += x2` residual
    are the ESRGAN+ additions over vanilla ESRGAN. The Gaussian noise path
    present in ESRGAN+ is omitted — this is the RDN† variant.

    Args:
        nc: number of input channels.
        gc: growth channels per conv.
    """

    def __init__(self, nc: int, gc: int = 32) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(nc, gc, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(nc + 0 * gc, gc, 3, padding=1)
        self.conv2 = nn.Conv2d(nc + 1 * gc, gc, 3, padding=1)
        self.conv3 = nn.Conv2d(nc + 2 * gc, gc, 3, padding=1)
        self.conv4 = nn.Conv2d(nc + 3 * gc, gc, 3, padding=1)
        self.conv5 = nn.Conv2d(nc + 4 * gc, nc, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x2 = x2 + self.conv1x1(x)  # ESRGAN+ extra residual
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x4 = x4 + x2  # ESRGAN+ extra residual
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5


class RRDB(nn.Module):
    """Basic Block: 3 `ResidualDenseBlock`s with β residual scaling (Smith 2022, Fig. 3).

    When `num_domains > 1`, a single `MultiDomainAdapter` is applied to the
    RRDB output (after the global residual). One adapter per RRDB instead of
    one per RDB cuts per-domain capacity 3×.
    """

    def __init__(self, nc: int, gc: int = 32, res_scale: float = 0.2, num_domains: int = 1) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nc, gc)
        self.rdb2 = ResidualDenseBlock(nc, gc)
        self.rdb3 = ResidualDenseBlock(nc, gc)
        self.res_scale = res_scale
        self.adapter = MultiDomainAdapter(nc, num_domains) if num_domains > 1 else None

    def forward(self, x: Tensor, idx: Tensor | None = None) -> Tensor:
        y = self.rdb1(x) * self.res_scale + x
        y = self.rdb2(y) * self.res_scale + y
        out = self.rdb3(y) * self.res_scale + y
        out = out * self.res_scale + x
        if self.adapter is not None:
            out = self.adapter(out, idx)
        return out


class UpsampleBlock(nn.Module):
    """Upsampling block from Figure 2 of Smith 2022"""

    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample(x)
        out = self.conv(out)
        out = self.act(out)
        return out


class RDNpp(nn.Module):
    """RDN† / RDN++ — ESRGAN+ generator backbone without noise injection.

    Args:
        in_channels: input channels (1 for single-product TMI).
        out_channels: output channels (match `in_channels` for pure SR).
        nf: feature channels through the trunk.
        nb: number of stacked Basic Blocks (RRDBs). Paper uses 23.
        gc: growth channels inside each dense block.
        upscale: total upsampling factor; supports 2, 3, 4, 8.
        res_scale: β in Figs. 3 and 4. 0.2 matches ESRGAN+.
        num_domains: if >1, enables per-domain residual adapters in the last
            `domain_rrdbs` RRDBs plus a residual-delta tail. `forward` then
            requires `block_ids` with 1-indexed labels in `[1, num_domains]`.
        domain_rrdbs: number of trailing RRDBs that carry a per-domain adapter
            when `num_domains > 1`. Concentrates per-block specialization near
            the output. Default 5; ignored when `num_domains == 1`.
        up_factors: per-stage upsample factors. Defaults to `[3, 1]` for x3,
            else `[2] * log2(upscale)`. Saved in `build_kwargs`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        nf: int = 64,
        nb: int = 23,
        gc: int = 32,
        upscale: int = 4,
        res_scale: float = 0.2,
        num_domains: int = 1,
        domain_rrdbs: int = 5,
        up_factors: list[int] | None = None,
    ) -> None:
        super().__init__()
        if upscale not in (2, 3, 4, 8):
            raise ValueError(f"upscale must be one of 2, 3, 4, 8; got {upscale}")
        if up_factors is None:
            up_factors = [3, 1] if upscale == 3 else [2] * int(math.log2(upscale))
        domain_rrdbs = max(0, min(domain_rrdbs, nb))
        self.build_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "nf": nf,
            "nb": nb,
            "gc": gc,
            "upscale": upscale,
            "res_scale": res_scale,
            "num_domains": num_domains,
            "domain_rrdbs": domain_rrdbs,
            "up_factors": up_factors,
        }
        self.num_domains = num_domains
        self.domain_rrdbs = domain_rrdbs
        self.out_channels = out_channels

        # Head: shallow feature extraction.
        self.head = nn.Conv2d(in_channels, nf, 3, padding=1)

        # Trunk: nb RRDBs. Only the last `domain_rrdbs` carry an adapter; the
        # earlier ones see num_domains=1 (no adapter created).
        first_domain_idx = nb - domain_rrdbs if num_domains > 1 else nb
        self.body = nn.ModuleList(
            [
                RRDB(nf, gc, res_scale, num_domains=num_domains if i >= first_domain_idx else 1)
                for i in range(nb)
            ]
        )
        self.body_conv = nn.Conv2d(nf, nf, 3, padding=1)

        # Upsample: nearest + conv per stage. x3 defaults to an extra factor-1
        # stage ([3, 1]) to smooth the 3×3 nearest-neighbor block pattern.
        self.up = nn.Sequential(*[UpsampleBlock(nf, f) for f in up_factors])

        # Tail (ESRGAN+ HR_conv0/HR_conv1). num_domains>1 → one shared tail with a per-domain
        # MultiDomainAdapter after each conv (Rebuffi-style). Adapters are zero-init residuals, so
        # weight decay pulls toward the shared tail's output — a valid identity (decaying a
        # standalone tail to zero would blank the SR image instead).
        if num_domains > 1:
            self.tail_conv1 = nn.Conv2d(nf, nf, 3, padding=1)
            self.tail_adapter1 = MultiDomainAdapter(nf, num_domains)
            self.tail_act = nn.LeakyReLU(0.2, inplace=True)
            self.tail_conv2 = nn.Conv2d(nf, out_channels, 3, padding=1)
            self.tail_adapter2 = MultiDomainAdapter(out_channels, num_domains)
        else:
            self.tail = nn.Sequential(
                nn.Conv2d(nf, nf, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(nf, out_channels, 3, padding=1),
            )

    def forward(self, x: Tensor, block_ids: Tensor | None = None) -> Tensor:
        if self.num_domains > 1:
            if block_ids is None:
                raise ValueError("block_ids required when num_domains > 1")
            idx = block_ids - 1  # blocks are 1-indexed (B1/B2/B3); adapters 0-indexed
        else:
            idx = None

        fea = self.head(x)
        trunk = fea
        for rrdb in self.body:
            trunk = rrdb(trunk, idx)
        trunk = self.body_conv(trunk)
        fea = fea + trunk  # global residual around the trunk
        fea = self.up(fea)

        if self.num_domains == 1:
            return self.tail(fea)

        out = self.tail_conv1(fea)
        out = self.tail_adapter1(out, idx)
        out = self.tail_act(out)
        out = self.tail_conv2(out)
        out = self.tail_adapter2(out, idx)
        return out

    def param_groups(
        self,
        *,
        base_weight_decay: float = 0.0,
        domain_weight_decay: float = 0.0,
    ) -> list[dict]:
        """Optimizer param groups split into shared-backbone vs domain-specific.

        Lets the trainer apply a stronger weight decay to per-domain params
        (adapter weights + per-domain tail) without itself knowing how those
        params are named. Returns a single group when `num_domains == 1`.
        """
        if self.num_domains <= 1:
            return [{"params": list(self.parameters()), "weight_decay": base_weight_decay}]
        shared, domain = [], []
        for name, p in self.named_parameters():
            # Matches body.{i}.adapter.* (trunk) and tail_adapter{1,2}.* (tail).
            if "adapter" in name:
                domain.append(p)
            else:
                shared.append(p)
        return [
            {"params": shared, "weight_decay": base_weight_decay},
            {"params": domain, "weight_decay": domain_weight_decay},
        ]


# ============================================================
# Factory presets
# ============================================================
def rdnpp_default_x4(**kwargs: Any) -> RDNpp:
    """Paper default: nb=23, nf=64, gc=32, upscale=4 (RDN†)."""
    return RDNpp(nb=23, nf=64, gc=32, upscale=4, **kwargs)


def rdnpp_small_x4(**kwargs: Any) -> RDNpp:
    """Lighter variant for quick experiments."""
    return RDNpp(nb=8, nf=64, gc=32, upscale=4, **kwargs)


def rdnpp_large_x4(**kwargs: Any) -> RDNpp:
    """Wider growth channels; heavier memory."""
    return RDNpp(nb=23, nf=64, gc=64, upscale=4, **kwargs)

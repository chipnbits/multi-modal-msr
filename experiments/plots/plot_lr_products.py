"""Compact 3-panel view of the KSA LR conditioning products: RTP, 1VD, ANS.

Pulls a structure-rich LR patch through the dataset library (so the products,
paths, and stacking match exactly what the model is fed) and renders the three
reduced-to-pole magnetic products on a shared red--blue (RdBu_r) scale. Sized for
a half-page column (authored ~2x, scaled down at include time).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from magsr import ROOT_FOLDER
from magsr.datasets import KSAAlignedConfig, KSAShieldAlignedDataset, PatchIndex

OUT = ROOT_FOLDER / "figures/lr_products"
OUT.mkdir(parents=True, exist_ok=True)

SCALE = 1.6
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9 * SCALE,
        "axes.titlesize": 10.5 * SCALE,
        "axes.linewidth": 0.8,
        "figure.dpi": 200,
    }
)

IDX = ROOT_FOLDER / "data/processed/ksa_aligned/patch_indices_cellgrid8s12_fold3"


def main():
    cfg = KSAAlignedConfig.from_yaml(lr_products=("RTP",), lr_aux_products=("1VD", "ANS"))
    ds = KSAShieldAlignedDataset(PatchIndex.load(IDX / "test.json"), config=cfg)

    # Pick a fully-valid, high-structure patch (scan a subset by RTP std).
    rng = np.random.default_rng(3)
    best = None
    for i in rng.choice(len(ds), size=300, replace=False):
        s = ds[int(i)]
        lr = s["lr"]["RTP"].numpy()
        if np.isfinite(lr).all():
            sd = float(np.std(lr))
            if best is None or sd > best[0]:
                best = (sd, int(i), s)
    _, idx, s = best
    rtp = s["lr"]["RTP"].numpy()
    vd = s["lr_aux"]["1VD"].numpy()
    ans = s["lr_aux"]["ANS"].numpy()
    print(f"patch idx {idx}  block {s['meta']['block']}  RTP std {best[0]:.0f} nT")

    panels = [
        ("RTP (nT)", rtp, "sym"),
        ("1VD", vd, "sym"),
        ("ANS", ans, "pos"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.15), constrained_layout=True)
    for ax, (title, img, kind) in zip(axes, panels):
        if kind == "sym":
            v = np.nanpercentile(np.abs(img), 98)
            vmin, vmax = -v, v
        else:
            vmin, vmax = 0.0, np.nanpercentile(img, 98)
        im = ax.imshow(img, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, orientation="horizontal")
        cb.ax.tick_params(labelsize=7 * SCALE, length=2)
        cb.locator = plt.MaxNLocator(3)
        cb.update_ticks()
    p = OUT / "lr_products_rtp_1vd_ans.png"
    fig.savefig(p, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()

"""Ensemble the 5-fold AOI reconstructions: pixel-wise mean + std (uncertainty) + heatmaps.

Loads the 5 recon_combo_f{0..4} HR GeoTIFFs (identical grid/transform), each produced by
`experiments/ksa_aligned_reconstruct_region.py` with the matching fold's combo checkpoint,
computes the pixel-wise mean (reconstruction product) and std (inter-fold spread =
uncertainty proxy), writes both as GeoTIFFs sharing the inputs' CRS/transform/nodata, and
renders an uncertainty heatmap plus a 3-panel (mean | std | one member) figure.

Usage:
    uv run python experiments/plots/plot_recon_zone_ensemble.py --dir figures/recon_zone_ensemble
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("figures/recon_zone_ensemble"))
    p.add_argument("--n-folds", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = [args.dir / f"recon_combo_f{n}.tif" for n in range(args.n_folds)]
    for pth in paths:
        if not pth.exists():
            raise FileNotFoundError(pth)

    # Build a cube of shape (n_folds, H, W) from the 5 reconstruction GeoTIFFs.
    profile = None
    transform = crs = None
    stack = []
    for pth in paths:
        with rasterio.open(pth) as s:
            arr = s.read(1).astype(np.float32)
            nd = s.nodata
            if nd is not None and not np.isnan(nd):
                arr = np.where(arr == nd, np.float32(np.nan), arr)
            stack.append(arr)
            if profile is None:
                profile = s.profile.copy()
                transform, crs = s.transform, s.crs
    cube = np.stack(stack, axis=0)  # (5, H, W)

    # Take the mean and std across folds dimension ignoring NaNs
    mean = np.nanmean(cube, axis=0).astype(np.float32)
    std = np.nanstd(cube, axis=0).astype(np.float32)
    valid = np.all(np.isfinite(cube), axis=0)
    mean = np.where(valid, mean, np.float32(np.nan))
    std = np.where(valid, std, np.float32(np.nan))

    # Write back the results to tif files
    profile.update(dtype="float32", count=1, nodata=np.nan)
    mean_tif = args.dir / "ensemble_mean.tif"
    std_tif = args.dir / "ensemble_std.tif"
    with rasterio.open(mean_tif, "w", **profile) as dst:
        dst.write(mean, 1)
    with rasterio.open(std_tif, "w", **profile) as dst:
        dst.write(std, 1)

    # Gather and print some statistics on the uncertainty map
    sv = std[np.isfinite(std)]
    mean_unc = float(sv.mean())
    p95_unc = float(np.percentile(sv, 95))
    print(f"ensemble mean → {mean_tif}")
    print(f"ensemble std  → {std_tif}")
    print(f"uncertainty (std) nT: mean={mean_unc:.2f}  p95={p95_unc:.2f}  max={float(sv.max()):.2f}")

    # World extent for georeferenced imshow.
    H, W = mean.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (W, H)
    extent = [left, right, bottom, top]

    field = np.ma.masked_invalid(mean)
    unc = np.ma.masked_invalid(std)
    member = np.ma.masked_invalid(stack[0])
    fv = mean[np.isfinite(mean)]
    fvmin, fvmax = np.percentile(fv, [1, 99])
    uvmax = p95_unc

    # Standalone uncertainty heatmap.
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(unc, cmap="magma", extent=extent, vmin=0, vmax=uvmax, origin="upper")
    ax.set_title("Inter-fold uncertainty (std of 5-fold ensemble)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("std (nT)")
    fig.tight_layout()
    heat = args.dir / "ensemble_uncertainty_heatmap.png"
    fig.savefig(heat, dpi=150)
    plt.close(fig)

    # 3-panel: mean | std | one member.
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    im0 = axes[0].imshow(field, cmap="viridis", extent=extent, vmin=fvmin, vmax=fvmax, origin="upper")
    axes[0].set_title("Ensemble mean (RTP, nT)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label("nT")
    im1 = axes[1].imshow(unc, cmap="magma", extent=extent, vmin=0, vmax=uvmax, origin="upper")
    axes[1].set_title("Ensemble std (uncertainty, nT)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label("std (nT)")
    im2 = axes[2].imshow(member, cmap="viridis", extent=extent, vmin=fvmin, vmax=fvmax, origin="upper")
    axes[2].set_title("Single member (fold 0)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label("nT")
    for ax in axes:
        ax.set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
    panel = args.dir / "ensemble_mean_std_member.png"
    fig.savefig(panel, dpi=150)
    plt.close(fig)

    print(f"heatmap → {heat}")
    print(f"3-panel → {panel}")


if __name__ == "__main__":
    main()

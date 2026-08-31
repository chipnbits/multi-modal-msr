"""Per-tile ferrous-ratio vs RTP-field examples — the readable version of the null.

Full-raster comparison hides the point in scale; individual 8 km tiles show it. For each
patch: the HR magnetic anomaly (fine, busy, the SR target) next to the co-registered
ferrous ratio (SWIR1/NIR, smooth surface fabric). Same footprint, read straight from the
source rasters via the patch world bounds -- magnetic at 60 m, ferrous at native 30 m.

Run:  uv run python experiments/plots/plot_ferrous_vs_rtp_tiles.py
      uv run python experiments/plots/plot_ferrous_vs_rtp_tiles.py --indices 722 1323 2017 1482
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize, TwoSlopeNorm
from rasterio.windows import from_bounds

from magsr import ROOT_FOLDER
from magsr.datasets import build_ksa_aligned_datasets

NAN_GREY = "#8a8a8a"
INDEX = Path("data/processed/ksa_aligned/patch_indices_cellgrid8_fold3")


def read_window(path: Path, bounds, out_px: int | None = None) -> np.ndarray:
    left, bottom, right, top = bounds
    with rasterio.open(path) as ds:
        win = from_bounds(left, bottom, right, top, ds.transform)
        shape = (out_px, out_px) if out_px else None
        arr = ds.read(1, window=win, out_shape=shape, boundless=True, fill_value=np.nan).astype(np.float32)
    arr[arr < -9e4] = np.nan
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indices", type=int, nargs="+", default=[722, 1323, 2017, 1482])
    p.add_argument(
        "--out", type=Path, default=ROOT_FOLDER / "figures" / "talk" / "ferrous_vs_rtp_tiles.png"
    )
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = build_ksa_aligned_datasets(index_dir=INDEX)["test"]
    mag_path = Path(ds.config.hr_product_path(ds.config.hr_products[0]))
    b6_path, b5_path = Path(ds.config.ms_band_path(6)), Path(ds.config.ms_band_path(5))

    n = len(args.indices)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.9), constrained_layout=True)
    mag_cmap = plt.get_cmap("RdBu_r").copy()
    mag_cmap.set_bad(NAN_GREY)
    fer_cmap = plt.get_cmap("magma").copy()
    fer_cmap.set_bad(NAN_GREY)

    def show(ax, arr, cmap, norm, title):
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("black")
            sp.set_linewidth(0.8)
        return im

    for j, idx in enumerate(args.indices):
        t = ds.index.patches[idx]
        bounds = (t.left, t.bottom, t.right, t.top)
        mag = read_window(mag_path, bounds)  # ~132 px, 60 m
        b6 = read_window(b6_path, bounds, out_px=264)  # native 30 m
        b5 = read_window(b5_path, bounds, out_px=264)
        fer = np.where(np.abs(b5) > 1e-6, b6 / b5, np.nan)

        mv = mag[np.isfinite(mag)]
        lo, hi = np.percentile(mv, [2, 98])
        mnorm = (
            TwoSlopeNorm(vmin=float(lo), vcenter=0.0, vmax=float(hi))
            if lo < 0 < hi
            else Normalize(float(lo), float(hi))
        )
        fv = fer[np.isfinite(fer)]
        flo, fhi = np.percentile(fv, [2, 98])

        blk = t.name.split("_")[2]  # e.g. 'B2'
        im_m = show(axes[0, j], mag, mag_cmap, mnorm, f"{blk}  #{idx}  ·  RTP field")
        im_f = show(axes[1, j], fer, fer_cmap, Normalize(float(flo), float(fhi)), "ferrous ratio (b6/b5)")
        fig.colorbar(im_m, ax=axes[0, j], fraction=0.046, pad=0.02).set_label("nT", fontsize=8)
        fig.colorbar(im_f, ax=axes[1, j], fraction=0.046, pad=0.02)

    axes[0, 0].set_ylabel("Magnetic anomaly", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Surface reflectance", fontsize=12, fontweight="bold")
    fig.suptitle(
        "Same 8 km tiles: the magnetic detail and the ferrous ratio are unrelated",
        fontsize=14,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}   (tiles {args.indices})")


if __name__ == "__main__":
    main()

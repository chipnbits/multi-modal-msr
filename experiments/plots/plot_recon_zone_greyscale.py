"""Regenerate the 3-panel AOI figure: True HR | ensemble-mean reconstruction | inter-fold std.

Reproduces ``figures/recon_zone_ensemble/greyscale_hr_mean_std.png`` (whose original
ad-hoc script was not committed) from the surviving ensemble GeoTIFFs, with fonts
sized for a paper figure. The left panel reads the true AMF-RTP HR reprojected onto
the ensemble grid, so it shows the *known HR coverage* (smaller than the
reconstruction footprint); the centre/right panels are the 5-fold ensemble mean and
inter-fold std written by ``plot_recon_zone_ensemble.py``.

HR and mean share one symmetric RdBu_r range (±p95 of their pooled magnitudes, ≈330 nT
— the value the original used); std is magma on ``[0, p95]``. Masked pixels render grey.

    uv run python experiments/plots/plot_recon_zone_greyscale.py
    uv run python experiments/plots/plot_recon_zone_greyscale.py --font-scale 1.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig

MASK_GREY = "0.6"


def read_field(path: Path) -> tuple[np.ndarray, list[float]]:
    """Read a single-band raster to float32 with NaN nodata and a km extent."""
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float32)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            a = np.where(a == ds.nodata, np.float32(np.nan), a)
        left, top = ds.transform * (0, 0)
        right, bottom = ds.transform * (ds.width, ds.height)
    return a, [left / 1e3, right / 1e3, bottom / 1e3, top / 1e3]


def hr_on_grid(ref_path: Path, hr_path: Path) -> np.ndarray:
    """True HR AMF-RTP nearest-resampled onto the reference raster's grid (NaN off-coverage)."""
    with rasterio.open(ref_path) as ref:
        dst = np.full((ref.height, ref.width), np.nan, np.float32)
        with rasterio.open(hr_path) as src:
            reproject(
                rasterio.band(src, 1),
                dst,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
    return dst


def panel(ax, img, extent, *, cmap, vmin, vmax, title, cbar_label, fig, fs, show_ylabel):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(MASK_GREY)
    im = ax.imshow(
        np.ma.masked_invalid(img),
        cmap=cm,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
    )
    ax.set_facecolor(MASK_GREY)
    ax.set_title(title, fontsize=fs["title"], pad=10)
    ax.set_xlabel("Easting (km)", fontsize=fs["label"])
    if show_ylabel:
        ax.set_ylabel("Northing (km)", fontsize=fs["label"])
    ax.tick_params(labelsize=fs["tick"])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=fs["label"])
    cb.ax.tick_params(labelsize=fs["tick"])
    return im


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=Path("figures/recon_zone_ensemble"))
    p.add_argument("--hr-product", default="AMF_RTP")
    p.add_argument(
        "--field-pct", type=float, default=95.0, help="symmetric percentile for the HR/mean range"
    )
    p.add_argument("--std-pct", type=float, default=99.0, help="upper percentile for the std range")
    p.add_argument("--font-scale", type=float, default=1.0, help="multiply all paper font sizes")
    p.add_argument("--out", type=Path, default=None, help="default: <dir>/greyscale_hr_mean_std.png")
    args = p.parse_args()

    cfg = KSAAlignedConfig.default()
    mean_path = args.dir / "ensemble_mean.tif"
    std_path = args.dir / "ensemble_std.tif"
    for q in (mean_path, std_path):
        if not q.exists():
            raise FileNotFoundError(q)

    mean, extent = read_field(mean_path)
    std, _ = read_field(std_path)
    hr = hr_on_grid(mean_path, cfg.hr_product_path(args.hr_product))

    # Shared symmetric range for the two field panels (matches the original ≈±330 nT).
    pooled = np.concatenate([hr[np.isfinite(hr)], mean[np.isfinite(mean)]])
    fvmax = float(np.percentile(np.abs(pooled), args.field_pct))
    uvmax = float(np.percentile(std[np.isfinite(std)], args.std_pct))
    print(
        f"field range ±{fvmax:.0f} nT (p{args.field_pct:g})  |  std range 0–{uvmax:.0f} nT (p{args.std_pct:g})"
    )

    s = args.font_scale
    fs = {"title": 22 * s, "label": 19 * s, "tick": 15 * s}

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.6), constrained_layout=True)
    panel(
        axes[0],
        hr,
        extent,
        cmap="RdBu_r",
        vmin=-fvmax,
        vmax=fvmax,
        title="True HR (AMF-RTP, known coverage)",
        cbar_label="nT",
        fig=fig,
        fs=fs,
        show_ylabel=True,
    )
    panel(
        axes[1],
        mean,
        extent,
        cmap="RdBu_r",
        vmin=-fvmax,
        vmax=fvmax,
        title="Ensemble mean reconstruction",
        cbar_label="nT",
        fig=fig,
        fs=fs,
        show_ylabel=False,
    )
    panel(
        axes[2],
        std,
        extent,
        cmap="magma",
        vmin=0,
        vmax=uvmax,
        title="Inter-fold std (uncertainty)",
        cbar_label="std (nT)",
        fig=fig,
        fs=fs,
        show_ylabel=False,
    )

    out = args.out or (args.dir / "greyscale_hr_mean_std.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

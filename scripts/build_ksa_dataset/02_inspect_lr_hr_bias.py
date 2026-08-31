"""Visualize the spatial bias between an LR and HR product on the aligned KSA grid.

Block-averages HR onto the LR grid, computes `LR - HR`, smooths it, and saves
a 4-panel figure plus a residual histogram. The block-averaged HR raster is
cached under `DATA_DIR/processed/ksa_aligned/hr_at_lr_grid/` keyed by
`(product, scale, lr_shape, source_mtime)`.

Examples:
    uv run python scripts/build_ksa_dataset/02_inspect_lr_hr_bias.py
    uv run python scripts/build_ksa_dataset/02_inspect_lr_hr_bias.py --hr AMF --lr TMI --sigma 80
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

from magsr import DATA_DIR, ROOT_FOLDER
from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig

CACHE_DIR = DATA_DIR / "processed" / "ksa_aligned" / "hr_at_lr_grid"
DEFAULT_STATS_DIR = DATA_DIR / "processed" / "ksa_aligned" / "block_means"
DEFAULT_FIG_DIR = ROOT_FOLDER / "figures" / "ksa_lr_hr_bias"
SENTINEL_ABS_THRESHOLD = 1e5  # HR AMF_RTP has stray -999999 px; legit values cap ~±5000 nT.


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, out_dtype=np.float32)
        nd = src.nodata
    if nd is not None and not np.isnan(nd):
        arr[arr == nd] = np.nan
    arr[np.abs(arr) > SENTINEL_ABS_THRESHOLD] = np.nan
    return arr


def block_mean_scale_down(
    hr: np.ndarray, scale: int, lr_shape: tuple[int, int], chunk_rows: int = 512
) -> np.ndarray:
    """NaN-aware (scale x scale) block-mean of `hr` onto `lr_shape`, streamed in row chunks."""
    H, W = lr_shape
    out = np.empty((H, W), dtype=np.float32)
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        block = hr[r0 * scale : r1 * scale, : W * scale].reshape(r1 - r0, scale, W, scale)
        valid = np.isfinite(block)
        s = np.where(valid, block, 0.0).sum(axis=(1, 3))
        n = valid.sum(axis=(1, 3))
        out[r0:r1] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def cached_block_mean(
    hr_path: Path, scale: int, lr_shape: tuple[int, int], *, refresh: bool = False
) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mtime = int(hr_path.stat().st_mtime)
    key = f"{hr_path.stem}_s{scale}_{lr_shape[0]}x{lr_shape[1]}_m{mtime}.npy"
    cache_path = CACHE_DIR / key
    if cache_path.exists() and not refresh:
        return np.load(cache_path)
    hr = load_raster(hr_path)
    out = block_mean_scale_down(hr, scale, lr_shape)
    np.save(cache_path, out)
    del hr
    gc.collect()
    return out


def load_mask_uint8(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8)


def load_block_means(stats_path: Path, side: str) -> dict[int, float]:
    """Read `{block: mean}` for `'hr'` or `'lr'` from a per_block_histograms.py JSON."""
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found. Run " "scripts/build_ksa_dataset/01_per_block_histograms.py first."
        )
    payload = json.loads(stats_path.read_text())
    return {int(b): float(payload["blocks"][b][side]["mean"]) for b in payload["blocks"]}


def subtract_per_block_mean_inplace(
    arr: np.ndarray, mask: np.ndarray, block_means: dict[int, float]
) -> np.ndarray:
    """Shift each block's pixels in `arr` by `−block_means[block]`, in place."""
    for b, mu in block_means.items():
        sel = mask == b
        arr[sel] -= mu
    return arr


def nan_gaussian(a: np.ndarray, sigma: float, min_weight: float = 0.05) -> np.ndarray:
    valid = np.isfinite(a).astype(np.float32)
    num = gaussian_filter(np.where(valid > 0, a, 0.0), sigma=sigma)
    den = gaussian_filter(valid, sigma=sigma)
    return np.where(den > min_weight, num / np.maximum(den, 1e-9), np.nan)


def plot_bias(
    hr_lr: np.ndarray,
    lr: np.ndarray,
    residual: np.ndarray,
    smooth: np.ndarray,
    *,
    hr_product: str,
    lr_product: str,
    sigma: float,
    px_m: int,
    display_stride: int = 4,
) -> tuple[plt.Figure, plt.Figure]:
    mag_vmin, mag_vmax = np.nanpercentile(
        np.concatenate(
            [
                hr_lr[np.isfinite(hr_lr)].ravel(),
                lr[np.isfinite(lr)].ravel(),
            ]
        ),
        [2, 98],
    )
    clim = float(np.nanpercentile(np.abs(residual), 98))

    s = display_stride
    hr_show = hr_lr[::s, ::s]
    lr_show = lr[::s, ::s]
    res_show = residual[::s, ::s]
    smooth_show = smooth[::s, ::s]

    div_cmap = plt.get_cmap("RdBu_r").copy()
    div_cmap.set_bad("0.7")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(hr_show, cmap="gray", vmin=mag_vmin, vmax=mag_vmax)
    axes[0].set_title(f"HR {hr_product} (avg → LR grid)")
    axes[1].imshow(lr_show, cmap="gray", vmin=mag_vmin, vmax=mag_vmax)
    axes[1].set_title(f"LR {lr_product}")
    axes[2].imshow(res_show, cmap=div_cmap, vmin=-clim, vmax=clim)
    axes[2].set_title(f"LR − HR  [μ={np.nanmean(residual):+.1f}, σ={np.nanstd(residual):.1f} nT]")
    axes[3].imshow(smooth_show, cmap=div_cmap, vmin=-clim, vmax=clim)
    axes[3].set_title(f"Smoothed (σ={sigma:g} LR px ≈ {sigma * px_m / 1000:.1f} km)")
    for ax in axes[:2]:
        ax.set_facecolor("0.92")
    for ax in axes[2:]:
        ax.set_facecolor("0.7")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    fig_h, ax_h = plt.subplots(figsize=(7, 3))
    valid = residual[np.isfinite(residual)].ravel()
    xlo, xhi = np.percentile(valid, [0.5, 99.5])
    ax_h.hist(valid, bins=np.linspace(xlo, xhi, 200))
    ax_h.axvline(valid.mean(), color="r", lw=1, label=f"μ={valid.mean():+.1f}")
    ax_h.set_xlim(xlo, xhi)
    ax_h.set_xlabel("LR − HR (nT)")
    ax_h.set_ylabel("LR pixels")
    ax_h.legend()
    fig_h.tight_layout()

    return fig, fig_h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr", default="AMF_RTP", help="HR product (default: AMF_RTP)")
    parser.add_argument("--lr", default="RTP", help="LR product (default: RTP)")
    parser.add_argument(
        "--sigma", type=float, default=12.0, help="Smoothing kernel σ in LR pixels (default: 12)"
    )
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute the block-meaned HR cache")
    parser.add_argument(
        "--source-hr",
        action="store_true",
        help="Force the source HR raster (skip the `_blockwise_unbiased.tif` preference in `hr_product_path`).",
    )
    parser.add_argument(
        "--zero-mean-hr",
        action="store_true",
        help="Subtract per-block trimmed HR mean (loaded from block-means JSON) before computing residual.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=DEFAULT_STATS_DIR,
        help=f"Where to read block-means JSON (default: {DEFAULT_STATS_DIR.relative_to(ROOT_FOLDER)})",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Where to save figures (default: {DEFAULT_FIG_DIR.relative_to(ROOT_FOLDER)})",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (headless mode)")
    args = parser.parse_args()

    cfg = KSAAlignedConfig.default()
    lr = load_raster(cfg.lr_product_path(args.lr))
    hr_path = cfg.hr_source_product_path(args.hr) if args.source_hr else cfg.hr_product_path(args.hr)
    hr_tag = "norm" if hr_path == cfg.hr_normalized_product_path(args.hr) else "source"
    print(f"HR raster: {hr_path.name}  (tag: {hr_tag})")
    hr_lr = cached_block_mean(hr_path, cfg.lr_scale, lr.shape, refresh=args.refresh_cache)
    print(f"hr_lr valid: {np.isfinite(hr_lr).mean():.1%}  " f"lr valid: {np.isfinite(lr).mean():.1%}")

    hr_label = args.hr
    if args.zero_mean_hr:
        stats_path = args.stats_dir / f"{args.hr}_vs_{args.lr}_block_means.json"
        hr_means = load_block_means(stats_path, side="hr")
        mask_hr = load_mask_uint8(cfg.mask_path)
        s = cfg.lr_scale
        H_lr, W_lr = lr.shape
        mask_lr = np.ascontiguousarray(mask_hr[1::s, 1::s][:H_lr, :W_lr])
        del mask_hr
        gc.collect()
        subtract_per_block_mean_inplace(hr_lr, mask_lr, hr_means)
        print(
            "Applied per-block HR mean removal: "
            + ", ".join(f"B{b}={mu:+.1f}" for b, mu in sorted(hr_means.items()))
        )
        hr_label = f"{args.hr} − μ_block"
        del mask_lr
        gc.collect()

    residual = lr - hr_lr
    smooth = nan_gaussian(residual, sigma=args.sigma)

    fig, fig_h = plot_bias(
        hr_lr,
        lr,
        residual,
        smooth,
        hr_product=hr_label,
        lr_product=args.lr,
        sigma=args.sigma,
        px_m=cfg.lr_px_m,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.hr}_vs_{args.lr}_s{int(args.sigma)}_{hr_tag}"
    if args.zero_mean_hr:
        stem += "_zerohr"
    fig_path = args.save_dir / f"{stem}_panels.png"
    hist_path = args.save_dir / f"{stem}_hist.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    fig_h.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_path}")
    print(f"Saved {hist_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

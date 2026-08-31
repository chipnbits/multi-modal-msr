"""Overlay HR and LR raw pixel histograms per KSA survey block.

Uses `magnetic_mask_grid60m.tif` (block IDs 0/1/2/3) to split each raster's
pixels into per-block subsets, then plots HR + LR distributions overlaid
per block side by side.

Examples:
    uv run python scripts/build_ksa_dataset/01_per_block_histograms.py
    uv run python scripts/build_ksa_dataset/01_per_block_histograms.py --hr AMF --lr TMI --log-y
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from magsr import DATA_DIR, ROOT_FOLDER
from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig

DEFAULT_FIG_DIR = ROOT_FOLDER / "figures" / "ksa_lr_hr_bias"
DEFAULT_STATS_DIR = DATA_DIR / "processed" / "ksa_aligned" / "block_means"
SENTINEL_ABS_THRESHOLD = 1e5  # HR AMF_RTP has stray -999999 px; legit values cap ~±5000 nT.
PCT_CLIP = (1.0, 99.0)  # heavy-tail trim used for both display and saved means.


def trimmed_stats(values: np.ndarray, pct: tuple[float, float] = PCT_CLIP) -> dict[str, float | int]:
    """Mean / std / count after clipping to `pct` percentile range."""
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(values, pct)
    clipped = values[(values >= lo) & (values <= hi)]
    return {
        "mean": float(clipped.mean()),
        "std": float(clipped.std()),
        "n": int(clipped.size),
        "lo": float(lo),
        "hi": float(hi),
    }


def _scrub_nan(arr: np.ndarray, nd: float | None) -> np.ndarray:
    if nd is not None and not np.isnan(nd):
        arr[arr == nd] = np.nan
    arr[np.abs(arr) > SENTINEL_ABS_THRESHOLD] = np.nan
    return arr


def load_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1, out_dtype=np.float32)
        nd = src.nodata
    return _scrub_nan(arr, nd)


def load_mask_uint8(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8)


def per_block_values(arr: np.ndarray, mask: np.ndarray, blocks: list[int]) -> dict[int, np.ndarray]:
    return {b: arr[(mask == b) & np.isfinite(arr)].ravel().astype(np.float32, copy=True) for b in blocks}


def streamed_per_block_values(
    path: Path,
    mask_hr: np.ndarray,
    blocks: list[int],
    chunk_rows: int = 1024,
) -> dict[int, np.ndarray]:
    """Stream HR raster in row chunks; return {block: float32 1D values}.

    Avoids holding the full HR array in memory — peak ≈ chunk_rows*W*4 bytes
    plus the per-block accumulators.
    """
    out: dict[int, list[np.ndarray]] = {b: [] for b in blocks}
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        nd = src.nodata
        for r0 in range(0, H, chunk_rows):
            r1 = min(r0 + chunk_rows, H)
            window = rasterio.windows.Window(0, r0, W, r1 - r0)
            arr = src.read(1, window=window, out_dtype=np.float32)
            _scrub_nan(arr, nd)
            chunk_mask = mask_hr[r0:r1]
            valid = np.isfinite(arr)
            for b in blocks:
                sel = (chunk_mask == b) & valid
                if sel.any():
                    out[b].append(arr[sel].copy())
    return {b: np.concatenate(v) if v else np.empty(0, np.float32) for b, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr", default="AMF_RTP", help="HR product (default: AMF_RTP)")
    parser.add_argument("--lr", default="RTP", help="LR product (default: RTP)")
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins (default: 200)")
    parser.add_argument("--log-y", action="store_true", help="Log scale on y-axis")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Where to save figures (default: {DEFAULT_FIG_DIR.relative_to(ROOT_FOLDER)})",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=DEFAULT_STATS_DIR,
        help=f"Where to save block-mean JSON (default: {DEFAULT_STATS_DIR.relative_to(ROOT_FOLDER)})",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (headless mode)")
    args = parser.parse_args()

    cfg = KSAAlignedConfig.default()
    mask_hr = load_mask_uint8(cfg.mask_path)
    blocks = sorted(int(b) for b in np.unique(mask_hr) if b > 0)
    print(f"Discovered blocks: {blocks}")

    lr = load_raster(cfg.lr_product_path(args.lr))
    H_lr, W_lr = lr.shape
    s = cfg.lr_scale
    mask_lr = mask_hr[1::s, 1::s][:H_lr, :W_lr]
    lr_per_block = per_block_values(lr, mask_lr, blocks)
    del lr, mask_lr
    gc.collect()

    hr_per_block = streamed_per_block_values(cfg.hr_source_product_path(args.hr), mask_hr, blocks)
    del mask_hr
    gc.collect()

    hr_stats = {b: trimmed_stats(hr_per_block[b]) for b in blocks}
    lr_stats = {b: trimmed_stats(lr_per_block[b]) for b in blocks}

    pooled = np.concatenate(list(hr_per_block.values()) + list(lr_per_block.values()))
    xlo, xhi = np.percentile(pooled, list(PCT_CLIP))
    bins = np.linspace(xlo, xhi, args.bins)

    fig, axes = plt.subplots(1, len(blocks), figsize=(5 * len(blocks), 4), squeeze=False)
    axes = axes[0]

    for ax, b in zip(axes, blocks):
        v_hr = hr_per_block[b]
        v_lr = lr_per_block[b]
        mu_hr, mu_lr = hr_stats[b]["mean"], lr_stats[b]["mean"]

        ax.hist(
            v_hr, bins=bins, alpha=0.5, color="C0", density=True, label=f"HR {args.hr}  n={v_hr.size:,}"
        )
        ax.hist(
            v_lr, bins=bins, alpha=0.5, color="C1", density=True, label=f"LR {args.lr}  n={v_lr.size:,}"
        )
        ax.axvline(mu_hr, color="C0", lw=1, ls="--")
        ax.axvline(mu_lr, color="C1", lw=1, ls="--")

        ax.set_title(f"Block {b}\nμ_HR={mu_hr:+.1f}   μ_LR={mu_lr:+.1f}   Δ={mu_lr - mu_hr:+.1f} nT")
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("pixel value (nT)")
        ax.set_ylabel("density")
        if args.log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    fig.suptitle(f"HR {args.hr} vs LR {args.lr} — per-block pixel distributions", y=1.02)
    fig.tight_layout()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    out = args.save_dir / f"{args.hr}_vs_{args.lr}_per_block_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    fig2, (ax_hr, ax_lr) = plt.subplots(1, 2, figsize=(12, 4))
    block_colors = [f"C{i}" for i in range(len(blocks))]

    for product_label, ax, per_block, stats in (
        (f"HR {args.hr}", ax_hr, hr_per_block, hr_stats),
        (f"LR {args.lr}", ax_lr, lr_per_block, lr_stats),
    ):
        pooled_p = np.concatenate(list(per_block.values()))
        plo, phi = np.percentile(pooled_p, list(PCT_CLIP))
        bins_p = np.linspace(plo, phi, args.bins)
        for b, color in zip(blocks, block_colors):
            v = per_block[b]
            mu = stats[b]["mean"]
            ax.hist(
                v,
                bins=bins_p,
                histtype="step",
                color=color,
                lw=1.5,
                density=True,
                label=f"Block {b}  μ={mu:+.1f}  n={v.size:,}",
            )
            ax.axvline(mu, color=color, lw=1, ls="--", alpha=0.7)
        ax.set_xlim(plo, phi)
        ax.set_title(f"{product_label} — across blocks")
        ax.set_xlabel("pixel value (nT)")
        ax.set_ylabel("density")
        if args.log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    fig2.tight_layout()
    out2 = args.save_dir / f"{args.hr}_vs_{args.lr}_block_overlay.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    stats_payload = {
        "hr_product": args.hr,
        "lr_product": args.lr,
        "percentile_clip": list(PCT_CLIP),
        "blocks": {str(b): {"hr": hr_stats[b], "lr": lr_stats[b]} for b in blocks},
    }
    stats_path = args.stats_dir / f"{args.hr}_vs_{args.lr}_block_means.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2))
    print(f"Saved {stats_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

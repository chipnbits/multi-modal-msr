"""Compute per-block + global percentile vmin/vmax on the aligned KSA HR raster.

Streams the AMF_RTP HR raster (`cfg.hr_product_path("AMF_RTP")`, which already
resolves to the `_blockwise_unbiased.tif` zero-mean version) row-chunked, masks
each row by the 60 m block-ID raster, and accumulates per-block float arrays.
For each block and globally, computes `np.percentile([q_low, q_high])` and
writes one JSON to `cfg.normalization_path` (`patch_indices/normalization.json`).

Also saves a side-by-side comparison figure overlaying per-block histograms
scaled by the global vs blockwise (vmin, vmax) — the visual diagnostic for
whether per-block scaling is necessary.

Examples:
    uv run python scripts/build_ksa_dataset/04_compute_ksa_aligned_normalization.py
    uv run python scripts/build_ksa_dataset/04_compute_ksa_aligned_normalization.py --q-low 0.5 --q-high 99.5 --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from magsr import ROOT_FOLDER
from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig
from magsr.normalize import Normalizer

DEFAULT_FIG_DIR = ROOT_FOLDER / "figures" / "ksa_normalization"
SENTINEL_ABS_THRESHOLD = 1e5


def _scrub_nan(arr: np.ndarray, nd: float | None) -> np.ndarray:
    if nd is not None and not np.isnan(nd):
        arr[arr == nd] = np.nan
    arr[np.abs(arr) > SENTINEL_ABS_THRESHOLD] = np.nan
    return arr


def load_mask_uint8(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8)


def streamed_per_block_values(
    path: Path,
    mask_hr: np.ndarray,
    blocks: list[int],
    chunk_rows: int = 1024,
) -> dict[int, np.ndarray]:
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


def percentile_bounds(values: np.ndarray, q_low: float, q_high: float) -> dict[str, float | int]:
    if values.size == 0:
        return {"vmin": float("nan"), "vmax": float("nan"), "n": 0}
    vmin, vmax = np.percentile(values, [q_low, q_high])
    return {"vmin": float(vmin), "vmax": float(vmax), "n": int(values.size)}


def plot_global_vs_blockwise(
    per_block: dict[int, np.ndarray],
    global_bounds: dict[str, float | int],
    block_bounds: dict[int, dict[str, float | int]],
    *,
    bins: int = 200,
    save_path: Path,
) -> None:
    """Two panels: each block scaled to [0, 1] under global vs its own bounds."""
    fig, (ax_g, ax_b) = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    block_colors = [f"C{i}" for i in range(len(per_block))]
    edges = np.linspace(0.0, 1.0, bins + 1)

    g_norm = Normalizer.from_stats(global_bounds)
    g_vmin, g_vmax = g_norm.vmin, g_norm.vmax
    for (b, v), color in zip(per_block.items(), block_colors):
        scaled = g_norm.normalize(v)
        ax_g.hist(
            scaled,
            bins=edges,
            histtype="step",
            color=color,
            lw=1.5,
            density=True,
            label=f"Block {b}  n={v.size:,}",
        )
    ax_g.set_title(f"Global scaling  (vmin={g_vmin:+.1f}, vmax={g_vmax:+.1f})")
    ax_g.set_xlabel("normalized AMF_RTP")
    ax_g.set_ylabel("density")
    ax_g.legend(fontsize=8)

    for (b, v), color in zip(per_block.items(), block_colors):
        bb = block_bounds[b]
        scaled = Normalizer.from_stats(bb).normalize(v)
        ax_b.hist(
            scaled,
            bins=edges,
            histtype="step",
            color=color,
            lw=1.5,
            density=True,
            label=f"Block {b}  vmin={bb['vmin']:+.1f}  vmax={bb['vmax']:+.1f}",
        )
    ax_b.set_title("Blockwise scaling")
    ax_b.set_xlabel("normalized AMF_RTP")
    ax_b.legend(fontsize=8)

    fig.suptitle("AMF_RTP HR — per-block distributions in [0, 1]", y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr", default="AMF_RTP", help="HR product (default: AMF_RTP)")
    parser.add_argument("--q-low", type=float, default=1.0)
    parser.add_argument("--q-high", type=float, default=99.0)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins for the comparison figure")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Where to save the comparison figure (default: {DEFAULT_FIG_DIR.relative_to(ROOT_FOLDER)})",
    )
    parser.add_argument("--no-fig", action="store_true", help="Skip the comparison figure")
    parser.add_argument("--force", action="store_true", help="Overwrite normalization.json if it exists")
    args = parser.parse_args()

    cfg = KSAAlignedConfig.default()
    out_path = cfg.normalization_path
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists. Pass --force to overwrite.")

    mask = load_mask_uint8(cfg.mask_path)
    blocks = sorted(int(b) for b in np.unique(mask) if b > 0)
    print(f"Discovered blocks: {blocks}")

    hr_path = cfg.hr_product_path(args.hr)
    print(f"Scanning {hr_path}")
    per_block = streamed_per_block_values(hr_path, mask, blocks, chunk_rows=args.chunk_rows)

    block_bounds = {b: percentile_bounds(per_block[b], args.q_low, args.q_high) for b in blocks}
    pooled = np.concatenate([per_block[b] for b in blocks])
    global_bounds = percentile_bounds(pooled, args.q_low, args.q_high)

    stats = {
        "product": args.hr,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "global": global_bounds,
        "blocks": {str(b): block_bounds[b] for b in blocks},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")
    print(
        f"  global: vmin={global_bounds['vmin']:+.2f}  vmax={global_bounds['vmax']:+.2f}  n={global_bounds['n']:,}"
    )
    for b in blocks:
        bb = block_bounds[b]
        print(f"  block {b}: vmin={bb['vmin']:+.2f}  vmax={bb['vmax']:+.2f}  n={bb['n']:,}")

    if not args.no_fig:
        fig_path = args.fig_dir / f"{args.hr}_global_vs_blockwise.png"
        plot_global_vs_blockwise(per_block, global_bounds, block_bounds, bins=args.bins, save_path=fig_path)
        print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()

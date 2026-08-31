"""Apply per-block HR mean removal to the source HR raster and save a new GeoTIFF.

Reads block-mean stats from the JSON written by `01_per_block_histograms.py`,
streams the source HR raster in row windows, subtracts each block's HR mean and
writes the corrected raster preserving the original georeferencing.

Pixels outside any survey block (mask == 0) and NaN pixels pass through unchanged.

Examples:
    uv run python scripts/build_ksa_dataset/03_apply_block_normalization.py
    uv run python scripts/build_ksa_dataset/03_apply_block_normalization.py --hr AMF --lr TMI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from magsr import DATA_DIR, ROOT_FOLDER
from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig

DEFAULT_STATS_DIR = DATA_DIR / "processed" / "ksa_aligned" / "block_means"
SENTINEL_ABS_THRESHOLD = 1e5


def load_block_means(stats_path: Path, side: str = "hr") -> dict[int, float]:
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found. Run " "scripts/build_ksa_dataset/01_per_block_histograms.py first."
        )
    payload = json.loads(stats_path.read_text())
    return {int(b): float(payload["blocks"][b][side]["mean"]) for b in payload["blocks"]}


def apply_correction_streamed(
    hr_path: Path,
    mask_path: Path,
    block_means: dict[int, float],
    out_path: Path,
    *,
    chunk_rows: int = 1024,
) -> tuple[int, int]:
    """Stream HR + mask in row windows, write corrected raster.

    Returns `(n_written, n_corrected)` — total written pixels and the subset
    that actually had a per-block shift applied.
    """
    n_total = 0
    n_corrected = 0
    with rasterio.open(hr_path) as src, rasterio.open(mask_path) as mask_src:
        if (src.height, src.width) != (mask_src.height, mask_src.width):
            raise ValueError(
                f"HR shape {src.shape} != mask shape {mask_src.shape}; "
                "this script assumes both live on the 60 m HR grid."
            )
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=float("nan"), compress="deflate", tiled=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            H, W = src.height, src.width
            nd = src.nodata
            for r0 in range(0, H, chunk_rows):
                r1 = min(r0 + chunk_rows, H)
                window = Window(0, r0, W, r1 - r0)
                arr = src.read(1, window=window, out_dtype=np.float32)
                if nd is not None and not np.isnan(nd):
                    arr[arr == nd] = np.nan
                arr[np.abs(arr) > SENTINEL_ABS_THRESHOLD] = np.nan
                mask = mask_src.read(1, window=window).astype(np.uint8)

                for b, mu in block_means.items():
                    sel = mask == b
                    if sel.any():
                        arr[sel] -= mu
                        n_corrected += int(sel.sum())

                dst.write(arr, 1, window=window)
                n_total += arr.size
    return n_total, n_corrected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr", default="AMF_RTP", help="HR product (default: AMF_RTP)")
    parser.add_argument(
        "--lr", default="RTP", help="LR product paired in the block-means JSON (default: RTP)"
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=DEFAULT_STATS_DIR,
        help=f"Where to find block-means JSON (default: {DEFAULT_STATS_DIR.relative_to(ROOT_FOLDER)})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override output GeoTIFF path (default: cfg.hr_normalized_product_path(--hr), i.e. the `_blockwise_unbiased.tif` sibling of the source HR).",
    )
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--force", action="store_true", help="Overwrite the output if it already exists")
    args = parser.parse_args()

    cfg = KSAAlignedConfig.default()
    hr_path = cfg.hr_source_product_path(args.hr)
    mask_path = cfg.mask_path

    stats_path = args.stats_dir / f"{args.hr}_vs_{args.lr}_block_means.json"
    hr_means = load_block_means(stats_path, side="hr")
    print("Block means to subtract: " + ", ".join(f"B{b}={mu:+.2f}" for b, mu in sorted(hr_means.items())))

    out_path = args.output_path or cfg.hr_normalized_product_path(args.hr)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} already exists. Pass --force to overwrite.")

    n_total, n_corrected = apply_correction_streamed(
        hr_path, mask_path, hr_means, out_path, chunk_rows=args.chunk_rows
    )
    print(
        f"Wrote {out_path}\n"
        f"  total pixels:     {n_total:>14,}\n"
        f"  corrected pixels: {n_corrected:>14,} ({n_corrected / n_total:.1%})"
    )


if __name__ == "__main__":
    main()

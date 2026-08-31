"""Illustrate the boundary patch-retention rule on the irregular KSA raster.

Training keeps a patch only if its valid fraction reaches the config's `min_valid_frac` (0.97). This
zooms into one boundary window of an SGS block and colours every candidate patch by that retain/reject
decision exactly as `build_aligned_block_patches` does at build time. The window is auto-selected to
straddle the data/NaN edge with a mix of patch fates, drawn on the clean non-overlapping patch grid
(132 px) — the training sampler strides denser, but the retention rule is identical either way.

Output: figures/boundary_patching/ksa_boundary_patching_B<block>[_compact].png

Run:  python experiments/plots/plot_boundary_patching.py             # 8x8 window, map axes
      python experiments/plots/plot_boundary_patching.py --compact   # wide 14x4 strip for a 2-col paper
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

from magsr import ROOT_FOLDER
from magsr.datasets import KSAAlignedConfig
from magsr.datasets.ksa_shield_aligned import build_aligned_block_patches
from magsr.viz import plot_boundary_patch_retention

OUT_DIR = ROOT_FOLDER / "figures" / "boundary_patching"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", type=int, default=2, help="SGS block to draw (default: 2).")
    p.add_argument(
        "--compact",
        action="store_true",
        help="Wide 14x4 strip, no map axes — drops straight into a two-column paper.",
    )
    args = p.parse_args()

    cfg = KSAAlignedConfig.from_yaml()
    # Zero-threshold sweep just to recover the effective (block AND LR) mask;
    # retention is applied per-patch in the plotter against the config threshold.
    spec = replace(cfg.patch_grid_spec, min_valid_frac=0.0)
    _, eff, meta = build_aligned_block_patches(args.block, spec=spec, config=cfg)

    out = OUT_DIR / f"ksa_boundary_patching_B{args.block}{'_compact' if args.compact else ''}.png"
    row0, col0, counts = plot_boundary_patch_retention(
        eff,
        meta,
        spec,
        title="",  # the LaTeX caption carries it
        out_path=out,
        crop_patches=(14, 4) if args.compact else 8,
        min_valid_frac=cfg.min_valid_frac,  # `spec`'s copy was zeroed above
        compact=args.compact,
        simple_legend=True,
    )
    kept = counts["full"] + counts["kept_nan"]
    print(
        f"Block {args.block}: window (row0={row0}, col0={col0}) "
        f"full={counts['full']} kept-with-NaN={counts['kept_nan']} "
        f"rejected={counts['rejected']} (kept {kept} total)"
    )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

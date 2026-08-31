"""
Assign cell-grid cells to train/val/test (Hedgementation-style).

Every cell from 05 (the cell-index build) is randomly assigned to a split. The
buffer strips built into the cell grid already guarantee that no two splits
share pixels, so there is no skip-zone pass: the split is pure bookkeeping over
the saved indices.

Flow:
  1. Load cells_B*.json + block_*.json pairs from patch_indices_cellgrid/.
  2. Per block: assign_cell_splits(cells, fractions, seed) — per-block
     assignment keeps each split evenly spread across the three survey
     blocks with no extra stratification code.
  3. Re-derive patch->cell membership (bucket_patches_by_cell) and route each
     cell's patches to its cell's split.
  4. Save train/val/test.json (the standard train/val/test PatchIndex shape the
     loaders expect, so downstream is unchanged) + cell_splits.json provenance +
     figures.

With --folds K the same machinery runs in cross-validation mode: per block,
cells are assigned to K patch-weighted folds; for each fold k the other K-1
folds train and fold k's cells are split evenly into val/test (~80/10/10 at
K=5). Each rotation lands in its own `{index_dir}_fold{k}/` with the usual
train/val/test.json, so the trainer just points --index-dir at one of them.

Examples:
    uv run python scripts/build_ksa_dataset/06_assign_cell_splits.py
    uv run python scripts/build_ksa_dataset/06_assign_cell_splits.py --seed 7 --val-frac 0.1 --test-frac 0.1
    uv run python scripts/build_ksa_dataset/06_assign_cell_splits.py --suffix 8 --folds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle

from magsr import ROOT_FOLDER
from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig
from magsr.datasets.patching import (
    CellGridSpec,
    PatchIndex,
    PatchWindow,
    assign_cell_splits,
    bucket_patches_by_cell,
)
from magsr.viz import load_magnetic_strided

DEFAULT_FIG_DIR = ROOT_FOLDER / "figures" / "ksa_cellgrid_splits"
SPLIT_COLORS = {"train": "#888888", "val": "#1f77b4", "test": "#2ca02c"}


def plot_cell_splits(
    mag_lo: np.ndarray,
    extent: tuple[float, float, float, float],
    cells: list[PatchWindow],
    split_map: dict[str, str],
    vmin: float,
    vmax: float,
    title: str,
    save_path: Path,
    *,
    zoom: bool = True,
) -> None:
    """HR magnetic background with cell rectangles colored by split.

    Train cells are thin gray outlines; val/test cells are filled
    translucently so the held-out areas pop. `zoom` fits the axes to the
    plotted cells (per-block view) instead of the full raster extent.
    """
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_facecolor("0.75")
    ax.imshow(
        mag_lo, extent=extent, origin="upper", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    for split in ("train", "val", "test"):  # held-out splits drawn on top
        rects = [
            Rectangle((c.left, c.bottom), c.width, c.height) for c in cells if split_map[c.name] == split
        ]
        color = SPLIT_COLORS[split]
        ax.add_collection(
            PatchCollection(
                rects,
                facecolors=to_rgba(color, 0.35) if split != "train" else "none",
                edgecolors=color,
                linewidths=1.5 if split != "train" else 0.6,
            )
        )
    ax.legend(
        handles=[Patch(facecolor=to_rgba(c, 0.35), edgecolor=c, label=s) for s, c in SPLIT_COLORS.items()],
        loc="upper right",
    )
    if zoom:
        pad = 5_000
        ax.set_xlim(min(c.left for c in cells) - pad, max(c.right for c in cells) + pad)
        ax.set_ylim(min(c.bottom for c in cells) - pad, max(c.top for c in cells) + pad)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("UTM37N easting (m)")
    ax.set_ylabel("UTM37N northing (m)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Fraction of patches held out for val (default: `cellgrid.val_frac` from datasets.yaml).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=None,
        help="Fraction of patches held out for test (default: `cellgrid.test_frac`).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Cell-permutation seed (default: `cellgrid.seed`)."
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Cross-validation mode: assign cells to this many patch-weighted folds per block "
        "and write one `{index_dir}_fold{k}` split set per fold (train = other folds, held-out "
        "fold split evenly into val/test). --val-frac/--test-frac are ignored.",
    )
    parser.add_argument(
        "--hr-product",
        default="AMF_RTP",
        help="HR product used as the figure background (default: AMF_RTP)",
    )
    parser.add_argument(
        "--display-stride",
        type=int,
        default=8,
        help="Decimation factor for the HR raster background (default: 8)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=f"Figure dir (default: {DEFAULT_FIG_DIR.relative_to(ROOT_FOLDER)} + suffix)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Variant tag matching 05's --suffix (e.g. '8' -> patch_indices_cellgrid8).",
    )
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = DEFAULT_FIG_DIR.with_name(DEFAULT_FIG_DIR.name + args.suffix)

    cfg = KSAAlignedConfig.from_yaml()
    if args.val_frac is None:
        args.val_frac = cfg.cellgrid.val_frac
    if args.test_frac is None:
        args.test_frac = cfg.cellgrid.test_frac
    if args.seed is None:
        args.seed = cfg.cellgrid.seed
    index_dir = cfg.patch_index_dir.parent / f"patch_indices_cellgrid{args.suffix}"
    fractions = {
        "train": 1.0 - args.val_frac - args.test_frac,
        "val": args.val_frac,
        "test": args.test_frac,
    }
    # Scenario "" = the plain single split; "fold{k}" = one CV rotation whose
    # outputs land in {index_dir}_fold{k}.
    scenarios = [f"fold{k}" for k in range(args.folds)] if args.folds else [""]

    cell_paths = sorted(index_dir.glob("cells_B*.json"))
    if not cell_paths:
        raise FileNotFoundError(
            f"No cell indices in {index_dir}. "
            "Run scripts/build_ksa_dataset/05_ksa_aligned_build_cell_indices.py first."
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    mag_lo, extent = load_magnetic_strided(cfg.hr_product_path(args.hr_product), stride=args.display_stride)
    finite = mag_lo[np.isfinite(mag_lo)]
    vmin, vmax = (
        (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else (-1.0, 1.0)
    )

    split_patches: dict[str, dict[str, list[PatchWindow]]] = {
        sc: {name: [] for name in fractions} for sc in scenarios
    }
    split_maps: dict[str, dict[str, str]] = {sc: {} for sc in scenarios}  # cell name -> split
    all_cells: list[PatchWindow] = []
    base_spec = product = None
    cell_patches_n = buffer_patches_n = None

    for cell_path in cell_paths:
        block = int(cell_path.stem.removeprefix("cells_B"))
        cells_idx = PatchIndex.load(cell_path)
        patches_idx = PatchIndex.load(index_dir / f"block_{block}.json")
        if base_spec is None:
            base_spec, product = patches_idx.spec, patches_idx.product
            cell_patches_n = cells_idx.extra["cell_patches"]
            buffer_patches_n = cells_idx.extra["buffer_patches"]
        spec = CellGridSpec(
            patch=patches_idx.spec,
            cell_patches=cells_idx.extra["cell_patches"],
            buffer_patches=cells_idx.extra["buffer_patches"],
        )

        # Weight assignment by per-cell patch count so val/test hold ~their
        # fraction of *patches*, not just of cells (edge cells are sparse).
        patches_by_cell = bucket_patches_by_cell(patches_idx.patches, cells_idx.patches, spec)
        weights = [len(patches_by_cell[c.name]) for c in cells_idx.patches]
        if args.folds:
            # K-way patch-weighted fold assignment; each rotation trains on the
            # other folds and splits the held-out fold's cells evenly val/test.
            fold_map = assign_cell_splits(
                cells_idx.patches,
                {sc: 1.0 / args.folds for sc in scenarios},
                seed=args.seed,
                weights=weights,
            )
            block_maps = {}
            for sc in scenarios:
                held = [c for c in cells_idx.patches if fold_map[c.name] == sc]
                vt = assign_cell_splits(
                    held,
                    {"val": 0.5, "test": 0.5},
                    seed=args.seed,
                    weights=[len(patches_by_cell[c.name]) for c in held],
                )
                block_maps[sc] = {c.name: vt.get(c.name, "train") for c in cells_idx.patches}
        else:
            block_maps = {
                "": assign_cell_splits(cells_idx.patches, fractions, seed=args.seed, weights=weights)
            }
        all_cells.extend(cells_idx.patches)

        for sc, block_map in block_maps.items():
            split_maps[sc] |= block_map

            patch_counts = {name: 0 for name in fractions}
            for name, cell_patches in patches_by_cell.items():
                split_patches[sc][block_map[name]].extend(cell_patches)
                patch_counts[block_map[name]] += len(cell_patches)

            cell_counts = {s: sum(1 for v in block_map.values() if v == s) for s in fractions}
            n_block = sum(patch_counts.values())
            tag = f" [{sc}]" if sc else ""
            print(
                f"Block {block}{tag}: {len(cells_idx)} cells, {n_block:,} patches  "
                + "  ".join(
                    f"{s}={cell_counts[s]} cells/{patch_counts[s]} patches "
                    f"({100 * patch_counts[s] / n_block:.1f}%)"
                    for s in fractions
                )
            )

            plot_cell_splits(
                mag_lo,
                extent,
                cells_idx.patches,
                block_map,
                vmin,
                vmax,
                f"Block {block}{tag} — {len(cells_idx)} cells "
                f"(val={cell_counts['val']} [blue], test={cell_counts['test']} [green])",
                args.save_dir / f"block_{block}_cell_splits{f'_{sc}' if sc else ''}.png",
            )

    for sc in scenarios:
        out_dir = index_dir.with_name(index_dir.name + f"_{sc}") if sc else index_dir
        provenance: dict = {"seed": args.seed}
        if args.folds:
            provenance |= {"n_folds": args.folds, "fold": int(sc.removeprefix("fold"))}
        else:
            provenance |= {"val_frac": args.val_frac, "test_frac": args.test_frac}

        for name, patches in split_patches[sc].items():
            out = PatchIndex(
                spec=base_spec,
                product=product,
                patches=patches,
                extra={
                    "dataset": "ksa_aligned",
                    "split": name,
                    **provenance,
                    "cell_patches": cell_patches_n,
                    "buffer_patches": buffer_patches_n,
                },
            )
            out.save(out_dir / f"{name}.json")
            print(f"Saved {out_dir / f'{name}.json'}  n={len(patches):,}")

        splits_path = out_dir / "cell_splits.json"
        splits_path.write_text(json.dumps({**provenance, "cells": split_maps[sc]}, indent=2))
        print(f"Saved cell->split provenance -> {splits_path}")

        plot_cell_splits(
            mag_lo,
            extent,
            all_cells,
            split_maps[sc],
            vmin,
            vmax,
            f"KSA aligned — cell-grid splits across all blocks{f' [{sc}]' if sc else ''}",
            args.save_dir / f"overview{f'_{sc}' if sc else ''}.png",
            zoom=False,
        )
    print(f"Saved figures to {args.save_dir}")


if __name__ == "__main__":
    main()

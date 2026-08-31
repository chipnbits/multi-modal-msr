"""Visualize the K-fold cell-grid partition (one color per fold) + patch-count table.

Companion to 06_assign_cell_splits.py --folds: that script writes one
{index_dir}_fold{k} split set per rotation; this reads them back and renders
every cell colored by the fold it is held out in (val/test), i.e. the fold it
"belongs" to.

Also prints and saves a per-fold (and per-block) patch-count table.

Example:
    uv run python experiments/plots/plot_cv_folds.py --suffix 8 --folds 5
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle

from magsr import ROOT_FOLDER
from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig
from magsr.datasets.patching import CellGridSpec, PatchIndex, PatchWindow, bucket_patches_by_cell
from magsr.viz import load_magnetic_strided

DEFAULT_FIG_DIR = ROOT_FOLDER / "figures" / "ksa_cellgrid_splits"
# 5 distinct qualitative colors (matplotlib tab10); extend if --folds > len.
FOLD_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_cell_folds(
    mag_lo: np.ndarray,
    extent: tuple[float, float, float, float],
    draw: list[tuple[PatchWindow, int]],
    n_folds: int,
    vmin: float,
    vmax: float,
    save_path: Path,
    *,
    title: str | None = None,
    fontsize: float = 15.4,
    figsize: float = 7.0,
) -> None:
    """HR magnetic background with every physical lattice square filled by its CV fold.

    `draw` is one entry per physical square: (cell, fold). Cross-block coincident
    squares are collapsed upstream to a single entry so none is painted twice.

    `title=None` omits the title; `fontsize` / `figsize` scale the text large
    relative to a small square canvas so it stays legible as a half-page column
    figure (labels/ticks/legend all keyed off `fontsize`).
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_facecolor("0.75")
    ax.imshow(
        mag_lo, extent=extent, origin="upper", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    for k in range(n_folds):
        color = FOLD_COLORS[k]
        rects = [Rectangle((c.left, c.bottom), c.width, c.height) for c, f in draw if f == k]
        ax.add_collection(
            PatchCollection(rects, facecolors=to_rgba(color, 0.45), edgecolors=color, linewidths=2.0)
        )
    # Legend in the empty gray lower-left (outside the AOI), so it covers no cells
    # and the figure stays compact and square.
    ax.legend(
        handles=[
            Patch(facecolor=to_rgba(FOLD_COLORS[k], 0.45), edgecolor=FOLD_COLORS[k], label=f"fold {k}")
            for k in range(n_folds)
        ],
        loc="lower left",
        title="CV fold",
        fontsize=fontsize * 0.9,
        title_fontsize=fontsize,
        borderpad=0.4,
        labelspacing=0.4,
        handlelength=1.3,
        handletextpad=0.5,
        framealpha=0.92,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("UTM37N easting (m)", fontsize=fontsize)
    ax.set_ylabel("UTM37N northing (m)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize * 0.85)
    ax.xaxis.get_offset_text().set_fontsize(fontsize * 0.8)
    ax.yaxis.get_offset_text().set_fontsize(fontsize * 0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="8", help="Cell-grid variant tag (e.g. '8').")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds to read/plot.")
    parser.add_argument("--hr-product", default="AMF_RTP", help="HR product used as the figure background.")
    parser.add_argument("--display-stride", type=int, default=8, help="Decimation of the HR background.")
    parser.add_argument(
        "--no-title", action="store_true", help="Omit the in-figure title (LaTeX caption carries it)."
    )
    parser.add_argument(
        "--fontsize", type=float, default=15.4, help="Base font size for labels/ticks/legend."
    )
    parser.add_argument(
        "--figsize",
        type=float,
        default=7.0,
        help="Square figure side in inches (smaller => larger relative text).",
    )
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = DEFAULT_FIG_DIR.with_name(DEFAULT_FIG_DIR.name + args.suffix)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    cfg = KSAAlignedConfig.from_yaml()
    index_dir = cfg.patch_index_dir.parent / f"patch_indices_cellgrid{args.suffix}"

    # Fold membership: a cell belongs to fold k iff it is held out (val/test) in
    # that fold's saved split. Read the on-disk provenance so the figure matches
    # exactly what the trainer consumes (no re-derivation of the assignment).
    fold_of: dict[str, int] = {}
    vt_of: dict[str, str] = {}  # cell name -> 'val'/'test' in its held-out fold
    for k in range(args.folds):
        fold_dir = index_dir.with_name(index_dir.name + f"_fold{k}")
        cell_map = json.loads((fold_dir / "cell_splits.json").read_text())["cells"]
        for name, split in cell_map.items():
            if split in ("val", "test"):
                if name in fold_of:
                    raise ValueError(f"cell {name} held out in folds {fold_of[name]} and {k}")
                fold_of[name] = k
                vt_of[name] = split

    # Per-cell patch counts (and block) from the base cell-grid index.
    cell_paths = sorted(index_dir.glob("cells_B*.json"))
    if not cell_paths:
        raise FileNotFoundError(f"No cell indices in {index_dir}.")
    all_cells: list[PatchWindow] = []
    patches_per_cell: dict[str, int] = {}
    cell_block: dict[str, int] = {}
    cell_patches_n = 0  # cell side length in base patches (for the title)
    for cell_path in cell_paths:
        block = int(cell_path.stem.removeprefix("cells_B"))
        cells_idx = PatchIndex.load(cell_path)
        patches_idx = PatchIndex.load(index_dir / f"block_{block}.json")
        cell_patches_n = cells_idx.extra["cell_patches"]
        spec = CellGridSpec(
            patch=patches_idx.spec,
            cell_patches=cells_idx.extra["cell_patches"],
            buffer_patches=cells_idx.extra["buffer_patches"],
        )
        by_cell = bucket_patches_by_cell(patches_idx.patches, cells_idx.patches, spec)
        for c in cells_idx.patches:
            patches_per_cell[c.name] = len(by_cell.get(c.name, []))
            cell_block[c.name] = block
        all_cells.extend(cells_idx.patches)

    missing = {c.name for c in all_cells} - set(fold_of)
    if missing:
        raise ValueError(f"{len(missing)} cells have no fold assignment (e.g. {next(iter(missing))})")

    blocks = sorted(set(cell_block.values()))

    # The three blocks share one global lattice, so a seam square can appear once
    # per block (identical geometry, block-specific names). Group cells by lattice
    # slot and resolve each physical square to a single overriding fold = the fold
    # of the block that owns the most patches there. Every patch in the square is
    # then counted under that fold, so no square is double-counted across folds.
    slots: dict[tuple[int, int], list[PatchWindow]] = defaultdict(list)
    for c in all_cells:
        slots[(c.row_px, c.col_px)].append(c)
    n_shared = sum(1 for members in slots.values() if len(members) > 1)

    fold_cells = {k: 0 for k in range(args.folds)}
    fold_patches = {k: 0 for k in range(args.folds)}
    fold_val = {k: 0 for k in range(args.folds)}
    fold_test = {k: 0 for k in range(args.folds)}
    fold_block_patches = {k: {b: 0 for b in blocks} for k in range(args.folds)}
    draw: list[tuple[PatchWindow, int]] = []
    for members in slots.values():
        best = max(members, key=lambda c: patches_per_cell[c.name])
        k = fold_of[best.name]  # overriding fold for the whole square
        draw.append((best, k))
        fold_cells[k] += 1
        # When fold k is held out, this square goes wholly to val or test,
        # following the majority-block cell's 50/50 assignment in that fold.
        bucket = fold_val if vt_of[best.name] == "val" else fold_test
        for c in members:  # all blocks' patches in the square land in fold k
            n = patches_per_cell[c.name]
            fold_patches[k] += n
            fold_block_patches[k][cell_block[c.name]] += n
            bucket[k] += n
    total_patches = sum(fold_patches.values())
    tot_val, tot_test = sum(fold_val.values()), sum(fold_test.values())

    # --- print table ---
    blk_hdr = "  ".join(f"B{b:>6}" for b in blocks)
    print(f"\n{args.folds}-fold cell-grid partition (cellgrid{args.suffix}), override dedup")
    if n_shared:
        print(
            f"  {len(slots)} physical squares ({len(all_cells)} split-unit cells); {n_shared} shared, overriding fold used"
        )
    hdr = f"{'fold':>4}  {'cells':>5}  {'patches':>8}  {'%':>5}  {'val':>6}  {'test':>6}  {'val%':>5}   {blk_hdr}"
    print(hdr)
    for k in range(args.folds):
        blk = "  ".join(f"{fold_block_patches[k][b]:>7}" for b in blocks)
        pct = 100.0 * fold_patches[k] / total_patches
        valpct = 100.0 * fold_val[k] / max(fold_patches[k], 1)
        print(
            f"{k:>4}  {fold_cells[k]:>5}  {fold_patches[k]:>8}  {pct:>5.1f}  "
            f"{fold_val[k]:>6}  {fold_test[k]:>6}  {valpct:>5.1f}   {blk}"
        )
    tot_blk = "  ".join(f"{sum(fold_block_patches[k][b] for k in range(args.folds)):>7}" for b in blocks)
    print(
        f"{'all':>4}  {len(slots):>5}  {total_patches:>8}  {100.0:>5.1f}  "
        f"{tot_val:>6}  {tot_test:>6}  {100.0 * tot_val / total_patches:>5.1f}   {tot_blk}"
    )

    # --- save CSV ---
    csv_path = args.save_dir / f"fold_patch_counts{args.suffix}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["fold", "cells", "patches", "pct", "val", "test", "val_pct"]
            + [f"B{b}_patches" for b in blocks]
        )
        for k in range(args.folds):
            w.writerow(
                [
                    k,
                    fold_cells[k],
                    fold_patches[k],
                    round(100.0 * fold_patches[k] / total_patches, 2),
                    fold_val[k],
                    fold_test[k],
                    round(100.0 * fold_val[k] / max(fold_patches[k], 1), 2),
                ]
                + [fold_block_patches[k][b] for b in blocks]
            )
        w.writerow(
            [
                "all",
                len(slots),
                total_patches,
                100.0,
                tot_val,
                tot_test,
                round(100.0 * tot_val / total_patches, 2),
            ]
            + [sum(fold_block_patches[k][b] for k in range(args.folds)) for b in blocks]
        )
    print(f"\nSaved table -> {csv_path}")

    # --- figure ---
    mag_lo, extent = load_magnetic_strided(cfg.hr_product_path(args.hr_product), stride=args.display_stride)
    finite = mag_lo[np.isfinite(mag_lo)]
    vmin, vmax = (
        (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else (-1.0, 1.0)
    )
    fig_path = args.save_dir / f"folds_overview{args.suffix}.png"
    title = (
        None
        if args.no_title
        else (f"KSA aligned — {cell_patches_n}×{cell_patches_n} cell grid, {args.folds}-fold CV partition")
    )
    plot_cell_folds(
        mag_lo,
        extent,
        draw,
        args.folds,
        vmin,
        vmax,
        fig_path,
        title=title,
        fontsize=args.fontsize,
        figsize=args.figsize,
    )
    print(f"Saved figure  -> {fig_path}")


if __name__ == "__main__":
    main()

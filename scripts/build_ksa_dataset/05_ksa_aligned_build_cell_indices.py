"""
Build Hedgementation-style cell-grid indices for the KSA aligned dataset.

Each of the three SGS survey blocks is one component; this script cuts each
block mask into buffered cells (square super-patches of --cell-patches x
--cell-patches base patches, separated by a --buffer-patches-wide dropped strip),
then keeps strided base patches that fall entirely inside one cell
interior. Cells are the split unit consumed by 06_assign_cell_splits.py.

Per block in `cfg.blocks` (or all present in the mask):
  1. Block mask + strided base patches via build_aligned_block_patches
  2. Cells: one extra sliding_window_patches sweep with spec.as_cell_sweep().
     The pitch grid is aligned to the full raster, so the three blocks
     share one cell lattice and their cells can never partially overlap.
  3. Bucket patches into cells (bucket_patches_by_cell); patches straddling a
     buffer strip drop out here.
  4. Save cells_B{block}.json + block_{block}.json (+ overlay figures).

Outputs go to patch_indices_cellgrid{suffix}/; downstream (06_assign_cell_splits.py,
then the trainers via --index-dir) points at that directory.

Reference:
    Senyard et al. (2026). "Hedgementation = Hedgerow Segmentation: A Remote
    Sensing Benchmark." ICLR 2026 ML4RS Workshop.
    https://openreview.net/forum?id=mOMTBBgq5n
"""

import argparse
from dataclasses import replace
from pathlib import Path

from magsr import ROOT_FOLDER
from magsr.datasets import (
    KSAAlignedConfig,
    PatchIndex,
    build_aligned_block_patches,
    iter_aligned_blocks,
)
from magsr.datasets.patching import bucket_patches_by_cell, sliding_window_patches
from magsr.viz import plot_patches_over_mask

CELLGRID_FIGURES = ROOT_FOLDER / "figures" / "ksa_aligned_cellgrid"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--blocks", type=int, nargs="+", default=None)
    parser.add_argument(
        "--cell-patches",
        type=int,
        default=None,
        help="Cell side, in base patches (default: `cellgrid.cell_patches` from datasets.yaml).",
    )
    parser.add_argument(
        "--buffer-patches",
        type=int,
        default=None,
        help="Dropped strip between cells, in base patches (default: `cellgrid.buffer_patches`).",
    )
    parser.add_argument(
        "--min-cell-valid-frac",
        type=float,
        default=None,
        help="Drop cells with less mask coverage than this (default: `cellgrid.min_cell_valid_frac`).",
    )
    parser.add_argument(
        "--stride-px",
        type=int,
        default=None,
        help="Base-patch stride in 60 m pixels (default: the spec's `stride_px`). Cell geometry "
        "(cell_px/pitch) depends only on patch_px x cell_patches, so a smaller stride just packs "
        "more overlapping patches into the same cells — denser sampling, identical cell lattice.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--suffix",
        default="",
        help="Variant tag appended to the output dirs (e.g. '8' -> patch_indices_cellgrid8), "
        "so differently-sized cell grids coexist.",
    )
    # TODO: forward 04's remaining base-spec overrides (--patch-px /
    # --min-valid-frac / --lr-validity-product) if ad-hoc sweeps are needed here.
    args = parser.parse_args()

    cfg = KSAAlignedConfig.from_yaml(args.config)
    if args.blocks is not None:
        cfg = replace(cfg, blocks=tuple(args.blocks))
    spec = cfg.cell_grid_spec
    spec_overrides = {
        k: v
        for k, v in (
            ("cell_patches", args.cell_patches),
            ("buffer_patches", args.buffer_patches),
            ("min_cell_valid_frac", args.min_cell_valid_frac),
        )
        if v is not None
    }
    if spec_overrides:
        spec = replace(spec, **spec_overrides)
    if args.stride_px is not None:
        spec = replace(spec, patch=replace(spec.patch, stride_px=args.stride_px))
    out_dir = cfg.patch_index_dir.parent / f"patch_indices_cellgrid{args.suffix}"
    fig_dir = CELLGRID_FIGURES.with_name(CELLGRID_FIGURES.name + args.suffix)

    blocks = list(cfg.blocks) if cfg.blocks else iter_aligned_blocks(cfg)
    print(
        f"Blocks to build: {blocks} (cell={spec.cell_px}px, pitch={spec.pitch_px}px, "
        f"patch={spec.patch.patch_px}px, stride={spec.patch.stride_px}px)",
        flush=True,
    )

    for block in blocks:
        print(f"\n=== Block {block} ===", flush=True)
        cells_path = out_dir / f"cells_B{block}.json"
        patches_path = out_dir / f"block_{block}.json"
        if cells_path.exists() and patches_path.exists() and not args.force and args.skip_plots:
            print(f"  {cells_path} exists — skipping (use --force to rebuild)")
            continue

        # 1. Large component -> mask + strided base patches, exactly as 04.
        patches, mask, meta = build_aligned_block_patches(block, spec=spec.patch, config=cfg)

        # 2. Cut the component into buffered cells: just a coarser sweep.
        # allow_partial keeps trailing lattice cells that overhang the raster
        # edge — otherwise every patch in the last (sub-cell-sized) band of a
        # block is silently dropped.
        cells = sliding_window_patches(
            mask, meta, spec.as_cell_sweep(), source_id=f"ksa_aligned/B{block}", allow_partial=True
        )

        # 3. Strided regular patches within each cell; buffer-strip patches drop out.
        patches_by_cell = bucket_patches_by_cell(patches, cells, spec)

        # Most lattice positions are empty for any one block (all blocks share
        # one raster), so keep only cells that own at least one patch — 06's
        # split fractions should count populated cells only.
        kept_cells = [c for c in cells if c.name in patches_by_cell]
        kept_patches = [t for ts in patches_by_cell.values() for t in ts]
        print(f"  cells: {len(kept_cells)} kept / {len(cells)} swept (empty dropped)")
        print(
            f"  patches: {len(kept_patches)} kept / {len(patches)} swept "
            f"({len(patches) - len(kept_patches)} on buffer strips or in dropped cells)"
        )

        extra = {
            "dataset": "ksa_aligned",
            "grid": "60m",
            "block": block,
            "lr_validity_product": cfg.lr_validity_product,
            "cell_patches": spec.cell_patches,
            "buffer_patches": spec.buffer_patches,
        }
        PatchIndex(
            spec=spec.as_cell_sweep(),
            product="AMF_RTP",
            patches=kept_cells,
            extra={**extra, "role": "cells"},
        ).save(cells_path)
        print(f"  saved cell index -> {cells_path} ({len(kept_cells)} cells)", flush=True)
        PatchIndex(
            spec=spec.patch,
            product="AMF_RTP",
            patches=kept_patches,
            extra={**extra, "role": "patches"},
        ).save(patches_path)
        print(f"  saved patch index -> {patches_path} ({len(kept_patches)} patches)", flush=True)

        if not args.skip_plots:
            # A cell is a PatchWindow and as_cell_sweep() a PatchGridSpec, so
            # the patch-overlay plot renders cells unchanged.
            plot_patches_over_mask(
                mask,
                meta,
                kept_cells,
                spec.as_cell_sweep(),
                title=f"KSA aligned B{block} — cell grid",
                out_path=fig_dir / f"cells_B{block}.png",
                color_by="grid_cycle",
            )
            plot_patches_over_mask(
                mask,
                meta,
                kept_patches,
                spec.patch,
                title=f"KSA aligned B{block} — patches within cells",
                out_path=fig_dir / f"patches_B{block}.png",
                color_by="grid_cycle",
            )
            print(f"  saved figures -> {fig_dir}/{{cells,patches}}_B{block}.png", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()

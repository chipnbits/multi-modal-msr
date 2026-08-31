"""
Route a (re-patched) cell-grid build into train/val/test using an EXISTING
cell->split assignment, instead of re-deriving the split with 06_assign_cell_splits.

Why: cell *geometry* (cell_px/pitch) depends only on patch_px x cell_patches, so
rebuilding the cell index (05) at a finer --stride-px yields the exact same cell lattice (same
cell names) but far more overlapping base patches per cell. 06's split, however,
is patch-count weighted, so re-running it at the new stride would shift which
cells land in val/test. To keep a dense rebuild directly comparable to an existing
benchmark fold, this script reuses that fold's saved `cell_splits.json` map and
only re-buckets the denser patches.

Flow:
  1. Load cells_B*.json + block_*.json from --index-dir (the dense 05 build).
  2. Re-derive patch->cell membership (bucket_patches_by_cell).
  3. Route each cell's patches to the split named in --cell-splits' map.
  4. Save train/val/test.json (same train/val/test shape 06 emits) + a provenance
     copy into --out-dir.

Example (stride-12 rebuild inheriting the cellgrid8 fold-3 split):
    uv run python scripts/build_ksa_dataset/06b_apply_cell_splits.py \
        --index-dir data/processed/ksa_aligned/patch_indices_cellgrid8s12 \
        --cell-splits data/processed/ksa_aligned/patch_indices_cellgrid8_fold3/cell_splits.json \
        --out-dir data/processed/ksa_aligned/patch_indices_cellgrid8s12_fold3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from magsr.datasets.patching import (
    CellGridSpec,
    PatchIndex,
    bucket_patches_by_cell,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Dense 05 build dir holding cells_B*.json + block_*.json.",
    )
    parser.add_argument(
        "--cell-splits",
        type=Path,
        required=True,
        help="Existing cell_splits.json (its `cells` map names every cell's split).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination for train/val/test.json + cell_splits.json.",
    )
    args = parser.parse_args()

    provenance = json.loads(args.cell_splits.read_text())
    split_map: dict[str, str] = provenance["cells"]
    split_names = sorted(set(split_map.values()))

    cell_paths = sorted(args.index_dir.glob("cells_B*.json"))
    if not cell_paths:
        raise FileNotFoundError(
            f"No cell indices in {args.index_dir}. "
            "Run scripts/build_ksa_dataset/05_ksa_aligned_build_cell_indices.py first."
        )

    split_patches: dict[str, list] = {name: [] for name in split_names}
    base_spec = product = None
    cell_patches_n = buffer_patches_n = None

    for cell_path in cell_paths:
        block = int(cell_path.stem.removeprefix("cells_B"))
        cells_idx = PatchIndex.load(cell_path)
        patches_idx = PatchIndex.load(args.index_dir / f"block_{block}.json")
        if base_spec is None:
            base_spec, product = patches_idx.spec, patches_idx.product
            cell_patches_n = cells_idx.extra["cell_patches"]
            buffer_patches_n = cells_idx.extra["buffer_patches"]
        spec = CellGridSpec(
            patch=patches_idx.spec,
            cell_patches=cells_idx.extra["cell_patches"],
            buffer_patches=cells_idx.extra["buffer_patches"],
        )

        patches_by_cell = bucket_patches_by_cell(patches_idx.patches, cells_idx.patches, spec)
        counts = {name: 0 for name in split_names}
        for name, cell_patches in patches_by_cell.items():
            if name not in split_map:
                raise KeyError(
                    f"Cell {name} not in {args.cell_splits} — the dense build's cell lattice "
                    "does not match the saved split (different cell_patches/buffer_patches?)."
                )
            split = split_map[name]
            split_patches[split].extend(cell_patches)
            counts[split] += len(cell_patches)
        n_block = sum(counts.values())
        print(
            f"Block {block}: {len(cells_idx)} cells, {n_block:,} patches  "
            + "  ".join(
                f"{s}={counts[s]:,} ({100 * counts[s] / max(n_block, 1):.1f}%)" for s in split_names
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, patches in split_patches.items():
        out = PatchIndex(
            spec=base_spec,
            product=product,
            patches=patches,
            extra={
                "dataset": "ksa_aligned",
                "split": name,
                **{k: v for k, v in provenance.items() if k != "cells"},
                "cell_patches": cell_patches_n,
                "buffer_patches": buffer_patches_n,
                "inherited_split_from": str(args.cell_splits),
            },
        )
        out.save(args.out_dir / f"{name}.json")
        print(f"Saved {args.out_dir / f'{name}.json'}  n={len(patches):,}")

    (args.out_dir / "cell_splits.json").write_text(json.dumps(provenance, indent=2))
    print(f"Copied cell->split provenance -> {args.out_dir / 'cell_splits.json'}")


if __name__ == "__main__":
    main()

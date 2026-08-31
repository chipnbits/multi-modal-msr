"""
Re-patch the EXISTING cellgrid8 cells with a LARGER internal patch size, for a
context-window study. The cell lattice (split unit) is kept to cellgrid8 —
cell interior 1056 px, pitch 1188 px (= 8 x the 132 px / 44 px-LR base)

Example (HR 198 = LR 66, stride 12):
    uv run python scripts/build_ksa_dataset/05b_repatch_fixed_cells.py \
        --patch-px 198 --stride-px 12 \
        --cell-index-dir data/processed/ksa_aligned/patch_indices_cellgrid8 \
        --cell-splits   data/processed/ksa_aligned/patch_indices_cellgrid8_fold3/cell_splits.json \
        --out-dir       data/processed/ksa_aligned/patch_indices_cellgrid8s12_p198_fold3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from magsr.datasets import KSAAlignedConfig, PatchIndex, build_aligned_block_patches
from magsr.datasets.patching import PatchGridSpec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--patch-px", type=int, required=True, help="HR patch side (must be multiple of lr_scale=3)."
    )
    p.add_argument("--stride-px", type=int, default=12, help="HR patch stride (multiple of 3).")
    p.add_argument(
        "--cell-index-dir", type=Path, required=True, help="Holds cells_B*.json (1056/1188 lattice)."
    )
    p.add_argument("--cell-splits", type=Path, required=True, help="fold cell_splits.json (name -> split).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--blocks", type=int, nargs="+", default=None)
    args = p.parse_args()

    cfg = KSAAlignedConfig.from_yaml()
    blocks = args.blocks or list(cfg.blocks)
    spec = PatchGridSpec(
        patch_px=args.patch_px, stride_px=args.stride_px, min_valid_frac=cfg.min_valid_frac
    )

    split_map: dict[str, str] = json.loads(args.cell_splits.read_text())["cells"]
    split_names = sorted(set(split_map.values()))
    split_patches: dict[str, list] = {s: [] for s in split_names}
    base_product = None

    for block in blocks:
        patches, _mask, _meta = build_aligned_block_patches(block, spec=spec, config=cfg)
        cells_idx = PatchIndex.load(args.cell_index_dir / f"cells_B{block}.json")
        if base_product is None:
            base_product = cells_idx.product
        cell_px = cells_idx.spec.patch_px  # 1056
        pitch = cells_idx.spec.stride_px  # 1188
        # cell (row_idx, col_idx) -> cell window (for its name)
        by_idx = {(c.row_px // pitch, c.col_px // pitch): c for c in cells_idx.patches}

        counts = {s: 0 for s in split_names}
        dropped_fit = dropped_cell = dropped_split = 0
        for t in patches:
            # Fit test against the FIXED cell interior using THIS patch's size.
            if (t.row_px % pitch) + args.patch_px > cell_px or (t.col_px % pitch) + args.patch_px > cell_px:
                dropped_fit += 1
                continue
            cell = by_idx.get((t.row_px // pitch, t.col_px // pitch))
            if cell is None:
                dropped_cell += 1
                continue
            split = split_map.get(cell.name)
            if split is None:
                dropped_split += 1
                continue
            split_patches[split].append(t)
            counts[split] += 1
        kept = sum(counts.values())
        print(
            f"Block {block}: swept {len(patches):,} -> kept {kept:,}  "
            + "  ".join(f"{s}={counts[s]:,}" for s in split_names)
            + f"  (drop: buffer/overhang {dropped_fit:,}, empty-cell {dropped_cell:,}, no-split {dropped_split:,})",
            flush=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {k: v for k, v in json.loads(args.cell_splits.read_text()).items() if k != "cells"}
    for name, plist in split_patches.items():
        PatchIndex(
            spec=spec,
            product=base_product or "AMF_RTP",
            patches=plist,
            extra={
                "dataset": "ksa_aligned",
                "split": name,
                **provenance,
                "patch_px": args.patch_px,
                "stride_px": args.stride_px,
                "cell_lattice": "cellgrid8 (cell_px=1056, pitch=1188)",
                "inherited_split_from": str(args.cell_splits),
            },
        ).save(args.out_dir / f"{name}.json")
        print(f"Saved {args.out_dir / f'{name}.json'}  n={len(plist):,}")
    (args.out_dir / "cell_splits.json").write_text(args.cell_splits.read_text())


if __name__ == "__main__":
    main()

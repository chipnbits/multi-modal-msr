#!/usr/bin/env bash
# Build the dense stride-12 cell-grid indices for ALL 5 CV folds (cellgrid8s12_fold{0..4}),
# reproducibly. Two stages:
#   1. 05 — enumerate the GLOBAL dense base-patch index (stride 12) on the same 8x8 cell lattice
#      (cell_patches=8, buffer=1) as the standard cellgrid8 benchmark → patch_indices_cellgrid8s12.
#      Cell geometry depends only on patch_px x cell_patches, so a 12px stride packs ~22x more
#      overlapping patches into the IDENTICAL cells (same cell names) — denser sampling, same lattice.
#   2. 06b — for each fold, re-bucket those dense patches using that fold's EXISTING cellgrid8
#      cell->split map (cell_splits.json), so the dense folds inherit the exact benchmark splits
#      (same held-out test cells) → patch_indices_cellgrid8s12_fold{0..4}.
#
# Idempotent: skips the global build if already present; re-running 06b just overwrites the folds.
#
# Usage: bash scripts/build_ksa_dataset/build_dense_s12_cv_folds.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=data/processed/ksa_aligned
GLOBAL="${ROOT}/patch_indices_cellgrid8s12"

# --- stage 1: global dense base-patch index (stride 12, 8x8 cells) ---
if [ ! -d "${GLOBAL}" ]; then
    echo "=== 05: building global dense index ${GLOBAL} (stride 12) ==="
    uv run python scripts/build_ksa_dataset/05_ksa_aligned_build_cell_indices.py \
        --suffix 8s12 --cell-patches 8 --buffer-patches 1 --stride-px 12 --blocks 1 2 3
else
    echo "=== 05: ${GLOBAL} already exists — skipping global build ==="
fi

# --- stage 2: per-fold split routing (inherit the cellgrid8 fold cell->split maps) ---
for fold in 0 1 2 3 4; do
    echo "=== 06b: dense fold ${fold} ==="
    uv run python scripts/build_ksa_dataset/06b_apply_cell_splits.py \
        --index-dir "${GLOBAL}" \
        --cell-splits "${ROOT}/patch_indices_cellgrid8_fold${fold}/cell_splits.json" \
        --out-dir "${ROOT}/patch_indices_cellgrid8s12_fold${fold}"
done

echo "=== done — dense s12 fold indices: ${ROOT}/patch_indices_cellgrid8s12_fold{0..4} ==="

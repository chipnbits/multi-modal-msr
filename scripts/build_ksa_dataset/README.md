# Building the KSA Shield dataset from the pre-snapped grid

Preprocessing pipeline for the **KSA Shield aligned** dataset: starting from the pre-snapped grid (EPSG:32637) pointed to by
`MAGSR_KSA_ALIGNED_ROOT` (see the repo `.env.example`), it produces the normalized HR raster, per-block normalization stats, and cell-grid train/val/test patch indices.

To build that snapped grid in the first place — SGS portal download → sorted tiles → master grid → snapped products — see
[`00_ksa_preprocessing/README_ksa_data.md`](00_ksa_preprocessing/README_ksa_data.md).

Shared parameters for defining the dataset are in the `ksa_aligned` section of [`configs/datasets.yaml`](../../configs/datasets.yaml); every script reads them through `KSAAlignedConfig` and exposes CLI flags to override per-run.

## Pipeline order

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 01 | `01_per_block_histograms.py` | source HR + LR rasters, 60 m block mask | per-block mean stats JSON (`processed/ksa_aligned/block_means/`) + histogram figures |
| 02 | `02_inspect_lr_hr_bias.py` | same + 01's stats | LR−HR bias figures — **diagnostic, optional** |
| 03 | `03_apply_block_normalization.py` | source HR + 01's block means | zero-mean HR raster (`*_blockwise_unbiased.tif`) |
| 04 | `04_compute_ksa_aligned_normalization.py` | zero-mean HR (from 03) | `normalization.json` (per-block + global vmin/vmax) the loader reads |
| 05 | `05_ksa_aligned_build_cell_indices.py` | block mask | cell-grid indices `patch_indices_cellgrid{suffix}/` (`cells_B*.json` + `block_*.json`) |
| 06 | `06_assign_cell_splits.py` | 05's cell indices | `train/val/test.json` + `cell_splits.json` provenance (+ split figures) |

```bash
uv run python scripts/build_ksa_dataset/01_per_block_histograms.py
uv run python scripts/build_ksa_dataset/03_apply_block_normalization.py
uv run python scripts/build_ksa_dataset/04_compute_ksa_aligned_normalization.py
uv run python scripts/build_ksa_dataset/05_ksa_aligned_build_cell_indices.py --suffix 8
uv run python scripts/build_ksa_dataset/06_assign_cell_splits.py --suffix 8
```

`--suffix 8` names the output `patch_indices_cellgrid8/` — the `8` is just a tag
(it matches the 8×8 `cell_patches` default) and is the name every downstream
example assumes. The trainer then points `--index-dir` at
`patch_indices_cellgrid8/` (or a fold dir — see below).

### Where things land

Stage 04's `normalization.json` goes under `patch_indices/` (not the cellgrid
dir); block means under `block_means/`; the zero-mean HR raster is written back
into the dataset root (`02_snap_owned/.../*_blockwise_unbiased.tif`, so the root
must be writable); split figures under `figures/ksa_cellgrid_splits{suffix}/`.

## The cell grid, and why it splits cleanly

A **cell** is a square super-patch of `cell_patches × cell_patches` base patches
(default 8×8), and adjacent cells are separated by a `buffer_patches`-wide dropped
strip (default 1 patch). At the 132 px base patch this is a 1056 px cell interior
on an 1188 px pitch. The lattice is aligned to the full raster, so the three SGS
blocks share one grid and their cells never partially overlap. Because whole cells
are the split unit and neighbours are separated by a dropped buffer, **no two
splits can share pixels** — that is the point of the scheme, and it removes the
need for any post-hoc skip-zone pass. Base patches straddling a buffer strip are
simply dropped when patches are bucketed into cells.

## Variants

- **Cross-validation:** build the cells once, then assign K folds:
  ```bash
  uv run python scripts/build_ksa_dataset/05_ksa_aligned_build_cell_indices.py --suffix 8
  uv run python scripts/build_ksa_dataset/06_assign_cell_splits.py --suffix 8 --folds 5
  ```
  writes one `patch_indices_cellgrid8_fold{k}/` split set per rotation: fold `k` is
  held out (split evenly ~50/50 into val/test using `--seed`) and the other K−1
  folds are train. In fold mode `--val-frac/--test-frac` are ignored (the fold
  rotation defines the sizes).

- **Denser strides / larger patches — reuse an existing split (`06b`).** Cell
  *geometry* depends only on `patch_px × cell_patches`, so rebuilding the cell
  index (05) at a finer `--stride-px` gives the exact same cell lattice with more
  overlapping patches per cell. But 06's assignment is *patch-count weighted*, so
  re-running 06 at the new stride would move which cells land in val/test. To keep
  a dense rebuild directly comparable to an existing benchmark fold, run 06b
  instead — it reuses that fold's saved `cell_splits.json` and only re-buckets the
  denser patches:
  ```bash
  uv run python scripts/build_ksa_dataset/05_ksa_aligned_build_cell_indices.py \
      --suffix 8s12 --stride-px 12
  uv run python scripts/build_ksa_dataset/06b_apply_cell_splits.py \
      --index-dir  data/processed/ksa_aligned/patch_indices_cellgrid8s12 \
      --cell-splits data/processed/ksa_aligned/patch_indices_cellgrid8_fold3/cell_splits.json \
      --out-dir    data/processed/ksa_aligned/patch_indices_cellgrid8s12_fold3
  ```
  **Rule of thumb:** use 06 to *create* a split, 06b to *inherit* one whenever you
  change stride/patch size and must stay comparable to an existing fold.

- **`build_dense_s12_cv_folds.sh`** chains a dense 05 build + 06b for all 5 folds.
  It requires the base folds to exist first, i.e. run `05 --suffix 8` and
  `06 --suffix 8 --folds 5` (above) before it, since each fold's
  `cell_splits.json` is its input.

- **`05b_repatch_fixed_cells.py`**: re-patch existing cells with a larger internal
  patch size (context-window study), keeping the cell lattice fixed.

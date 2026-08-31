# experiments/

Training, evaluation, and figure scripts for the MSR study. Everything here
imports the library (`magsr`); nothing in `src/` depends on this folder.

**Reproducing the paper tables?** Start at
[`ksa_ablations/README.md`](ksa_ablations/README.md) — the self-contained
reproduction package: the full 62-run sweep from `sweep_configs/*.yaml`,
scored into one CSV that renders every table.

## Trainers

| Script | Task |
|---|---|
| `ksa_aligned_rdn_train.py` | RDN++ ×3 on the KSA aligned dataset (the canonical trainer; every KSA RDN ablation run launches through it) |
| `ksa_aligned_unet_train.py` | Attention U-Net ×3 on KSA (bicubic + learned corrector) |
| `wa_dataset/wa_rdn.py` | RDN++ ×4 on WA Goldfields (Smith et al. 2022 reproduction) |
| `wa_dataset/wa_unet_train.py` | U-Net ×4 on WA (reuses the KSA U-Net model definition) |

## Evaluation

| Script | Task |
|---|---|
| `ksa_aligned_evaluate.py` | Per-patch RMSE / SSIM / MS-SSIM on KSA splits; also the shared `evaluate_loader` library used by the other evaluators |
| `sweep_all_models.py` | Score every checkpoint in `checkpoints/`, auto-classifying model/dataset/channels from the stored config → `results/all_models_eval.csv` |
| `eval_synuc.py` | Score checkpoints on seeded operator-consistent synthetic-UC pairs |
| `eval_cv_folds.py` | Standalone 5-fold CV evaluation driver (pass `--prefix` for the checkpoint family) |
| `wa_dataset/wa_evaluate_models.py` | WA test-set comparison table |

## Reconstruction

| Script | Task |
|---|---|
| `ksa_aligned_reconstruct.py` | Patchwise SR + cosine blending over each block's held-out test region → GeoTIFF, metrics CSV, and comparison PNG |
| `ksa_aligned_reconstruct_region.py` | Same pipeline over a polygonal AOI (vector file) — the driver behind the full-AOI ensemble figure; run once per CV fold, then `plots/plot_recon_zone_ensemble.py` |

## `plots/`

Figure scripts: paper/talk figures (`plot_recon_triptych`, `plot_talk_results`,
`plot_synuc_study`, `plot_recon_zone_*`, `plot_lr_products`, …), operator schematics
(`schematic_fvd_operator*`, `schematic_drape_uc_3d`), and diagnostics
(`plot_cv_folds`, `plot_boundary_patching`, `plot_uc_inconsistency_ksa`,
`plot_pixel_distribution`). Each docstring states what it reads and produces.
Note: `plot_talk_results.py` reads the frozen `results/ablation_summary.csv`
snapshot from an earlier evaluation pass; the live, reproducible record is
`results/ksa_ablations_eval.csv` from `ksa_ablations/evaluate.py`.

## `_archive/`

Superseded one-off scripts, kept for the record. Includes the self-contained
multispectral null-result study (`_archive/multispectral/` — Landsat bands
carry no usable signal for magnetic SR; the `ms_bands` pathway remains
supported in `magsr.datasets` for anyone who wants the data).

# Building the WA Goldfields dataset

Preprocessing pipeline for the **Western Australia Goldfields** dataset. It clips the
state-wide GSWA aeromagnetic grids to the Goldfields window, snaps a DEM onto the LR grid,
and writes normalized HR/LR `.npy` patch pairs.

Source grids and download links are documented in the top-level [`README.md`](../../README.md)
(the `data/WA/` layout section). Shared parameters live in the `wa` section of
[`configs/datasets.yaml`](../../configs/datasets.yaml); every script reads them through
`WAConfig` and exposes CLI flags to override per-run. Patch tiling follows
[Smith et al., 2022](https://doi.org/10.1016/j.oregeorev.2022.105119) (128×128).

## Pipeline order

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 01 | `01_clip_wa_goldfields.py` | state-wide GSWA HR TMI (20 m) + LR TMI/1VD (80 m) | Goldfields HR TMI raster + 2-band LR `[TMI, 1VD]` raster (verifies CRS + 4× HR/LR ratio) |
| 02 | `02_plot_wa_overview.py` | 01's clipped rasters + train/test split polygons | `figures/wa_study_area_overview.png`  |
| 03 | `03_fetch_snap_wa_dem.py` | Geoscience Australia 1" SRTM DEM-S (fetched) + 01's LR grid | `snapped_cubicspline_dtm80m.tif` on the LR grid |
| 04 | `04_patch_wa_dataset.py` | 01's rasters (+ 03's DEM) | normalized HR/LR `.npy` patch pairs in `data/processed/wa/patches/` + `normalization.json` |

```bash
uv run python scripts/build_wa_dataset/01_clip_wa_goldfields.py
uv run python scripts/build_wa_dataset/02_plot_wa_overview.py
uv run python scripts/build_wa_dataset/03_fetch_snap_wa_dem.py
# --lr-aux appends LR product channels (e.g. 1VD); --use-dem adds the DEM-gradient channel.
uv run python scripts/build_wa_dataset/04_patch_wa_dataset.py --lr-aux 1VD --use-dem
```

Omit `--lr-aux` / `--use-dem` for the in=1 magnetic-only patch set. The channel set chosen
here must match what the trainer (`experiments/wa_dataset/wa_rdn.py`) is configured to consume.

## Diagnostic-only figures

- **`02b_plot_wa_satellite_overview.py`** — Esri World Imagery basemap over the AOI with the
  train/test boxes.
- **`03b_plot_wa_dem_overview.py`** — renders the snapped LR-grid DEM for artifact inspection.
- **`plot_survey_and_patch_footprints.py`** — source-survey line-spacing footprints (the
  ≤300 m selection criterion) + the 128×128 patch grid over the TMI raster.

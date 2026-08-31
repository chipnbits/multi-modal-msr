# KSA master grids

The grid definition the KSA dataset is built on.

| File | Size | Values | Purpose |
|---|---|---|---|
| `magnetic_base_grid30m.tif` | 38472×40032 | 0/1 | DEM grid (30 m) |
| `magnetic_base_grid60m.tif` | 19236×20016 | 0/1 | **HR master** — `03_snap_raster.py` target for the 60 m products |
| `magnetic_base_grid180m.tif` | 6412×6672 | 0/1 | **LR master** — target for the 180 m products |
| `magnetic_mask_grid60m.tif` | 19236×20016 | 0/1/2/3 | Same footprint, labelled by survey block (B1/B2/B3); drives per-block normalization and scoring |
| `msr_inference_grid180m.tif` | 7988×10457 | 0/1 | Extended LR grid for inference beyond the training extent |

All share EPSG:32637 and the origin (126679.09701728472, 3193301.539676539); 1 LR px =
3×3 HR px = 6×6 DEM px. 0 means outside the survey; `03_snap_raster.py` writes NaN there.

Stored as ZSTD-compressed GeoTIFF (33 MB of LZW → 634 KB, lossless: pixels, transform,
CRS, nodata and dtype all verified identical to the originals).

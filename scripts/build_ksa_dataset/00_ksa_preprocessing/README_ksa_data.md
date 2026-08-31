### KSA Shield aeromagnetic survey — from portal download to the snapped grid

Stage 00 turns the raw SGS portal download into the co-registered single-grid tree that
`MAGSR_KSA_ALIGNED_ROOT` points at.

#### 1. Aeromagnetic Survey (Data Acquisition)

The raw tiles are sourced from the [National Geological Database Portal](https://ngp.sgs.gov.sa/) (SGS). Two different portal layers are used depending on the product:

| Product | Portal layer |
|---|---|
| High-resolution (HR) survey blocks `GEOPH1`/`GEOPH2`/`GEOPH3` | `RGP – GPAS` |
| Legacy low-resolution (LR) sheets | `Geophysical (Legacy)` |

The portal is a map-based GIS application with no bulk-download API, so acquisition is a manual step rather
than a script.

1. Register and log in at https://ngp.sgs.gov.sa/.
2. In the **Maps** panel, enable the layer for the product you need — `RGP – GPAS` for the HR blocks, `Geophysical (Legacy)` for the LR sheets — and disable the others to avoid selecting features from the wrong dataset.
3. Use one of the **select by** tools (point, polyline, polygon, or bounding box) to define your AOI.
4. With the relevant tiles highlighted, click **Request Data** and choose the derivative products you need (e.g. AMF, RTP, RTP1VD, RTP2VD, RTPTILT, RTPHG, ANS, HGLAT, HGLONG, DTM for HR; TMI/RTP/RTP_ANSIG/RTP_FVD for the legacy sheets).
5. Submit the request. SGS emails a download link to your registered account — it is not an instant in-browser download.
6. Download and unzip into a single incoming folder, then sort and inventory:

   ```bash
   uv run python scripts/build_ksa_dataset/00_ksa_preprocessing/01_organize_raw_downloads.py \
       --incoming data/raw/ksa_shield/_incoming \
       --output   data/raw/ksa_shield
   ```

   This sorts the **HR** block tiles into `GEOPH1/GEOPH2/GEOPH3/`. The **legacy LR**
   sheets carry no block code, so they land in `UNSORTED/` — move them to
   `data/raw/ksa_shield/legacy/`, which is where the LR snap config globs.

#### 2. The master grid

The master grids ship pre-built in [`data/ksa_base_grids/`](../../../data/ksa_base_grids/)
(`magnetic_base_grid{30,60,180}m.tif`, the block mask `magnetic_mask_grid60m.tif`, and
`msr_inference_grid180m.tif`). They define the extent, pixel lattice and validity mask
every product shares, so **you do not need to build them to reproduce the dataset** —
step 3 snaps directly onto these committed files.

To regenerate them (a new AOI, or to reproduce their provenance), run `02_base_grid.py`
once per pixel size; its config writes back over the committed grid so step 3 is
unchanged:

```bash
uv run python scripts/build_ksa_dataset/00_ksa_preprocessing/02_base_grid.py   # 60 m (default config)
# edit config/02_base_grid.yaml: output -> ...180m.tif, pixel_size_m -> 180, then rerun for the LR grid
```

#### 3. Merging and Snapping to the Master Grid

`03_snap_raster.py` reprojects, resamples, mosaics and clips one product onto a master grid in one pass.
`cubicspline` is used throughout; for the LR sheets this also performs the upscaling from the native 200 m
grid to the 180 m project grid.

```yaml
# 00_ksa_preprocessing/config/03_snap_raster.yaml (the --config default) — LR legacy RTP
paths:
  master_grid: data/ksa_base_grids/magnetic_base_grid180m.tif
  output_dir: data/processed/ksa_aligned/02_snap_owned/aeromagnetics/180m/cubicspline
processing:
  resample_alg: cubicspline
  apply_mask: true
  overlap_strategy: first
  output_name: snapped_cubicspline_RTP.tif
inputs:
  glob:
    dir: data/raw/ksa_shield/legacy
    pattern: "**/*_RTP.tif"     # excludes *_RTP_ANSIG.tif / *_RTP_FVD.tif
output_format:
  compress: LZW
  bigtiff: IF_SAFER
  block_size: 512
```

```bash
uv run python scripts/build_ksa_dataset/00_ksa_preprocessing/03_snap_raster.py                              # LR RTP (default config)
uv run python scripts/build_ksa_dataset/00_ksa_preprocessing/03_snap_raster.py --config <dir>/config/snap_hr_mag_amf.yaml   # HR AMF
```

Two example configs ship: `03_snap_raster.yaml` (LR legacy RTP → 180 m grid) and
`snap_hr_mag_amf.yaml` (HR AMF → 60 m grid). **Copy one per product** — repoint
`inputs.glob` and `output_name` — to populate `02_snap_owned/`. The loader builds each
filename as `snapped_cubicspline_<stem>.tif`, so `output_name` must match the stem it
expects for the products in `configs/datasets.yaml`:

| Role | `datasets.yaml` key | `output_name` | Directory |
|---|---|---|---|
| HR default | `hr_products: [AMF_RTP]` | `snapped_cubicspline_MAG_AMF_RTP.tif` | `aeromagnetics/60m/cubicspline/` |
| LR default | `lr_products: [RTP]` | `snapped_cubicspline_RTP.tif` | `aeromagnetics/180m/cubicspline/` |
| DEM (optional) | `load_dem: true` | `snapped_cubicspline_dem30m.tif` | `elevation/` |

(the shipped `snap_hr_mag_amf.yaml` produces `MAG_AMF`, an example — for the default
pipeline set `output_name: snapped_cubicspline_MAG_AMF_RTP.tif` and glob the RTP tiles.)
Additional HR derivatives (`AMF_RTP1VD`, `AMF_RTPTILT`, …) and LR aux products
(`RTP_ANSIG`→`ANS`, `RTP_FVD`→`1VD`) follow the same pattern.

The legacy LR sheets declare `nodata=0` but carry meaningful zeros, and the shipped dataset was built
letting them through. The per-tile path honours each file's own nodata (`-999999` for the HR blocks).
Override with `processing.honor_nodata`.

#### 4. Closing the loop — point `MAGSR_KSA_ALIGNED_ROOT` at the result

The snapped products land under `data/processed/ksa_aligned/02_snap_owned/`. Point the
env var at that tree:

```bash
echo "MAGSR_KSA_ALIGNED_ROOT=data/processed/ksa_aligned" >> .env
```

The loader reads the block mask from `<root>/00_base_grid/`; when you build the dataset
yourself that folder won't exist, so it automatically falls back to the committed
`data/ksa_base_grids/` mask — nothing to copy. (If you maintain a full dedicated dataset
tree with its own `00_base_grid/`, that one takes precedence.)

Now continue with the normalization + cell-split stages in
[`../README.md`](../README.md) (stages 01–06).

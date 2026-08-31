"""Fetch the Geoscience Australia 1" DEM-S and snap it onto the WA LR magnetic grid.

Geoscience Australia 1" SRTM-derived DEM-S (smoothed) bare-earth DTM
SRTM C-band vegetation offset not fully removed.
https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/72759

The dataset is openly distributed under a Creative Commons
Attribution 4.0 International Licence (CC BY 4.0).

Gallant, J., Wilson, N., Dowling, T., Read, A., Inskeep, C. 2011. SRTM-derived
1 Second Digital Elevation Models Version 1.0. Record 1. Geoscience Australia, Canberra.

The DEM is reprojected/resampled (cubic spline) onto the LR magnetic raster grid.
Cubic spline is used in favor over cubic to

Output: data/WA/snapped_cubicspline_dtm80m.tif

Usage: uv run python scripts/build_wa_dataset/03_fetch_snap_wa_dem.py
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds

from magsr.datasets import WAConfig

# Geoscience Australia 1" SRTM-derived DEM-S (smoothed) single national COG on public source
# DEM-S is the smoothed product for the downstream gradient.
GA_DEMS_URL = (
    "/vsicurl/https://dea-public-data.s3.ap-southeast-2.amazonaws.com/"
    "projects/elevation/ga_srtm_dem1sv1_0/dems1sv1_0.tif"
)


def _ga_dems_window(bounds):
    """Windowed /vsicurl read of the GA DEM-S COG over `bounds` → (array, transform, crs, nodata)."""
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
    print("Reading GA DEM-S (smoothed, bare-earth) windowed via /vsicurl ...")
    with rasterio.open(GA_DEMS_URL) as d:
        left, bottom, right, top = bounds
        pad = 20 * d.transform.a  # ~20 source px padding so cubic resampling has support
        win = from_bounds(left - pad, bottom - pad, right + pad, top + pad, transform=d.transform)
        win = win.round_offsets().round_lengths()
        arr = d.read(1, window=win).astype(np.float32)
        return arr, d.window_transform(win), d.crs, d.nodata


def main() -> None:
    cfg = WAConfig.from_yaml()
    lr_path = cfg.data_dir / cfg.lr_filename
    out_path = cfg.data_dir / "snapped_cubicspline_dtm80m.tif"

    with rasterio.open(lr_path) as lr:
        dst_crs, dst_transform = lr.crs, lr.transform
        dst_h, dst_w = lr.height, lr.width
        bounds = lr.bounds
    print(f"Target grid (from {lr_path.name}): CRS={dst_crs} shape=({dst_h},{dst_w})")
    print(f"Bounds: {[round(b, 3) for b in bounds]}")

    src_arr, src_transform, src_crs, src_nodata = _ga_dems_window(bounds)

    print("Reprojecting -> LR grid (cubic spline), nodata -> NaN ...")
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.cubic_spline,
    )

    profile = {
        "driver": "GTiff",
        "height": dst_h,
        "width": dst_w,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as d:
        d.write(dst, 1)

    valid = np.isfinite(dst)
    print(f"\nWrote {out_path}")
    print(
        f"  shape={dst.shape} valid={valid.mean():.1%} "
        f"elev[min/med/max]={np.nanmin(dst):.1f}/{np.nanmedian(dst):.1f}/{np.nanmax(dst):.1f} m"
    )
    with rasterio.open(out_path) as d, rasterio.open(lr_path) as lr:
        assert (
            d.crs == lr.crs and d.transform == lr.transform and d.shape == lr.shape
        ), "DEM not aligned to LR grid!"
    print("  ALIGNED to LR magnetic grid ✓")


if __name__ == "__main__":
    main()

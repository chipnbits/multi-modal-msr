"""Build the master reference grid that every product is snapped to (rasterio only).

Reads the AOI polygon, buffers it, projects to the target UTM zone, and writes a
uint8 GeoTIFF of 1s covering the AOI extent — dimensions floored to a whole number of
`tile_divisor` blocks and the origin snapped to a whole pixel. `03_snap_raster.py`
warps every product onto exactly this grid and reads 0 as "outside the AOI", so the
master defines both the extent and the validity mask shared by the whole dataset.

Two grids are needed, one per target pixel size — 60 m for the HR products, 180 m for
the LR products. Run twice, changing `output` and `pixel_size_m` in the config.

Run:  uv run python scripts/build_ksa_dataset/00_ksa_preprocessing/02_base_grid.py
      uv run python .../02_base_grid.py --config config/02_base_grid_180m.yaml --yes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import yaml
from rasterio.transform import Affine
from shapely.ops import transform

from magsr import ROOT_FOLDER

# Relative paths in the config resolve against the repo root, so a config reads
# `data/raw/...` / `data/processed/...` the same way every other script does.
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "02_base_grid.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT_FOLDER / p


def to_crs(geom, src_epsg: int, dst_epsg: int):
    """Reproject a shapely geometry between two EPSG codes."""
    transformer = pyproj.Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return transform(transformer.transform, geom)


def best_grid_dims(width_m: float, height_m: float, pixel_size: float, target_px: int = 256):
    """Pixel counts floored to a whole number of `target_px` blocks, so the grid tiles
    evenly. Returns (nx, ny, adjusted_width_m, adjusted_height_m)."""
    nx = int(np.floor(width_m / pixel_size / target_px) * target_px)
    ny = int(np.floor(height_m / pixel_size / target_px) * target_px)
    return nx, ny, nx * pixel_size, ny * pixel_size


def recommend(config: dict) -> dict | None:
    """AOI -> grid parameters. KSA always sets `override_epsg` (32637), so no
    automatic UTM-zone detection is done here."""
    aoi_file = _resolve(config["paths"]["aoi"])
    if not aoi_file.exists():
        print(f"[!] AOI file missing: {aoi_file}")
        return None

    aoi_uri = f"zip://{aoi_file.as_posix()}" if aoi_file.suffix.lower() == ".zip" else aoi_file.as_posix()
    print(f"[*] Reading AOI: {aoi_uri}")
    gdf = gpd.read_file(aoi_uri).to_crs(epsg=4326)

    utm_epsg = int(config["grid"]["override_epsg"])
    aoi_utm = to_crs(gdf.geometry.union_all(), 4326, utm_epsg)

    buffer_m = config["grid"].get("buffer_m", 1000)  # positive = inward, negative = outward
    eroded = aoi_utm.buffer(-buffer_m)
    if eroded.is_empty or eroded.area == 0:
        print(f"[!] AOI collapsed after {abs(buffer_m)}m {'inner' if buffer_m > 0 else 'outer'} buffer.")
        return None

    minx, miny, maxx, maxy = eroded.bounds
    ps = config["grid"]["pixel_size_m"]
    target_px = config["grid"].get("tile_divisor", 256)
    nx, ny, adj_w, adj_h = best_grid_dims(maxx - minx, maxy - miny, ps, target_px)

    # Centre the adjusted extent on the eroded AOI, then snap the origin to whole
    # pixels. nx/ny are recomputed after snapping to absorb any floor() residual.
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    snap_minx = np.floor((cx - adj_w / 2) / ps) * ps
    snap_miny = np.floor((cy - adj_h / 2) / ps) * ps
    snap_maxx, snap_maxy = snap_minx + adj_w, snap_miny + adj_h
    nx = int(round((snap_maxx - snap_minx) / ps))
    ny = int(round((snap_maxy - snap_miny) / ps))

    print(
        f"\n{'=' * 55}\n"
        f"  CRS (override)         : EPSG:{utm_epsg}\n"
        f"  Pixel size             : {ps} m\n"
        f"  Grid dimensions        : {nx} x {ny} pixels\n"
        f"  Origin (snapped)       : {snap_minx:.0f}, {snap_miny:.0f} m\n"
        f"  Extent                 : {snap_maxx:.0f}, {snap_maxy:.0f} m\n"
        f"  Tile blocks            : {nx // target_px} x {ny // target_px} ({target_px}px each)\n"
        f"{'=' * 55}\n"
    )
    return {
        "utm_epsg": utm_epsg,
        "pixel_size": ps,
        "nx": nx,
        "ny": ny,
        "minx": snap_minx,
        "miny": snap_miny,
        "maxx": snap_maxx,
        "maxy": snap_maxy,
    }


def create_grid(params: dict, config: dict) -> Path:
    """Write the uint8 master grid, filled with 1s (0 = nodata = outside the AOI)."""
    out_file = _resolve(config["paths"]["output"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ps, nx, ny = params["pixel_size"], params["nx"], params["ny"]
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": nx,
        "height": ny,
        "crs": rasterio.crs.CRS.from_epsg(params["utm_epsg"]),
        "transform": Affine(ps, 0.0, params["minx"], 0.0, -ps, params["maxy"]),
        "nodata": 0,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
    }
    with rasterio.open(out_file, "w", **profile) as ds:
        ds.write(np.ones((ny, nx), dtype=np.uint8), 1)

    print(f"[+] Master grid saved: {out_file}")
    print(f"    EPSG:{params['utm_epsg']} | {nx} x {ny} px | {ps} m/px")
    return out_file


def main(config_file: Path, assume_yes: bool = False) -> None:
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return
    config = load_config(config_path)

    params = recommend(config)
    if params is None:
        return
    if not assume_yes:
        print("Proceed with this grid? [y/n]: ", end="", flush=True)
        if input().strip().lower() != "y":
            print("[*] Cancelled. Adjust config and re-run.")
            return
    create_grid(params, config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Grid config YAML.")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = ap.parse_args()
    main(args.config, assume_yes=args.yes)

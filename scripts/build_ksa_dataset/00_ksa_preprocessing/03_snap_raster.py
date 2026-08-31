"""Mosaic + reproject + clip one product onto a master grid (rasterio only).

Every product in `02_snap_owned/` is produced by this: the per-sheet survey tiles are
reprojected onto the master grid (`data/ksa_base_grids/`), mosaicked, and clipped to
the master's valid extent. Output is float32 with NaN nodata, on exactly the master
grid — so all products are pixel-aligned and can be stacked without further warping.

Two mosaic paths, matching how GDAL handles the two cases:

  homogeneous  All tiles share a CRS, a resolution, and a pixel lattice: they are laid
               into one array first, then warped in a single pass. The resampling
               kernel therefore sees across tile seams, so mosaic joins stay smooth.
  mixed        Otherwise (e.g. the HR survey blocks, which span UTM zones 36-38N):
               each tile is warped independently into the shared destination. Earlier
               tiles survive wherever the current tile is nodata.

`overlap="first"` means the first file in sorted order wins where tiles overlap.

NOTE ON NODATA: in the homogeneous path the tiles' declared nodata is deliberately
NOT honoured — the legacy LR sheets declare nodata=0 but carry meaningful zeros, and
the reference dataset was built letting them through. The mixed path honours each
file's own nodata. `honor_nodata` overrides this per call.

Run:  uv run python scripts/build_ksa_dataset/00_ksa_preprocessing/03_snap_raster.py
      uv run python .../03_snap_raster.py --config config/snap_hr_amf.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

from magsr import ROOT_FOLDER

# Relative paths in a config resolve against the repo root, so a config reads
# `data/raw/...` / `data/processed/...` the same way every other script does.
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "03_snap_raster.yaml"

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubicspline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
}

# Refuse to build an in-memory mosaic larger than this: float32, so ~8 GB. The KSA
# legacy sheets mosaic to ~59 M pixels, well inside it.
MAX_MOSAIC_PX = 2_000_000_000


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT_FOLDER / p


def read_master(path: Path) -> dict:
    """Master grid geometry + its 0/1 validity array (0 = outside the AOI)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nd = ds.nodata
        return {
            "transform": ds.transform,
            "crs": ds.crs,
            "width": ds.width,
            "height": ds.height,
            "bounds": ds.bounds,
            "valid": arr != 0 if nd is None else (arr != 0) & (arr != nd),
            "has_zeros": bool((arr == 0).any()),
        }


def filter_overlapping(files: list[Path], master: dict) -> list[Path]:
    """Drop tiles whose footprint misses the master grid, reprojecting each tile's
    bounds into the master CRS first (the HR blocks span three UTM zones)."""
    m = master["bounds"]
    keep = []
    for f in files:
        try:
            with rasterio.open(f) as ds:
                b = ds.bounds
                if ds.crs != master["crs"]:
                    b = transform_bounds(ds.crs, master["crs"], *b, densify_pts=21)
        except rasterio.errors.RasterioIOError:
            print(f"    [!] cannot open {f.name} — skipping")
            continue
        if b[2] > m.left and b[0] < m.right and b[3] > m.bottom and b[1] < m.top:
            keep.append(f)
        else:
            print(f"    [!] {f.name} lies entirely outside the master grid — skipped")
    return keep


def _homogeneous(files: list[Path]) -> tuple[bool, str]:
    """True when every tile shares a CRS, a resolution and a pixel lattice — the
    condition under which laying them into one array reproduces a GDAL VRT mosaic."""
    with rasterio.open(files[0]) as ds:
        crs, (rx, ry) = ds.crs, ds.res
    x0, y0 = None, None
    for f in files:
        with rasterio.open(f) as ds:
            if ds.crs != crs:
                return False, "tiles span multiple CRSs"
            if ds.res != (rx, ry):
                return False, "tiles have mixed resolutions"
            t = ds.transform
            if x0 is None:
                x0, y0 = t.c, t.f
            elif abs((t.c - x0) / rx % 1) > 1e-6 or abs((t.f - y0) / ry % 1) > 1e-6:
                return False, "tiles are not on a common pixel lattice"
    return True, "shared CRS, resolution and lattice"


def build_mosaic(files: list[Path], *, honor_nodata: bool, overlap: str):
    """Lay lattice-aligned tiles into one union-extent array (last written wins)."""
    metas = []
    for f in files:
        with rasterio.open(f) as ds:
            metas.append((f, ds.transform, ds.width, ds.height, ds.crs, ds.nodata))
    rx, ry = metas[0][1].a, -metas[0][1].e
    xmin = min(m[1].c for m in metas)
    ymax = max(m[1].f for m in metas)
    xmax = max(m[1].c + m[2] * m[1].a for m in metas)
    ymin = min(m[1].f - m[3] * ry for m in metas)
    W, H = int(round((xmax - xmin) / rx)), int(round((ymax - ymin) / ry))
    if W * H > MAX_MOSAIC_PX:
        raise MemoryError(
            f"mosaic would be {W}x{H} = {W * H / 1e9:.1f}G pixels; raise MAX_MOSAIC_PX "
            "or split the product into several configs"
        )

    mosaic = np.full((H, W), np.nan, dtype=np.float32)
    order = list(reversed(metas)) if overlap == "first" else metas
    for f, t, w, h, _crs, nd in order:
        c0 = int(round((t.c - xmin) / rx))
        r0 = int(round((ymax - t.f) / ry))
        with rasterio.open(f) as ds:
            a = ds.read(1).astype(np.float32)
        if honor_nodata and nd is not None:
            a = np.where(a == nd, np.nan, a)
        mosaic[r0 : r0 + h, c0 : c0 + w] = a
    return mosaic, Affine(rx, 0.0, xmin, 0.0, -ry, ymax), metas[0][4], (W, H)


def snap_rasters(
    files: list[Path],
    master_path: Path,
    out_path: Path,
    *,
    resampling: Resampling = Resampling.cubic_spline,
    overlap: str = "first",
    apply_mask: bool = True,
    honor_nodata: bool | None = None,
    creation: dict | None = None,
) -> Path:
    """Warp `files` onto the master grid of `master_path` and write `out_path`.

    Inputs are ordered by *filename*, not by full path: with `overlap="first"` the
    order decides who wins where tiles overlap, and sorting on the tile's own name
    keeps that outcome independent of how the download happened to be foldered.
    """
    master = read_master(master_path)
    H, W = master["height"], master["width"]
    files = filter_overlapping(sorted(files, key=lambda p: p.name), master)
    if not files:
        raise RuntimeError("No input files overlap the master grid — nothing to merge.")

    homogeneous, why = _homogeneous(files)
    print(f"[*] Merging  : {len(files)} file(s) -> {out_path.name}")
    print(f"    Master   : {W} x {H} px | {master['transform'].a:g} m/px")
    print(f"    Mosaic   : {'single-pass' if homogeneous else 'per-tile'} ({why})")

    dst = np.full((H, W), np.nan, dtype=np.float32)
    if homogeneous:
        keep_nodata = bool(honor_nodata) if honor_nodata is not None else False
        mosaic, src_transform, src_crs, (mw, mh) = build_mosaic(
            files, honor_nodata=keep_nodata, overlap=overlap
        )
        print(f"    Source   : {mw} x {mh} px mosaic, nodata {'honoured' if keep_nodata else 'ignored'}")
        reproject(
            source=mosaic,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=master["transform"],
            dst_crs=master["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
            init_dest_nodata=True,
            num_threads=8,
            warp_mem_limit=4096,
        )
    else:
        keep_nodata = True if honor_nodata is None else bool(honor_nodata)
        order = list(reversed(files)) if overlap == "first" else list(files)
        for i, f in enumerate(order, 1):
            with rasterio.open(f) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    src_nodata=src.nodata if keep_nodata else None,
                    dst_transform=master["transform"],
                    dst_crs=master["crs"],
                    dst_nodata=np.nan,
                    resampling=resampling,
                    init_dest_nodata=False,  # keep what earlier tiles wrote
                    num_threads=8,
                    warp_mem_limit=1024,
                )
            if i % 50 == 0 or i == len(order):
                print(f"    warped {i}/{len(order)}")

    if apply_mask and master["has_zeros"]:
        n = int((~master["valid"]).sum())
        dst[~master["valid"]] = np.nan
        print(f"    Mask     : {n:,} pixels outside the master extent set to NaN")
    elif apply_mask:
        print("    Mask     : skipped (master grid has no 0s — extent-only grid)")

    c = creation or {}
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": W,
        "height": H,
        "crs": master["crs"],
        "transform": master["transform"],
        "nodata": np.nan,
        "tiled": True,
        "blockxsize": int(c.get("block_size", 512)),
        "blockysize": int(c.get("block_size", 512)),
        "compress": str(c.get("compress", "LZW")).lower(),
        "BIGTIFF": str(c.get("bigtiff", "IF_SAFER")).upper(),
    }
    if profile["compress"] == "deflate":
        profile["predictor"] = int(c.get("predictor", 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(dst, 1)
    print(f"    Output   : {out_path}\n")
    return out_path


def main(config_file: Path) -> None:
    config_path = Path(config_file).resolve()
    if not config_path.exists():
        print(f"[!] Config not found: {config_path}")
        return
    cfg = load_config(config_path)

    master_file = _resolve(cfg["paths"]["master_grid"])
    if not master_file.exists():
        print(f"[!] Master grid missing: {master_file}")
        return
    out_dir = _resolve(cfg["paths"]["output_dir"])

    proc = cfg.get("processing", {})
    alg = str(proc.get("resample_alg", "cubicspline")).lower()
    if alg not in RESAMPLING:
        print(f"[!] Unknown resample_alg '{alg}'. Falling back to cubicspline.")
    resampling = RESAMPLING.get(alg, Resampling.cubic_spline)

    inputs = cfg["inputs"]
    if "glob" in inputs:
        base = _resolve(inputs["glob"]["dir"])
        files = sorted(base.glob(inputs["glob"]["pattern"]))
    else:
        files = [_resolve(f) for f in inputs.get("files", [])]
    if not files:
        print("[!] No input files found. Check the config's inputs section.")
        return

    print(f"[*] Master grid      : {master_file.name}")
    print(f"[*] Resample         : {alg}")
    print(f"[*] Files to process : {len(files)}\n")
    try:
        snap_rasters(
            files,
            master_file,
            out_dir / proc.get("output_name", "merged_output.tif"),
            resampling=resampling,
            overlap=str(proc.get("overlap_strategy", "first")).lower(),
            apply_mask=bool(proc.get("apply_mask", True)),
            honor_nodata=proc.get("honor_nodata"),
            creation=cfg.get("output_format", {}),
        )
    except Exception as e:  # keep batch runs alive; the message names the product
        print(f"[!] Merge failed — {e}")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Snap config YAML (one per product)."
    )
    main(ap.parse_args().config)

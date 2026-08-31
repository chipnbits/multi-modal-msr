"""
See the README for WA dataset download instructions.
Stage 11: Clip the downloaded state-wide WA grids down to the Goldfields window.

Place the decompressed GSWA grids under --raw-dir
(default data/WA/raw/).

    data/WA/raw/
    ├── 20m_mag/
    │   ├── WA_20m_Mag_Merge_v1_2023          # ~31 GB binary  (HR TMI source)
    │   ├── WA_20m_Mag_Merge_v1_2023.ers      #   <- the script opens this
    │   └── WA_20m_Mag_Merge_v1_2023.ers.aux.xml
    └── 80m_mag/
        ├── WA_80m_Mag_Merge_v1_2023(.ers)        # ~2 GB  (LR TMI source)
        └── WA_80m_Mag_Merge_1VD_v1_2023(.ers)    # ~2 GB  (LR 1VD source)

Produces two smaller Goldfields rasters in `data/WA` (or `--out-dir`):

    Goldfields_20m_HR.tif   single-band HR  (~0.000208 deg cells): TMI
    Goldfields_80m_LR.tif   two-band LR     (~0.000833 deg cells, 4x HR): [TMI, 1VD]

The 80 m TMI and 80 m 1VD are on the same GSWA lattice.

Clips to the bbox in `data/WA/goldfields_clip.json`. Edit the json bounds for a new region.

Sources must be EPSG:7844 (GDA2020 Geodetic)

Usage:
    uv run python scripts/build_wa_dataset/01_clip_wa_goldfields.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

from magsr.datasets import WAConfig

TARGET_CRS = CRS.from_epsg(7844)  # GDA2020 geographic — the GSWA native grid CRS
RASTER_EXTS = {".tif", ".tiff", ".ers", ".grd", ".dat", ".vrt", ".img", ".nc"}
NODATA = -99999.0
VD_RE = re.compile(r"1vd", re.IGNORECASE)


def _discover_sources(raw_dir: Path, skip_names: tuple[str, ...]) -> tuple[Path, Path, Path | None]:
    """Find state-wide grids in raw_dir: HR (finest cell), LR TMI + 1VD (coarse, split by name)."""
    cands: list[tuple[float, Path]] = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in RASTER_EXTS or p.name in skip_names:
            continue
        try:
            with rasterio.open(p) as ds:
                cands.append((abs(ds.transform.a), p))
        except rasterio.errors.RasterioIOError:
            continue
    if len(cands) < 2:
        raise FileNotFoundError(
            f"Need the decompressed 20 m + 80 m grids in {raw_dir}; found {len(cands)}. "
            "Pass --hr-src/--lr-src/--vd-src explicitly."
        )
    cands.sort(key=lambda t: t[0])  # ascending cell size -> finest first
    hr = cands[0][1]
    coarse = [p for _, p in cands[1:]]
    vd = next((p for p in coarse if VD_RE.search(p.name)), None)
    lr = next((p for p in coarse if p is not vd), None)
    print(f"Discovered HR={hr.name}  LR(TMI)={lr.name if lr else None}  1VD={vd.name if vd else None}")
    return hr, lr, vd


def _read(src, win: Window) -> np.ndarray:
    """Read a window as float32, remapping the source nodata to NODATA."""
    data = src.read(1, window=win, boundless=True, fill_value=NODATA).astype(np.float32)
    if src.nodata is not None and src.nodata != NODATA:
        data = np.where(data == src.nodata, NODATA, data)
    return data


def _clip(
    bbox: tuple[float, float, float, float], primary: Path, *extra: Path
) -> tuple[Affine, list[np.ndarray]]:
    """Crop EPSG:7844 GSWA grids to bbox. `primary` fixes the pixel window;
    `extra` bands (asserted to share its pixel lattice) are read on that exact window."""
    with rasterio.open(primary) as src:
        assert src.crs == TARGET_CRS, f"{primary.name}: expected {TARGET_CRS}, got {src.crs}"
        win = from_bounds(*bbox, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height)).round_offsets().round_lengths()
        transform = window_transform(win, src.transform)
        bands = [_read(src, win)]
    for p in extra:
        with rasterio.open(p) as src:
            assert src.crs == TARGET_CRS and np.isclose(
                abs(src.transform.a), abs(transform.a), rtol=1e-4
            ), f"{p.name}: not on the {primary.name} lattice (crs={src.crs}, cell={src.transform.a})"
            bands.append(_read(src, win))
    return transform, bands


def _write(dst_path: Path, bands: list[np.ndarray], transform: Affine, descriptions: list[str]) -> None:
    """Write a (multi-band) EPSG:7844 GeoTIFF with per-band descriptions."""
    profile = {
        "driver": "GTiff",
        "height": bands[0].shape[0],
        "width": bands[0].shape[1],
        "count": len(bands),
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
    }
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        for i, (arr, desc) in enumerate(zip(bands, descriptions), start=1):
            dst.write(arr, i)
            dst.set_band_description(i, desc)
    summ = ", ".join(f"{d}={float((b != NODATA).mean()):.0%}" for b, d in zip(bands, descriptions))
    print(
        f"  wrote {dst_path.name}: {len(bands)}-band shape={bands[0].shape} cell={transform.a:.6g} valid[{summ}]"
    )


def _verify(hr_path: Path, lr_path: Path, cfg: WAConfig) -> None:
    """Sanity-check the clipped pair: shared CRS and the expected HR/LR cell ratio."""
    with rasterio.open(hr_path) as hr, rasterio.open(lr_path) as lr:
        assert hr.crs == lr.crs, f"CRS mismatch: {hr.crs} vs {lr.crs}"
        ratio_x, ratio_y = lr.transform.a / hr.transform.a, lr.transform.e / hr.transform.e
        print(f"\nHR: CRS={hr.crs} cell=({hr.transform.a:.10f}, {hr.transform.e:.10f}) shape={hr.shape}")
        print(
            f"LR: CRS={lr.crs} cell=({lr.transform.a:.10f}, {lr.transform.e:.10f}) shape={lr.shape} "
            f"bands={lr.count} {lr.descriptions}"
        )
        print(f"HR/LR cell ratio: x={ratio_x:.4f} y={ratio_y:.4f} (expect ~{cfg.scale_factor})")
        assert np.isclose(ratio_x, cfg.scale_factor, rtol=0.02), f"x ratio {ratio_x} != {cfg.scale_factor}"
        assert np.isclose(ratio_y, cfg.scale_factor, rtol=0.02), f"y ratio {ratio_y} != {cfg.scale_factor}"


def main() -> None:
    cfg = WAConfig.from_yaml()
    ap = argparse.ArgumentParser(description="Clip state-wide WA grids to the Goldfields window")
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=cfg.data_dir / "raw",
        help="Dir holding the decompressed state-wide grids (default: data/WA/raw).",
    )
    ap.add_argument(
        "--hr-src", type=Path, default=None, help="20 m TMI merge grid (auto-discovered if omitted)."
    )
    ap.add_argument("--lr-src", type=Path, default=None, help="80 m TMI grid (auto-discovered if omitted).")
    ap.add_argument(
        "--vd-src", type=Path, default=None, help="80 m 1VD-of-TMI grid (auto-discovered if omitted)."
    )
    ap.add_argument("--out-dir", type=Path, default=cfg.data_dir, help="Where to write the clipped TIFs.")
    ap.add_argument(
        "--clip",
        type=Path,
        default=cfg.data_dir / "goldfields_clip.json",
        help="Clip region (bbox json) reproducing the Goldfields window.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing clipped outputs.")
    args = ap.parse_args()

    hr_out, lr_out = args.out_dir / cfg.hr_filename, args.out_dir / cfg.lr_filename
    if hr_out.exists() and lr_out.exists() and not args.force:
        print(
            f"{cfg.hr_filename} and {cfg.lr_filename} already exist in {args.out_dir} (use --force to overwrite)."
        )
        return

    hr_src, lr_src, vd_src = args.hr_src, args.lr_src, args.vd_src
    if hr_src is None or lr_src is None or vd_src is None:
        d_hr, d_lr, d_vd = _discover_sources(args.raw_dir, (cfg.hr_filename, cfg.lr_filename))
        hr_src, lr_src, vd_src = hr_src or d_hr, lr_src or d_lr, vd_src or d_vd
    if vd_src is None:
        raise FileNotFoundError("No 1VD grid found (need a *1VD* raster for the LR band 2). Pass --vd-src.")

    if not args.clip.exists():
        raise FileNotFoundError(f"Clip region {args.clip} not found.")
    bbox = tuple(json.loads(args.clip.read_text())["bounds"])
    print(f"Clipping to {args.clip.name}: {[round(b, 4) for b in bbox]}")

    print(f"\nHR  {hr_src} -> {hr_out}")
    hr_transform, hr_bands = _clip(bbox, hr_src)
    _write(hr_out, hr_bands, hr_transform, ["TMI"])

    print(f"\nLR  TMI={lr_src.name} + 1VD={vd_src.name} -> {lr_out}")
    lr_transform, lr_bands = _clip(bbox, lr_src, vd_src)
    _write(lr_out, lr_bands, lr_transform, ["TMI", "1VD"])

    _verify(hr_out, lr_out, cfg)
    print("Done")


if __name__ == "__main__":
    main()

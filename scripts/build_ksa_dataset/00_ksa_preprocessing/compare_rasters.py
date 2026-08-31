"""Compare a rebuilt raster against a reference one, pixel for pixel.

Used to verify that a rebuild of `02_snap_owned/` reproduces the shipped dataset (see
the "Verification" section of README_ksa_data.md), and useful any time a pipeline
change should be a no-op.

Reports the validity-mask agreement, the residual, and how much of the residual sits
inside GDAL's default 0.125-pixel transformer tolerance — the reference products were
warped by `gdal.Warp`, which approximates the coordinate mapping to that tolerance,
while `rasterio.warp.reproject` transforms exactly. A residual bounded by
0.125 * |grad| is that approximation, not a defect.

Run:  uv run python scripts/build_ksa_dataset/00_ksa_preprocessing/compare_rasters.py \
          --built  data/processed/ksa_aligned/02_snap_owned/.../snapped_cubicspline_RTP.tif \
          --reference $MAGSR_KSA_ALIGNED_ROOT/02_snap_owned/.../snapped_cubicspline_RTP.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


def compare(built: Path, reference: Path, *, blocks: Path | None = None) -> dict:
    with rasterio.open(built) as a, rasterio.open(reference) as b:
        if (a.width, a.height) != (b.width, b.height):
            sys.exit(f"[!] shape mismatch: {a.width}x{a.height} vs {b.width}x{b.height}")
        if a.transform != b.transform:
            print("[!] WARNING: transforms differ — the rasters are not on the same grid")
        got, ref = a.read(1), b.read(1)

    g_nan, r_nan = np.isnan(got), np.isnan(ref)
    both = ~g_nan & ~r_nan
    d = np.abs(got[both] - ref[both]).astype(np.float64)
    ref_std = float(np.nanstd(ref))

    print(f"built     : {built}")
    print(f"reference : {reference}\n")
    print(f"  validity mask identical : {np.array_equal(g_nan, r_nan)}")
    print(f"    data where ref is NaN : {int((~g_nan & r_nan).sum()):,}")
    print(f"    NaN where ref has data: {int((g_nan & ~r_nan).sum()):,}")
    print(f"  compared pixels         : {int(both.sum()):,}")
    print(f"  residual  max / mean    : {d.max():.6g} / {d.mean():.6g}")
    print(
        f"  residual  RMS           : {np.sqrt((d**2).mean()):.6g}  "
        f"({np.sqrt((d**2).mean()) / ref_std:.4%} of reference std {ref_std:.6g})"
    )
    for t in (1e-3, 1e-2, 1e-1, 1.0):
        print(f"    |d| < {t:<6g}          : {(d < t).mean():.4%}")

    gy, gx = np.gradient(np.where(r_nan, np.nan, ref))
    gmag = np.hypot(gy, gx)
    m = both & np.isfinite(gmag)
    envelope = (
        float((np.abs(got[m] - ref[m]) <= 0.125 * gmag[m] + 1e-6).mean()) if m.any() else float("nan")
    )
    print(f"  within GDAL's 0.125-px transformer envelope: {envelope:.4%}")

    out = {"rms": float(np.sqrt((d**2).mean())), "envelope": envelope}

    if blocks is not None:
        with rasterio.open(blocks) as ds:
            blk = ds.read(1)
        print("\n  per-block statistics (the numbers downstream normalization uses):")
        print(
            f"    {'blk':>4} {'n':>12} {'mean shipped':>14} {'mean built':>14} {'d mean':>10} {'d std':>10}"
        )
        for b_ in sorted(v for v in np.unique(blk) if v != 0):
            mm = (blk == b_) & both
            if not mm.any():
                continue
            x, y = ref[mm].astype(np.float64), got[mm].astype(np.float64)
            print(
                f"    {b_:>4} {int(mm.sum()):>12,} {x.mean():>14.6f} {y.mean():>14.6f} "
                f"{abs(x.mean() - y.mean()):>10.3e} {abs(x.std() - y.std()):>10.3e}"
            )
        vr, vb = np.nanmin(ref), np.nanmax(ref)
        gr, gb = np.nanmin(got), np.nanmax(got)
        print(f"    global vmin/vmax shipped: {vr:.6f} / {vb:.6f}")
        print(f"    global vmin/vmax built  : {gr:.6f} / {gb:.6f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", required=True, type=Path, help="Raster produced by 03_snap_raster.py")
    ap.add_argument("--reference", required=True, type=Path, help="Reference raster to compare against")
    ap.add_argument(
        "--blocks",
        type=Path,
        default=None,
        help="Optional block-ID raster (e.g. magnetic_mask_grid60m.tif) for per-block stats",
    )
    a = ap.parse_args()
    compare(a.built, a.reference, blocks=a.blocks)


if __name__ == "__main__":
    main()

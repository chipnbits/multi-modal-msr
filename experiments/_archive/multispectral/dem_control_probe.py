"""DEM control for the multispectral null — does the SAME pixelwise probe say the DEM is useful?

The multispectral study concluded "R^2 ~ 0 pixelwise ⇒ no SR signal". That inference is
only sound if the probe can actually DETECT a modality we know helps. The DEM is exactly
that modality: as a conditioning channel it is worth ~1.6 nT under the drape operator and
is in the deployed model. So run the identical probe on it.

Two outcomes, and they mean opposite things:
  * DEM scores clearly above zero  -> the probe detects useful modalities, so the
    multispectral zero is a real null. The screening argument stands.
  * DEM also scores ~zero          -> pixelwise R^2 does NOT predict channel usefulness,
    because the DEM helps through the FORWARD OPERATOR (it sets the drape height that
    modulates the continuation kernel), not through correlation with the target. The
    multispectral argument must then rest on the network trial + physics, NOT on R^2.

Identical to target_pc_components.py in every other respect: same 60 m grid, same valid
footprint, same targets (D_hp = HR - up3(down3(HR)); D_real = HR - bicubic(LR)), same
6x6 spatial-block GroupKFold, same linear / gradient-boosted estimators.

Predictors, matching what the network is actually fed (`dataset.dem_features`):
    elev    raw elevation (m)
    grad    d/dx, d/dy of elevation      <- the channel in the deployed combo model
    relief  elevation minus local mean   <- the channel that wins on synthetic drape
    all     the three above, stacked
    bands   the 7 Landsat bands          <- reference, the known null

Usage:
    uv run --with scikit-learn python experiments/multispectral/dem_control_probe.py
    uv run --with scikit-learn python experiments/multispectral/dem_control_probe.py --decimate 4
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from magsr import ROOT_FOLDER
from magsr.datasets import KSAAlignedConfig

BANDS = (1, 2, 3, 4, 5, 6, 7)
EPS = 1e-3
RELIEF_W = 3  # local-mean window in LR pixels (3 * 180 m ~ the patch-mean scale of dem_to_lr)


def read_resampled(path, out_shape, resampling):
    with rasterio.open(path) as ds:
        return ds.read(1, out_shape=out_shape, resampling=resampling).astype(np.float32)


def cv_r2(make_model, X, y, groups) -> tuple[float, float]:
    gkf = GroupKFold(n_splits=5)
    r2s = []
    for tr, te in gkf.split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], m.predict(X[te]).ravel()))
    return float(np.mean(r2s)), float(np.std(r2s))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--subsample", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT_FOLDER / "results/multispectral/dem_control_probe.csv")
    args = p.parse_args()
    t0, rng = time.time(), np.random.default_rng(args.seed)

    cfg = KSAAlignedConfig()
    hr_path, lr_path = cfg.hr_product_path(cfg.hr_products[0]), cfg.lr_product_path("RTP")
    with rasterio.open(hr_path) as ds:
        H, W = ds.height, ds.width
    with rasterio.open(lr_path) as ds:
        lrH, lrW = ds.height, ds.width

    d = args.decimate
    out, lr_out = (H // d, W // d), (lrH // d, lrW // d)
    print(f"[{time.time()-t0:6.1f}s] grid {out} ({60*d} m/px)", flush=True)

    # --- targets, byte-identical construction to target_pc_components.py
    hr = read_resampled(hr_path, out, Resampling.average if d > 1 else Resampling.nearest)
    valid = np.isfinite(hr)
    lrup = read_resampled(lr_path, out, Resampling.cubic)
    hr_lo = read_resampled(hr_path, lr_out, Resampling.average)
    hr_lo = np.where(np.isfinite(hr_lo), hr_lo, np.nanmean(hr_lo)).astype(np.float32)
    hr_loup = ndimage.zoom(hr_lo, (out[0] / lr_out[0], out[1] / lr_out[1]), order=3).astype(np.float32)
    D_real = hr - lrup
    D_hp = hr - hr_loup
    valid &= np.isfinite(lrup)

    # --- DEM channels, as the network receives them.
    # Derive from a hole-filled DEM: np.gradient / uniform_filter propagate NaN outward,
    # which would poison otherwise-valid pixels adjacent to a nodata cell. The `valid`
    # mask still excludes the filled cells themselves from every score.
    dem = read_resampled(cfg.dem_path, out, Resampling.average)
    dem_ok = np.isfinite(dem)
    valid &= dem_ok
    dem_f = np.where(dem_ok, dem, np.nanmean(dem[dem_ok])).astype(np.float32)
    gy, gx = np.gradient(dem_f)  # d/dy, d/dx (per 60 m px)
    relief = dem_f - ndimage.uniform_filter(dem_f, size=RELIEF_W * 3)  # mean-centred local height
    dem = dem_f

    # --- the known-null reference
    ms_dir = Path(cfg.ms_band_path(1)).parent
    bands = {
        i: read_resampled(ms_dir / f"snapped_landsat_b{i}.tif", out, Resampling.average) for i in BANDS
    }
    for i in BANDS:
        valid &= np.isfinite(bands[i])
    print(f"[{time.time()-t0:6.1f}s] valid pixels {int(valid.sum()):,}", flush=True)

    FEATS = {
        "DEM elevation (raw)": [dem],
        "DEM gradient (dx, dy)": [gx, gy],
        "DEM relief (mean-centred)": [relief],
        "DEM all (elev+grad+relief)": [dem, gx, gy, relief],
        "Landsat 7 bands (reference null)": [bands[i] for i in BANDS],
        "Upsampled LR (positive control)": [lrup],
    }

    # --- shared subsample + spatial groups (6x6 blocks), identical to the MS study
    rr, cc = np.where(valid)
    nb = 6
    grp_all = (rr * nb // out[0]) * nb + (cc * nb // out[1])
    idx = rng.choice(rr.size, min(args.subsample, rr.size), replace=False)
    gs = grp_all[idx]
    y_hp, y_real, y_field = D_hp[valid][idx], D_real[valid][idx], hr[valid][idx]

    rows = []
    for name, chans in FEATS.items():
        X = np.stack([c[valid][idx] for c in chans], axis=1).astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + EPS)
        assert np.isfinite(X).all(), f"non-finite predictor in {name!r}"
        # pixelwise |r| against the detail (the same quantity the PCA rows report)
        r_hp = abs(np.corrcoef(X[:, 0], y_hp)[0, 1]) if X.shape[1] == 1 else float("nan")
        lin_hp, _ = cv_r2(LinearRegression, X, y_hp, gs)
        gbm_hp, _ = cv_r2(
            lambda: HistGradientBoostingRegressor(
                max_depth=4, learning_rate=0.1, max_iter=250, random_state=0
            ),
            X,
            y_hp,
            gs,
        )
        lin_fd, _ = cv_r2(LinearRegression, X, y_field, gs)
        gbm_fd, _ = cv_r2(
            lambda: HistGradientBoostingRegressor(
                max_depth=4, learning_rate=0.1, max_iter=250, random_state=0
            ),
            X,
            y_field,
            gs,
        )
        lin_re, _ = cv_r2(LinearRegression, X, y_real, gs)
        rows.append(
            dict(
                predictor=name,
                n_ch=X.shape[1],
                abs_r_detail=r_hp,
                r2_detail_linear=lin_hp,
                r2_detail_gbm=gbm_hp,
                r2_real_linear=lin_re,
                r2_field_linear=lin_fd,
                r2_field_gbm=gbm_fd,
            )
        )
        print(
            f"[{time.time()-t0:6.1f}s] {name:36s} "
            f"R2(detail) lin {lin_hp:+.5f} gbm {gbm_hp:+.5f} | "
            f"R2(field) lin {lin_fd:+.4f} gbm {gbm_fd:+.4f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

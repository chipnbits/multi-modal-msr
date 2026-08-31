"""Does LARGER-THAN-PIXEL spectral STRUCTURE predict the magnetic field/detail, where
pixelwise values do not?

The pixelwise/local battery (rank_bands_fullraster.py) found R2(SR-detail ~ bands) ~ 0.
But that scores co-located values and small fixed filters (3x3 gradient, 5x5 band-pass).
A CNN could instead learn ANISOTROPIC, MULTI-SCALE TEXTURE / SPATIAL PATTERNS over a
neighborhood (lineament orientation, roughness, ring/fabric structure) — information a
local-magnitude feature is blind to (a striped dike field and a flat field can share the
same local mean). This probe is the cheap stand-in for that CNN: build a bank of
multi-scale, orientation-aware texture features from the spectra and ask a gradient-
boosted model (spatial-block CV) whether they predict the magnetics better than the raw
pixel values.

Base maps (span the spectral space incl. a nonlinear ratio): PC1, PC2, PC3, Ferrous
(b6/b5), Clay (b6/b7). For each, at scales s = {3,9,27,81} px (180 m .. 4.9 km):
  - local mean (uniform_filter)            -> large-scale brightness/blob structure
  - local std  (texture energy/roughness)  -> fabric amplitude
and at structure-tensor scales rho = {9,27}:
  - tensor energy (Jxx+Jyy)                -> edge density
  - tensor coherence ((Jxx-Jyy)^2+4Jxy^2)/(Jxx+Jyy)^2 -> ORIENTED fabric strength
=> 5 base x (4 mean + 4 std + 2 energy + 2 coherence) = 60 structural features.

Targets: SR detail d_hp (primary) and the raw field HR. Compared against the PIXELWISE
baseline (the 5 base values at the centre pixel). A real lift of R2(d_hp) above ~0 would
say "structure matters, try a CNN"; no lift strengthens the negative result to "even
multi-scale oriented texture carries nothing about the SR detail".

Spatial-block GroupKFold guards against autocorrelation. Full raster; features computed
on the full grid then sampled on a regular lattice (no patches).

Usage:
    uv run --with scikit-learn python experiments/multispectral/texture_probe.py
    uv run --with scikit-learn python experiments/multispectral/texture_probe.py --decimate 4   # fast
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from magsr import ROOT_FOLDER
from magsr.datasets import KSAAlignedConfig

BANDS = (1, 2, 3, 4, 5, 6, 7)
EPS = 1e-3
SCALES = (3, 9, 27, 81)  # uniform_filter window (px); 180 m .. 4860 m
RHOS = (9, 27)  # structure-tensor smoothing (px)


def log(m, t0):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def read_resampled(path, out_shape, resampling):
    with rasterio.open(path) as ds:
        return ds.read(1, out_shape=out_shape, resampling=resampling).astype(np.float32)


def fill(a):
    fin = np.isfinite(a)
    return np.where(fin, a, float(a[fin].mean())).astype(np.float32)


def texture_features(base, idx):
    """Sample 12 multi-scale structural features of `base` at flat indices `idx`."""
    f = fill(base)
    cols, names = [], []
    for s in SCALES:
        m = ndimage.uniform_filter(f, s)
        sq = ndimage.uniform_filter(f * f, s)
        std = np.sqrt(np.clip(sq - m * m, 0, None))
        cols += [m.reshape(-1)[idx], std.reshape(-1)[idx]]
        names += [f"mean{s}", f"std{s}"]
    gx = ndimage.gaussian_filter(f, 2, order=(0, 1))
    gy = ndimage.gaussian_filter(f, 2, order=(1, 0))
    for rho in RHOS:
        Jxx = ndimage.gaussian_filter(gx * gx, rho)
        Jyy = ndimage.gaussian_filter(gy * gy, rho)
        Jxy = ndimage.gaussian_filter(gx * gy, rho)
        energy = Jxx + Jyy
        coh = ((Jxx - Jyy) ** 2 + 4 * Jxy * Jxy) / (energy * energy + EPS)
        cols += [energy.reshape(-1)[idx], coh.reshape(-1)[idx]]
        names += [f"energy{rho}", f"coher{rho}"]
    return cols, names


def cv_r2(X, y, groups, seed):
    gkf = GroupKFold(n_splits=5)
    r2s = []
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.1, max_iter=250, random_state=seed)
        m.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--subsample", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT_FOLDER / "results/multispectral")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    cfg = KSAAlignedConfig.from_yaml()
    hr_path = cfg.hr_product_path(cfg.hr_products[0])
    ms_dir = cfg.root / "02_snap_owned" / "multispectral" / "cubicspline"
    with rasterio.open(hr_path) as ds:
        H, W = ds.height, ds.width
    with rasterio.open(cfg.lr_product_path("RTP")) as ds:
        lrH, lrW = ds.height, ds.width
    d = max(1, args.decimate)
    out, lr_out = (H // d, W // d), (lrH // d, lrW // d)
    log(f"grid {out} ({60*d} m/px)", t0)

    hr = read_resampled(hr_path, out, Resampling.average if d > 1 else Resampling.nearest)
    valid = np.isfinite(hr)
    hr_lo = fill(read_resampled(hr_path, lr_out, Resampling.average))
    hr_loup = ndimage.zoom(hr_lo, (out[0] / lr_out[0], out[1] / lr_out[1]), order=3).astype(np.float32)
    d_hp = hr - hr_loup

    b = {i: read_resampled(ms_dir / f"snapped_landsat_b{i}.tif", out, Resampling.average) for i in BANDS}
    for i in BANDS:
        valid &= np.isfinite(b[i])
    log(f"valid {int(valid.sum()):,}", t0)

    # base maps: 3 PCs (span linear band space) + 2 nonlinear ratios
    Braw = np.stack([b[i][valid] for i in BANDS], axis=1)
    mu, sd = Braw.mean(0), Braw.std(0) + EPS
    _, _, Vt = np.linalg.svd((Braw - mu) / sd - ((Braw - mu) / sd).mean(0), full_matrices=False)
    zf = (np.stack([b[i] for i in BANDS], axis=-1) - mu) / sd
    bases = {f"PC{k+1}": (zf @ Vt[k]).astype(np.float32) for k in range(3)}
    bases["Ferrous"] = (b[6] / (b[5] + EPS)).astype(np.float32)
    bases["Clay"] = (b[6] / (b[7] + EPS)).astype(np.float32)
    del zf

    # regular-lattice subsample over the valid set
    vidx = np.flatnonzero(valid)
    take = rng.choice(vidx.size, min(args.subsample, vidx.size), replace=False)
    idx = vidx[take]
    rr, cc = np.unravel_index(idx, out)
    nb = 6
    groups = (rr * nb // out[0]) * nb + (cc * nb // out[1])
    y_hp = d_hp.reshape(-1)[idx].astype(np.float32)
    y_hr = hr.reshape(-1)[idx].astype(np.float32)

    # pixelwise baseline = the 5 base values at the centre pixel
    base_px = np.stack([bases[n].reshape(-1)[idx] for n in bases], axis=1).astype(np.float32)
    # full structural feature bank
    feat_cols, feat_names = [], []
    for n, base in bases.items():
        log(f"features: {n}", t0)
        cols, names = texture_features(base, idx)
        feat_cols += cols
        feat_names += [f"{n}_{x}" for x in names]
    Xtex = np.stack(feat_cols, axis=1).astype(np.float32)
    log(f"feature matrix {Xtex.shape}", t0)

    results = {}
    for tgt, y in [("d_hp", y_hp), ("HR", y_hr)]:
        r2_px = cv_r2(base_px, y, groups, args.seed)
        r2_tex = cv_r2(Xtex, y, groups, args.seed)
        results[tgt] = (r2_px, r2_tex)
        log(
            f"  {tgt}: pixelwise R2={r2_px[0]:.4f}+/-{r2_px[1]:.4f}   "
            f"MULTI-SCALE TEXTURE R2={r2_tex[0]:.4f}+/-{r2_tex[1]:.4f}   "
            f"lift={r2_tex[0]-r2_px[0]:+.4f}",
            t0,
        )

    with open(args.out_dir / "texture_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "pixelwise_r2", "pixelwise_std", "texture_r2", "texture_std", "lift"])
        for tgt, (px, tex) in results.items():
            w.writerow(
                [
                    tgt,
                    f"{px[0]:.5f}",
                    f"{px[1]:.5f}",
                    f"{tex[0]:.5f}",
                    f"{tex[1]:.5f}",
                    f"{tex[0]-px[0]:.5f}",
                ]
            )
    print("\n=== VERDICT ===")
    for tgt, (px, tex) in results.items():
        verdict = (
            "STRUCTURE HELPS — try a CNN"
            if (tex[0] - px[0]) > 0.02 and tex[0] > 0.03
            else "no meaningful lift"
        )
        print(f"{tgt:5}: pixelwise {px[0]:+.4f} -> texture {tex[0]:+.4f}  ({verdict})")
    log(f"wrote {args.out_dir/'texture_probe.csv'}", t0)


if __name__ == "__main__":
    main()

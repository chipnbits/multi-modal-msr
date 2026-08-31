"""TARGET-ORIENTED spectral decomposition: the SR target is the LR->HR diff
`D = HR - upsample(LR)` (the information SR must synthesize). Which spectral
components explain the most of D?

Two decompositions over the full raster (60 m grid, valid survey footprint):
  1. Unsupervised band PCA, then rank PCs by the fraction of D they explain
     (R²(D ~ PC_k) and cumulative PCR). This answers your literal question:
     "which PC components explain the most of the target."
  2. PLS (partial least squares) bands -> D: the SUPERVISED optimum. PLS component 1
     is, by construction, the single band combination that covaries most with the
     target — i.e. the best possible HR-free 1-channel "spectral detail index" to feed
     the SR net. Cumulative spatial-CV R² over PLS components is the ceiling on how
     much of D any spectral input can explain.

Targets reported BOTH ways:
  * D_real = HR - bicubic(LR)  (the literal LR->HR diff; ~30% of HR variance, but
    includes product leveling/processing mismatch between the 60 m and 180 m grids)
  * D_hp   = HR - up3(down3(HR))  (the clean 60-180 m detail octave, calibration-free)

A few fold-3 TEST patches are rendered: D next to the PLS-1 index and the best PC, so
you can see whether the best spectral component actually looks like the target.

Usage:
    uv run --with scikit-learn python experiments/multispectral/target_pc_components.py
    uv run --with scikit-learn python experiments/multispectral/target_pc_components.py --decimate 4
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from experiments._archive.multispectral.inspect_multispectral import (
    BAND_NAMES,
    BANDS,
    band_path,
    pool2x2_nanmean,
    robust_clim,
)
from magsr import ROOT_FOLDER
from magsr.datasets import KSAAlignedConfig
from magsr.datasets.io import open_raster
from magsr.datasets.patching import PatchIndex

EPS = 1e-3


def log(m, t0):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def read_resampled(path, out_shape, resampling):
    with rasterio.open(path) as ds:
        return ds.read(1, out_shape=out_shape, resampling=resampling).astype(np.float32)


def cv_r2_model(make_model, X, y, groups):
    gkf = GroupKFold(n_splits=5)
    r2s = []
    for tr, te in gkf.split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], m.predict(X[te]).ravel()))
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decimate", type=int, default=1)
    p.add_argument("--subsample", type=int, default=2_000_000)
    p.add_argument("--n-panels", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT_FOLDER / "results/multispectral")
    p.add_argument("--fig-dir", type=Path, default=ROOT_FOLDER / "figures/multispectral")
    p.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT_FOLDER / "data/processed/ksa_aligned/patch_indices_cellgrid8_fold3",
    )
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    cfg = KSAAlignedConfig.from_yaml()
    hr_path = cfg.hr_product_path(cfg.hr_products[0])
    lr_path = cfg.lr_product_path("RTP")
    ms_dir = cfg.root / "02_snap_owned" / "multispectral" / "cubicspline"
    with rasterio.open(hr_path) as ds:
        H, W = ds.height, ds.width
    with rasterio.open(lr_path) as ds:
        lrH, lrW = ds.height, ds.width
    d = max(1, args.decimate)
    out, lr_out = (H // d, W // d), (lrH // d, lrW // d)
    log(f"grid {out} ({60*d} m/px)", t0)

    hr = read_resampled(hr_path, out, Resampling.average if d > 1 else Resampling.nearest)
    valid = np.isfinite(hr)
    lrup = read_resampled(lr_path, out, Resampling.cubic)
    hr_lo = read_resampled(hr_path, lr_out, Resampling.average)
    hr_lo = np.where(np.isfinite(hr_lo), hr_lo, np.nanmean(hr_lo)).astype(np.float32)
    hr_loup = ndimage.zoom(hr_lo, (out[0] / lr_out[0], out[1] / lr_out[1]), order=3).astype(np.float32)
    D_real = hr - lrup
    D_hp = hr - hr_loup
    valid &= np.isfinite(lrup)

    bands = {
        i: read_resampled(ms_dir / f"snapped_landsat_b{i}.tif", out, Resampling.average) for i in BANDS
    }
    for i in BANDS:
        valid &= np.isfinite(bands[i])
    log(
        f"valid {int(valid.sum()):,}; var(D_real)/var(HR)={np.var(D_real[valid])/np.var(hr[valid]):.3f} "
        f"var(D_hp)/var(HR)={np.var(D_hp[valid])/np.var(hr[valid]):.3f}",
        t0,
    )

    # standardize bands over valid
    Bv = np.stack([bands[i][valid] for i in BANDS], axis=1)
    mu, sd = Bv.mean(0), Bv.std(0) + EPS
    Z = (Bv - mu) / sd  # (Nvalid, 7) z-scored bands

    # subsample with spatial groups
    rr, cc = np.where(valid)
    nb = 6
    grp_all = (rr * nb // out[0]) * nb + (cc * nb // out[1])
    idx = rng.choice(Z.shape[0], min(args.subsample, Z.shape[0]), replace=False)
    Zs, gs = Z[idx], grp_all[idx]
    Dr, Dh = D_real[valid][idx], D_hp[valid][idx]

    # ---- 1. unsupervised PCA, then rank PCs by variance of D explained ----
    _, S, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    evr = (S**2) / (S**2).sum()
    PCs = Zs @ Vt.T  # (n,7) PC scores
    rows = []
    for k in range(7):
        rows.append(
            {
                "component": f"PC{k+1}",
                "band_evr": float(evr[k]),
                "r2_Dreal": float(np.corrcoef(PCs[:, k], Dr)[0, 1] ** 2),
                "r2_Dhp": float(np.corrcoef(PCs[:, k], Dh)[0, 1] ** 2),
                "loadings": " ".join(f"b{BANDS[i]}={Vt[k,i]:+.2f}" for i in range(7)),
            }
        )

    # ---- 2. PLS bands->D: supervised components, cumulative spatial-CV R² ----
    def pls_curve(y):
        cum = []
        for nc in range(1, 8):
            cum.append(cv_r2_model(lambda nc=nc: PLSRegression(n_components=nc), Zs, y, gs))
        return cum

    log("PLS curves...", t0)
    pls_real, pls_hp = pls_curve(Dr), pls_curve(Dh)
    # PLS-1 spectral index weights (in z-scored band space) for deployment
    pls1 = PLSRegression(n_components=1).fit(Zs, Dh)
    w1 = pls1.x_weights_[:, 0]
    # linear all-band ceiling for reference
    lin_real = cv_r2_model(LinearRegression, Zs, Dr, gs)
    lin_hp = cv_r2_model(LinearRegression, Zs, Dh, gs)

    # ---- report ----
    print("\n=== Which spectral components explain the LR->HR diff D? ===")
    print("(1) Unsupervised band-PCA, ranked by fraction of D explained:")
    print(f"{'comp':>5} {'band_EVR':>9} {'R2(D_real)':>11} {'R2(D_hp)':>9}   loadings")
    for r in sorted(rows, key=lambda r: r["r2_Dhp"], reverse=True):
        print(
            f"{r['component']:>5} {r['band_evr']:9.3f} {r['r2_Dreal']:11.4f} {r['r2_Dhp']:9.4f}   {r['loadings']}"
        )
    print("\n(2) PLS bands->D (SUPERVISED best components), cumulative spatial-CV R²:")
    print(f"{'#comp':>5} {'R2(D_real)':>14} {'R2(D_hp)':>14}")
    for nc in range(7):
        print(
            f"{nc+1:>5} {pls_real[nc][0]:8.4f}+/-{pls_real[nc][1]:.4f} {pls_hp[nc][0]:8.4f}+/-{pls_hp[nc][1]:.4f}"
        )
    print(f"\nlinear all-7-band ceiling:  R2(D_real)={lin_real[0]:.4f}  R2(D_hp)={lin_hp[0]:.4f}")
    print(
        "PLS-1 'spectral detail index' weights (z-band space): "
        + " ".join(f"b{BANDS[i]}({BAND_NAMES[BANDS[i]]})={w1[i]:+.2f}" for i in range(7))
    )

    with open(args.out_dir / "target_pc_components.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "component", "band_evr", "r2_Dreal", "r2_Dhp", "detail"])
        for r in rows:
            w.writerow(
                [
                    "PCA",
                    r["component"],
                    f"{r['band_evr']:.5f}",
                    f"{r['r2_Dreal']:.5f}",
                    f"{r['r2_Dhp']:.5f}",
                    r["loadings"],
                ]
            )
        for nc in range(7):
            w.writerow(
                ["PLS_cumCV", f"{nc+1}comp", "", f"{pls_real[nc][0]:.5f}", f"{pls_hp[nc][0]:.5f}", ""]
            )
        w.writerow(["ceiling", "linear7", "", f"{lin_real[0]:.5f}", f"{lin_hp[0]:.5f}", ""])
    log("wrote target_pc_components.csv", t0)

    # ---- figure: R²(D) per PC and PLS curve ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    order = sorted(range(7), key=lambda k: rows[k]["r2_Dhp"], reverse=True)
    a1.bar(
        [f"PC{k+1}" for k in order],
        [rows[k]["r2_Dhp"] for k in order],
        color="tab:purple",
        label="R²(D_hp)",
    )
    a1.bar(
        [f"PC{k+1}" for k in order],
        [rows[k]["r2_Dreal"] for k in order],
        color="tab:orange",
        alpha=0.5,
        label="R²(D_real)",
    )
    a1.set_ylabel("fraction of target D explained")
    a1.set_title("Band-PCA components vs SR target")
    a1.legend()
    a2.plot(range(1, 8), [v[0] for v in pls_hp], "o-", label="PLS R²(D_hp)")
    a2.plot(range(1, 8), [v[0] for v in pls_real], "s-", label="PLS R²(D_real)")
    a2.set_xlabel("# PLS components")
    a2.set_ylabel("spatial-CV R²(D)")
    a2.set_title("Supervised ceiling: spectral -> SR target")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "target_pc_components.png", dpi=130)
    plt.close(fig)

    # ---- test-patch panels: D vs PLS-1 index vs best PC ----
    best_pc = max(range(7), key=lambda k: rows[k]["r2_Dhp"])
    hr_src = open_raster(hr_path)
    lr_src = open_raster(lr_path)
    bsrc = {i: open_raster(band_path(cfg, i)) for i in BANDS}
    pidx = PatchIndex.load(args.index_dir / "test.json")
    Pp = pidx.spec.patch_px
    pats = list(pidx.patches)
    rng.shuffle(pats)
    for n, t in enumerate(pats[: args.n_panels]):
        hrp = hr_src.read_window(t.row_px, t.col_px, Pp, Pp)
        lrp = lr_src.read_window(
            t.row_px // cfg.lr_scale, t.col_px // cfg.lr_scale, Pp // cfg.lr_scale, Pp // cfg.lr_scale
        )
        lrpu = np.asarray(
            ndimage.zoom(np.nan_to_num(lrp, nan=float(np.nanmean(lrp))), cfg.lr_scale, order=3)
        )[:Pp, :Pp]
        Dp = hrp - lrpu
        stack = np.stack(
            [
                pool2x2_nanmean(
                    bsrc[i].read_window(
                        t.row_px * cfg.dem_scale,
                        t.col_px * cfg.dem_scale,
                        Pp * cfg.dem_scale,
                        Pp * cfg.dem_scale,
                    )
                )
                for i in BANDS
            ],
            axis=-1,
        )
        Zp = (stack - mu) / sd
        pls1_map = Zp @ w1
        pc_map = Zp @ Vt[best_pc]
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
        for a, img, ttl, cm in [
            (ax[0], hrp, "HR magnetic", "RdBu_r"),
            (ax[1], Dp, "target D = HR - up(LR)", "RdBu_r"),
            (ax[2], pls1_map, "PLS-1 spectral index", "cividis"),
            (ax[3], pc_map, f"best band-PC (PC{best_pc+1})", "cividis"),
        ]:
            lo, hi = robust_clim(img)
            a.imshow(img, cmap=cm, vmin=lo, vmax=hi)
            a.set_title(ttl, fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle(f"test patch {n}  row={t.row_px} col={t.col_px}", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"target_patch_{n:02d}.png", dpi=110)
        plt.close(fig)
    log(f"wrote panels + {args.fig_dir/'target_pc_components.png'}", t0)


if __name__ == "__main__":
    main()

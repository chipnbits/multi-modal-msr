"""Rank the 7 Landsat bands (and alteration indices) by how much they inform
aeromagnetic SUPER-RESOLUTION, computed over the ENTIRE raster — and decide
whether they inform it at all.

Why the earlier patch-based raw-Pearson analysis was inconclusive
-----------------------------------------------------------------
Global Pearson(band, HR field) was ~-0.2 on one fold's patches and the bands are
92-99% collinear (PC1 = 96% var = scene albedo). Two flaws:
  (1) The raw RTP field is dominated by long-wavelength regional / deep-source
      trends surface reflectance cannot sense. SR needs only the SHORT-wavelength
      detail living between the 180 m and 60 m scales, not that.
  (2) One linear univariate number cannot separate 7 collinear channels.

The correct framing
-------------------
A band helps SR iff it carries information about the detail LR lacks. On the 60 m
grid define:
    d_hp   = HR - upsample3( downsample3(HR) )    # scale-consistent 60-180 m detail
                                                   #  (exactly what SR must add)
    d_real = HR - cubic_upsample(real LR)          # operational SR residual
and score channels with a battery robust to nonlinearity, collinearity and spatial
autocorrelation:
  * univariate   : Pearson + Spearman + mutual information
  * structural   : Pearson(|grad band|, |grad HR|)
  * MULTIVARIATE : linear incremental R2 (drop-one) + a nonlinear
                   HistGradientBoosting model with spatial-block-CV permutation
                   importance  <-- PRIMARY (only thing that gives non-redundant
                   credit under heavy collinearity).
Each metric is computed against BOTH the SR detail (d_hp) and the RAW field (HR),
so we separate "tracks the magnetic geology" from "helps recover SR detail".

Guards an adversarial reviewer would demand (all reported):
  * positive-control anchors: R2(HR ~ LRup) (must be ~1), R2(HR ~ bands) vs
    R2(d_hp ~ bands) (regional vs detail spectral signal);
  * variance accounting: var(d_hp)/var(HR), var(d_real)/var(HR);
  * a multi-scale sweep: R2(detail ~ bands) and R2(smooth ~ bands) vs low-pass
    cutoff, to locate at WHAT scale (if any) spectral information about the
    magnetics lives;
  * a registration offset check (|grad band| vs |grad HR| cross-correlation), so a
    null can't be blamed on sub-pixel misalignment.

Usage:
    uv run --with scikit-learn python experiments/multispectral/rank_bands_fullraster.py
    uv run --with scikit-learn python experiments/multispectral/rank_bands_fullraster.py --decimate 4   # fast (coarser scale!)
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from magsr import ROOT_FOLDER
from magsr.datasets import KSAAlignedConfig

BAND_NAMES = {1: "coastal", 2: "blue", 3: "green", 4: "red", 5: "nir", 6: "swir1", 7: "swir2"}
BANDS = tuple(BAND_NAMES)
EPS = 1e-3
HP_W = 5  # high-pass window (px); 5*60 m = 300 m, isolates the ~SR octave


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def read_resampled(path: Path, out_shape: tuple[int, int], resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1, out_shape=out_shape, resampling=resampling).astype(np.float32)


def grad_mag(a: np.ndarray, fill: float) -> np.ndarray:
    f = np.where(np.isfinite(a), a, fill).astype(np.float32)
    gx = ndimage.sobel(f, axis=1, mode="nearest")
    gy = ndimage.sobel(f, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def highpass(a: np.ndarray, w: int) -> np.ndarray:
    """Spatial high-pass a - uniform_filter(a, w), NaN-safe (fill with valid mean
    before filtering, then restore NaN). Isolates structure finer than ~w pixels."""
    fin = np.isfinite(a)
    f = np.where(fin, a, float(a[fin].mean())).astype(np.float32)
    hp = a - ndimage.uniform_filter(f, size=w)
    return np.where(fin, hp, np.nan).astype(np.float32)


def ridge_cv_r2(X, y, groups, alpha, seed):
    """Spatial-block-CV R2 of a standardized ridge y ~ X (the best HR-free LINEAR
    7->1 compression; weights are constants fit on HR pixels, applied everywhere)."""
    fin = np.isfinite(y)
    X, y, groups = X[fin], y[fin], groups[fin]
    from sklearn.linear_model import Ridge

    gkf = GroupKFold(n_splits=5)
    r2s = []
    for tr, te in gkf.split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + EPS
        m = Ridge(alpha=alpha, random_state=seed).fit((X[tr] - mu) / sd, y[tr])
        r2s.append(r2_score(y[te], m.predict((X[te] - mu) / sd)))
    return float(np.mean(r2s)), float(np.std(r2s))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size > n:
        idx = rng.choice(x.size, n, replace=False)
        x, y = x[idx], y[idx]
    if x.size < 8:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="Extra integer downsample of the 60 m grid (1 = full res; >1 changes the detail SCALE).",
    )
    p.add_argument("--mi-subsample", type=int, default=200_000)
    p.add_argument("--spearman-subsample", type=int, default=5_000_000)
    p.add_argument("--model-subsample", type=int, default=2_000_000)
    p.add_argument("--perm-eval", type=int, default=200_000)
    p.add_argument("--sweep-subsample", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT_FOLDER / "results/multispectral")
    p.add_argument("--fig-dir", type=Path, default=ROOT_FOLDER / "figures/multispectral")
    return p.parse_args()


def nonlinear_skill_and_importance(X, y, groups, raw_names, rng, perm_eval, seed):
    """Spatial-block-CV R2 and permutation importance of a HistGBR model y ~ X."""
    gkf = GroupKFold(n_splits=5)
    perm_acc = np.zeros(X.shape[1])
    cv_r2 = []
    for tr, te in gkf.split(X, y, groups):
        model = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.1, max_iter=200, random_state=seed
        )
        model.fit(X[tr], y[tr])
        cv_r2.append(r2_score(y[te], model.predict(X[te])))
        ev = te if te.size <= perm_eval else rng.choice(te, perm_eval, replace=False)
        pi = permutation_importance(model, X[ev], y[ev], n_repeats=3, random_state=seed, n_jobs=-1)
        perm_acc += pi.importances_mean
    return float(np.mean(cv_r2)), float(np.std(cv_r2)), perm_acc / gkf.get_n_splits()


def main() -> None:
    args = parse_args()
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
    px_m = 60 * d
    log(f"grids: HR {H}x{W} -> work {out} ({px_m} m/px); LR {lrH}x{lrW}", t0)

    # ---- read HR, LR, targets ----
    hr = read_resampled(hr_path, out, Resampling.average if d > 1 else Resampling.nearest)
    valid = np.isfinite(hr)
    hr_fill = float(hr[valid].mean())
    lrup = read_resampled(lr_path, out, Resampling.cubic)
    hr_lo = read_resampled(hr_path, lr_out, Resampling.average)
    hr_lo = np.where(np.isfinite(hr_lo), hr_lo, np.nanmean(hr_lo)).astype(np.float32)
    hr_loup = ndimage.zoom(hr_lo, (out[0] / lr_out[0], out[1] / lr_out[1]), order=3).astype(np.float32)
    d_hp = hr - hr_loup
    d_real = hr - lrup
    log(f"valid HR pixels: {int(valid.sum()):,} ({valid.mean():.1%})", t0)

    # ---- bands at working grid ----
    bands = {
        i: read_resampled(ms_dir / f"snapped_landsat_b{i}.tif", out, Resampling.average) for i in BANDS
    }
    for i in BANDS:
        valid &= np.isfinite(bands[i])
    # on-disk analytic-signal amplitude: polarity-free magnetic EDGE target (peaks
    # over geologic contacts at SR depths), the correct target for structural corr.
    ans = read_resampled(
        cfg.hr_dir / "snapped_cubicspline_MAG_AMF_ANS.tif",
        out,
        Resampling.average if d > 1 else Resampling.nearest,
    )
    valid &= np.isfinite(ans)
    log(f"valid after band+ANS intersection: {int(valid.sum()):,}", t0)
    b = bands

    # ---- candidate channels ----
    channels: dict[str, np.ndarray] = {f"b{i}_{BAND_NAMES[i]}": b[i] for i in BANDS}
    channels["IronOxide_b4/b2"] = b[4] / (b[2] + EPS)
    channels["Ferrous_b6/b5"] = b[6] / (b[5] + EPS)
    channels["Clay_b6/b7"] = b[6] / (b[7] + EPS)
    channels["NDVI_b5b4"] = (b[5] - b[4]) / (b[5] + b[4] + EPS)
    Braw = np.stack([b[i][valid] for i in BANDS], axis=1)
    mu, sd = Braw.mean(0), Braw.std(0) + EPS
    _, _, Vt = np.linalg.svd((Braw - mu) / sd - ((Braw - mu) / sd).mean(0), full_matrices=False)
    zfull = (np.stack([b[i] for i in BANDS], axis=-1) - mu) / sd
    for k in range(3):
        channels[f"PC{k + 1}"] = (zfull @ Vt[k]).astype(np.float32)
    del zfull
    names = list(channels)
    raw_names = [f"b{i}_{BAND_NAMES[i]}" for i in BANDS]

    # ---- flatten ----
    vv = valid
    y_hp = d_hp[vv].astype(np.float32)
    y_real = d_real[vv].astype(np.float32)
    y_hr = hr[vv].astype(np.float32)
    valid_in = ndimage.binary_erosion(vv, iterations=3)  # kill high-pass/sobel edge bleed
    g_hr_in = grad_mag(hr, hr_fill)[valid_in].astype(np.float32)
    ans_hp_in = highpass(ans, HP_W)[valid_in].astype(np.float32)  # polarity-free edge target
    Nv = y_hp.size

    # ---- variance accounting ----
    var_hr = float(np.var(y_hr))
    print(
        f"\nvariance: HR={var_hr:.3g} nT^2  d_hp/HR={np.var(y_hp)/var_hr:.3f}  "
        f"d_real/HR={np.nanvar(y_real)/var_hr:.3f}"
    )

    # ---- univariate + structural metrics vs d_hp and HR ----
    log("univariate + structural metrics...", t0)
    rows = []
    for name in names:
        cv = channels[name][vv].astype(np.float32)
        chp = highpass(channels[name], HP_W)[vv].astype(np.float32)  # octave-isolated channel
        gc = grad_mag(channels[name], float(np.nanmean(channels[name][vv])))[valid_in].astype(np.float32)
        rows.append(
            {
                "channel": name,
                # REPAIRED univariate: high-pass channel vs high-pass detail (defeats albedo+regional trend)
                "rbp_dhp": pearson(chp, y_hp),
                "rbp_dreal": pearson(chp, y_real),
                "pearson_dhp": pearson(cv, y_hp),
                "spearman_dhp": spearman(cv, y_hp, rng, args.spearman_subsample),
                "struct_edge": pearson(gc, g_hr_in),
                "edge_ans": pearson(gc, ans_hp_in),
                "pearson_dreal": pearson(cv, y_real),
                "pearson_hr": pearson(cv, y_hr),
                "spearman_hr": spearman(cv, y_hr, rng, args.spearman_subsample),
            }
        )

    # ---- mutual information vs d_hp and HR ----
    log("mutual information...", t0)
    mi_idx = rng.choice(Nv, min(args.mi_subsample, Nv), replace=False)
    Xmi = np.stack([channels[n][vv][mi_idx] for n in names], axis=1)
    mi_hp = mutual_info_regression(Xmi, y_hp[mi_idx], random_state=args.seed)
    mi_hr = mutual_info_regression(Xmi, y_hr[mi_idx], random_state=args.seed)
    for r, a, c in zip(rows, mi_hp, mi_hr):
        r["mi_dhp"], r["mi_hr"] = float(a), float(c)

    # ---- shared subsample for the multivariate models ----
    rr, cc = np.where(vv)
    nb = 6
    groups_full = (rr * nb // out[0]) * nb + (cc * nb // out[1])
    sub = rng.choice(Nv, min(args.model_subsample, Nv), replace=False)
    g_sub = groups_full[sub]
    Xraw = np.stack([channels[n][vv] for n in raw_names], axis=1).astype(np.float32)[sub]
    LRsub = lrup[vv][sub].astype(np.float32)

    # ---- anchors ----
    log("anchors (positive controls)...", t0)
    anchors = {}
    fin = np.isfinite(LRsub)
    anchors["R2(HR ~ LRup) linear"] = float(
        r2_score(
            y_hr[sub][fin],
            LinearRegression().fit(LRsub[fin, None], y_hr[sub][fin]).predict(LRsub[fin, None]),
        )
    )
    for tgt, yt in [("HR", y_hr[sub]), ("d_hp", y_hp[sub])]:
        lin = LinearRegression().fit(Xraw, yt)
        anchors[f"R2({tgt} ~ 7bands) linear"] = float(r2_score(yt, lin.predict(Xraw)))
    for k, v in anchors.items():
        print(f"  anchor {k:28} = {v:.4f}")

    # ---- linear incremental R2 (drop-one) on d_hp and HR ----
    log("linear incremental R2...", t0)

    def incremental(yt):
        full = r2_score(yt, LinearRegression().fit(Xraw, yt).predict(Xraw))
        out_inc = {}
        for j, n in enumerate(raw_names):
            keep = [k for k in range(len(raw_names)) if k != j]
            out_inc[n] = float(
                full - r2_score(yt, LinearRegression().fit(Xraw[:, keep], yt).predict(Xraw[:, keep]))
            )
        return out_inc

    inc_hp, inc_hr = incremental(y_hp[sub]), incremental(y_hr[sub])

    # ---- nonlinear model + spatial-CV permutation importance, d_hp and HR ----
    log("nonlinear spatial-CV model: d_hp...", t0)
    r2_hp, r2_hp_sd, perm_hp = nonlinear_skill_and_importance(
        Xraw, y_hp[sub], g_sub, raw_names, rng, args.perm_eval, args.seed
    )
    log(f"  nonlinear spatial-CV R2(d_hp ~ 7bands) = {r2_hp:.4f} +/- {r2_hp_sd:.4f}", t0)
    log("nonlinear spatial-CV model: HR...", t0)
    r2_hrm, r2_hr_sd, perm_hr = nonlinear_skill_and_importance(
        Xraw, y_hr[sub], g_sub, raw_names, rng, args.perm_eval, args.seed
    )
    log(f"  nonlinear spatial-CV R2(HR ~ 7bands) = {r2_hrm:.4f} +/- {r2_hr_sd:.4f}", t0)

    # ---- ridge "spectral detail index": best HR-free LINEAR 7->1 compression ----
    log("ridge spectral-detail-index (HR-free 7->1 compression)...", t0)
    Xbp = np.stack([highpass(channels[n], HP_W)[vv] for n in raw_names], axis=1).astype(np.float32)[sub]
    comp = {
        "ridge raw-bands -> d_hp": ridge_cv_r2(Xraw, y_hp[sub], g_sub, 1.0, args.seed),
        "ridge raw-bands -> d_real": ridge_cv_r2(Xraw, y_real[sub], g_sub, 1.0, args.seed),
        "ridge highpass-bands -> d_hp": ridge_cv_r2(Xbp, y_hp[sub], g_sub, 1.0, args.seed),
        "ridge raw-bands -> HR": ridge_cv_r2(Xraw, y_hr[sub], g_sub, 1.0, args.seed),
    }
    for k, (m, s) in comp.items():
        print(f"  compression {k:32} spatial-CV R2 = {m:.4f} +/- {s:.4f}")
    with open(args.out_dir / "compression.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "cv_r2_mean", "cv_r2_std"])
        w.writerows([[k, f"{m:.5f}", f"{s:.5f}"] for k, (m, s) in comp.items()])

    for r in rows:
        n = r["channel"]
        r["lin_inc_dhp"] = inc_hp.get(n, float("nan"))
        r["lin_inc_hr"] = inc_hr.get(n, float("nan"))
        r["perm_dhp"] = float(perm_hp[raw_names.index(n)]) if n in raw_names else float("nan")
        r["perm_hr"] = float(perm_hr[raw_names.index(n)]) if n in raw_names else float("nan")

    # ---- multi-scale sweep: where does spectral info about the field live? ----
    log("multi-scale sweep...", t0)
    sweep_idx = rng.choice(Nv, min(args.sweep_subsample, Nv), replace=False)
    Xsw = np.stack([channels[n][vv][sweep_idx] for n in raw_names], axis=1).astype(np.float32)
    sigmas_m = [60, 120, 240, 480, 960, 1920, 3840]
    sweep = []
    for s_m in sigmas_m:
        s_px = max(0.5, s_m / px_m)
        smooth = ndimage.gaussian_filter(np.where(vv, hr, hr_fill).astype(np.float32), s_px)
        detail = hr - smooth
        yd, ys = detail[vv][sweep_idx], smooth[vv][sweep_idx]
        r2d = r2_score(yd, LinearRegression().fit(Xsw, yd).predict(Xsw))
        r2s = r2_score(ys, LinearRegression().fit(Xsw, ys).predict(Xsw))
        sweep.append((s_m, r2d, r2s, float(np.var(yd) / var_hr)))
        log(
            f"  sigma {s_m:5d} m: R2(detail~bands)={r2d:.4f}  R2(smooth~bands)={r2s:.4f}  var(detail)/var(HR)={np.var(yd)/var_hr:.3f}",
            t0,
        )

    # ---- registration offset check (|grad b6| vs |grad HR|) ----
    log("registration offset check...", t0)
    ys_, xs_ = np.where(valid_in)
    cy, cx = int(np.median(ys_)), int(np.median(xs_))
    half = min(512, cy, cx, out[0] - cy - 1, out[1] - cx - 1)
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    A = grad_mag(hr, hr_fill)[sl]
    B = grad_mag(b[6], float(np.nanmean(b[6][vv])))[sl]
    A = (A - A.mean()) / (A.std() + EPS)
    B = (B - B.mean()) / (B.std() + EPS)
    best, best_off = -2.0, (0, 0)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            r = float(np.mean(A * np.roll(np.roll(B, dy, 0), dx, 1)))
            if r > best:
                best, best_off = r, (dy, dx)
    print(
        f"  registration: peak |grad| NCC={best:.3f} at offset (dy,dx)={best_off} px  "
        f"({'ALIGNED' if best_off == (0, 0) else 'SHIFT — investigate'})"
    )

    # ---- write CSV ----
    fields = [
        "channel",
        "perm_dhp",
        "lin_inc_dhp",
        "rbp_dhp",
        "rbp_dreal",
        "mi_dhp",
        "spearman_dhp",
        "pearson_dhp",
        "struct_edge",
        "edge_ans",
        "pearson_dreal",
        "perm_hr",
        "mi_hr",
        "spearman_hr",
        "pearson_hr",
    ]
    csv_path = args.out_dir / "band_ranking_fullraster.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: (f"{r[k]:.5f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in fields}
            )
    with open(args.out_dir / "scale_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_m", "r2_detail_bands", "r2_smooth_bands", "var_detail_frac"])
        w.writerows([[s, f"{a:.5f}", f"{c:.5f}", f"{v:.5f}"] for s, a, c, v in sweep])
    log(f"wrote {csv_path}", t0)

    # ---- print rankings ----
    def show(title, key, absval, raw_only, cols):
        items = [r for r in rows if not (raw_only and r["channel"] not in raw_names)]
        items = [r for r in items if isinstance(r.get(key), float) and np.isfinite(r[key])]
        items.sort(key=lambda r: (abs(r[key]) if absval else r[key]), reverse=True)
        print("\n" + title)
        print("rank  " + "channel".ljust(22) + " ".join(c.rjust(9) for c, _ in cols))
        for i, r in enumerate(items, 1):
            print(f"{i:>4}  {r['channel']:22} " + " ".join(f"{f(r):9.4f}" for _, f in cols))

    print("\n" + "=" * 84)
    print(
        f"SR-detail target d_hp ({px_m}-{px_m*3} m octave)  —  nonlinear spatial-CV R2(d_hp~bands) = {r2_hp:.4f}"
    )
    print("=" * 84)
    show(
        "PRIMARY: marginal value for SR detail (perm importance on d_hp, raw bands)",
        "perm_dhp",
        False,
        True,
        [
            ("perm", lambda r: r["perm_dhp"]),
            ("incR2", lambda r: r["lin_inc_dhp"]),
            ("MI", lambda r: r["mi_dhp"]),
            ("|rho|", lambda r: abs(r["spearman_dhp"])),
            ("edge", lambda r: r["struct_edge"]),
        ],
    )
    show(
        "REPAIRED univariate: octave band-pass channel vs SR detail (|r_bp|, all candidates)",
        "rbp_dhp",
        True,
        False,
        [
            ("|rbp|dhp", lambda r: abs(r["rbp_dhp"])),
            ("|rbp|dreal", lambda r: abs(r["rbp_dreal"])),
            ("edgeANS", lambda r: abs(r["edge_ans"])),
            ("MI", lambda r: r["mi_dhp"]),
        ],
    )
    show(
        "Tracks the RAW magnetic field (perm importance on HR, raw bands)",
        "perm_hr",
        False,
        True,
        [
            ("perm", lambda r: r["perm_hr"]),
            ("MI", lambda r: r["mi_hr"]),
            ("|rho|", lambda r: abs(r["spearman_hr"])),
            ("|r|", lambda r: abs(r["pearson_hr"])),
        ],
    )
    show(
        "ALL candidates vs RAW field by |Spearman| (incl. ratios & PCs)",
        "spearman_hr",
        True,
        False,
        [
            ("|rho|hr", lambda r: abs(r["spearman_hr"])),
            ("MIhr", lambda r: r["mi_hr"]),
            ("edge", lambda r: r["struct_edge"]),
            ("|r|hr", lambda r: abs(r["pearson_hr"])),
        ],
    )

    # ---- figures ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = [
        ("perm_dhp", "PRIMARY: perm importance on d_hp (SR detail)", True),
        ("perm_hr", "perm importance on RAW field HR", True),
        ("spearman_hr", "Spearman with RAW field (all candidates)", False),
        ("struct_edge", "structural edge corr |grad c| vs |grad HR|", False),
    ]
    for ax, (key, title, raw_only) in zip(axes.ravel(), panels):
        items = [r for r in rows if not (raw_only and r["channel"] not in raw_names)]
        items = [r for r in items if isinstance(r.get(key), float) and np.isfinite(r[key])]
        items.sort(key=lambda r: abs(r[key]), reverse=True)
        ax.barh(range(len(items)), [r[key] for r in items], color="tab:purple")
        ax.set_yticks(range(len(items)), [r["channel"] for r in items], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Spectral-channel ranking for aeromagnetic SR (full raster, {px_m} m px)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "band_ranking_fullraster.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sm = [s for s, _, _, _ in sweep]
    ax.plot(sm, [a for _, a, _, _ in sweep], "o-", label="R2(detail ~ bands)")
    ax.plot(sm, [c for _, _, c, _ in sweep], "s-", label="R2(smooth ~ bands)")
    ax.set_xscale("log")
    ax.set_xlabel("Gaussian low-pass sigma (m)")
    ax.set_ylabel("linear R2")
    ax.set_title("At what scale do spectral bands explain the magnetic field?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "scale_sweep.png", dpi=130)
    plt.close(fig)
    log(f"wrote figures to {args.fig_dir}", t0)


if __name__ == "__main__":
    main()

"""Starter inspection of the 7 Landsat multispectral bands, aligned to the HR
magnetic survey grid.

The bands live at `02_snap_owned/multispectral/cubicspline/snapped_landsat_b{1..7}.tif`
on the canonical 30 m grid — the *same* grid as the DEM, sharing the exact origin
with the 60 m HR magnetic raster and exactly 2x finer (`ms_scale = 2`, identical to
`config.dem_scale`). So an HR patch at `(row_px, col_px)` of side `patch_px` (60 m)
maps to multispectral window `(row_px*2, col_px*2)` of side `patch_px*2` (30 m).

Goal this serves: decide which band(s) carry the most information about the HR
magnetic signal so spectral channels can be fed into LR->HR super-resolution
*without overwhelming the model with all 7*. This script gives the first evidence:

  1. Visual panels (HR magnetic | b1..b7 | true-color RGB) to eyeball co-registration.
  2. Inter-band Pearson correlation matrix (redundancy among the 7 bands).
  3. Band <-> HR-magnetic correlation, ranked (the "which bands inform HR" answer).
  4. PCA compression sniff-test: can the 7 bands collapse into ~1 layer, and does
     that layer track HR? PCA is unsupervised, so the transform needs no HR at
     inference (works where HR is unavailable) — a key requirement.

Patches are drawn from the canonical fold-3 benchmark index so every window is
in-survey and valid, matching the existing experiments.

Reuses `magsr.datasets.io.open_raster` / `RasterSource.read_window`,
`magsr.datasets.patching.PatchIndex`, and `KSAAlignedConfig` — no shared library
code is modified. Once a winner band/component is picked it can be promoted into
`KSAAlignedConfig`/`KSAShieldAlignedDataset` as an `ms_*` modality (mirroring the
DEM / `lr_aux` machinery).

Next steps (not built here, to keep the starter lean):
  - SR-residual correlation: correlate each band with the HR *detail*
    `HR - bicubic_upsample(LR)` (the high-frequency content LR lacks and SR must
    recover), via `cfg.lr_product_path("RTP")`. This is the most SR-relevant metric.
  - Supervised projection: a linear map 7 bands -> 1 channel fit on HR-available
    cells and applied everywhere (transferable, HR-free at inference).

Usage:
    uv run python experiments/multispectral/inspect_multispectral.py
    uv run python experiments/multispectral/inspect_multispectral.py --n-panels 4 --n-stats 150
"""

from __future__ import annotations

import argparse
import csv
import random
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magsr import ROOT_FOLDER
from magsr.datasets import KSAAlignedConfig
from magsr.datasets.io import clear_source_cache, open_raster
from magsr.datasets.patching import PatchIndex

# Landsat 8/9 band -> short name (b1..b7 cover coastal/aerosol through SWIR-2).
BAND_NAMES: dict[int, str] = {
    1: "coastal",
    2: "blue",
    3: "green",
    4: "red",
    5: "nir",
    6: "swir1",
    7: "swir2",
}
BANDS: tuple[int, ...] = tuple(BAND_NAMES)
RGB_BANDS: tuple[int, int, int] = (4, 3, 2)  # true color = red/green/blue


def band_path(cfg: KSAAlignedConfig, i: int) -> Path:
    """Path to one snapped Landsat band (local analog of `cfg.dem_path`)."""
    return cfg.root / "02_snap_owned" / "multispectral" / "cubicspline" / f"snapped_landsat_b{i}.tif"


def pool2x2_nanmean(a: np.ndarray) -> np.ndarray:
    """NaN-aware 2x2 mean-pool of a 30 m `(2P, 2P)` array down to the 60 m `(P, P)`
    grid, so a band co-locates pixel-for-pixel with the HR magnetic raster. Blocks
    that are entirely NaN stay NaN."""
    h, w = a.shape
    blocks = a.reshape(h // 2, 2, w // 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN block -> NaN, expected
        return np.nanmean(blocks, axis=(1, 3))


def robust_clim(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    """Percentile color limits over finite values; falls back to (0, 1) if empty."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT_FOLDER / "data/processed/ksa_aligned/patch_indices_cellgrid8_fold3",
        help="Patch index dir (canonical fold-3 benchmark).",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-stats", type=int, default=200, help="Patches pooled for correlation/PCA stats.")
    p.add_argument("--n-panels", type=int, default=6, help="Patches rendered as visual panels.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT_FOLDER / "figures/multispectral")
    p.add_argument("--results-dir", type=Path, default=ROOT_FOLDER / "results/multispectral")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cfg = KSAAlignedConfig.from_yaml()
    ms_scale = cfg.dem_scale  # multispectral shares the 30 m DEM grid: 2x the 60 m HR grid

    idx = PatchIndex.load(args.index_dir / f"{args.split}.json")
    patch_px = idx.spec.patch_px  # HR pixels (60 m); source of truth for these windows
    rng = random.Random(args.seed)
    patches = list(idx.patches)
    rng.shuffle(patches)
    stat_patches = patches[: args.n_stats]
    panel_patches = patches[: args.n_panels]
    print(
        f"{args.split}: {len(idx.patches)} patches; using {len(stat_patches)} for stats, "
        f"{len(panel_patches)} for panels (patch_px={patch_px} @60m, ms_scale={ms_scale})"
    )

    hr_src = open_raster(cfg.hr_product_path(cfg.hr_products[0]))
    band_src = {i: open_raster(band_path(cfg, i)) for i in BANDS}

    # ---- accumulate common-valid pixels across the stat patches ----
    P = patch_px
    nan_count = {i: 0 for i in BANDS}
    n_total = 0
    xs: list[np.ndarray] = []  # each (7, k): bands on common-valid pixels
    hrs: list[np.ndarray] = []  # each (k,)
    for t in stat_patches:
        hr = hr_src.read_window(t.row_px, t.col_px, P, P)
        pooled = np.stack(
            [
                pool2x2_nanmean(
                    band_src[i].read_window(
                        t.row_px * ms_scale, t.col_px * ms_scale, P * ms_scale, P * ms_scale
                    )
                )
                for i in BANDS
            ]
        )  # (7, P, P) on the 60 m grid
        n_total += P * P
        for k, i in enumerate(BANDS):
            nan_count[i] += int(np.isnan(pooled[k]).sum())
        mask = np.isfinite(pooled).all(axis=0) & np.isfinite(hr)  # (P, P) common to all channels
        if mask.any():
            xs.append(pooled[:, mask])
            hrs.append(hr[mask])

    if not xs:
        raise RuntimeError("No common-valid pixels found across the sampled patches.")
    X = np.concatenate(xs, axis=1)  # (7, N)
    hr_vec = np.concatenate(hrs)  # (N,)
    n_valid = hr_vec.size
    print(f"Accumulated {n_valid:,} common-valid 60 m pixels.")

    # ---- per-band stats ----
    stats_csv = args.results_dir / "band_stats.csv"
    print("\nPer-band stats (over common-valid pixels; NaN%% over all sampled pixels):")
    print(f"{'band':>10} {'min':>9} {'max':>9} {'mean':>9} {'std':>9} {'nan%':>7}")
    with open(stats_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "name", "min", "max", "mean", "std", "nan_frac"])
        for k, i in enumerate(BANDS):
            row = X[k]
            nan_frac = nan_count[i] / max(n_total, 1)
            print(
                f"{f'b{i}({BAND_NAMES[i]})':>10} {row.min():9.4f} {row.max():9.4f} "
                f"{row.mean():9.4f} {row.std():9.4f} {100 * nan_frac:6.2f}%"
            )
            w.writerow([f"b{i}", BAND_NAMES[i], row.min(), row.max(), row.mean(), row.std(), nan_frac])
    print(f"-> {stats_csv}")

    # ---- correlations ----
    band_labels = [f"b{i}\n{BAND_NAMES[i]}" for i in BANDS]
    inter = np.corrcoef(X)  # (7, 7) inter-band redundancy
    band_hr = np.array([np.corrcoef(X[k], hr_vec)[0, 1] for k in range(len(BANDS))])  # (7,)

    print("\nBand <-> HR magnetic correlation (ranked by |r|):")
    order = np.argsort(-np.abs(band_hr))
    for k in order:
        i = BANDS[k]
        print(f"  b{i:<2} {BAND_NAMES[i]:<8} r = {band_hr[k]:+.3f}")

    # ---- PCA compression sniff-test (z-scored bands, unsupervised) ----
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    Z = (X - mu) / np.where(sd > 0, sd, 1.0)  # (7, N) standardized
    # SVD of the standardized data; right singular vectors are PC scores per pixel.
    U, S, _Vt = np.linalg.svd(Z @ Z.T)  # (7,7) covariance eigvecs in U, eigvals in S
    evr = S / S.sum()
    scores = U.T @ Z  # (7, N): PC scores per pixel (PC1 = scores[0])
    pc_hr = np.array([np.corrcoef(scores[k], hr_vec)[0, 1] for k in range(len(BANDS))])

    print("\nPCA over the 7 z-scored bands (compression sniff-test):")
    cum = 0.0
    for k in range(len(BANDS)):
        cum += evr[k]
        print(
            f"  PC{k + 1}: explained var {evr[k]:6.3f} (cum {cum:6.3f})   corr(PC{k + 1}, HR) = {pc_hr[k]:+.3f}"
        )
    print("  PC1 loadings:", "  ".join(f"b{BANDS[k]}={U[k, 0]:+.2f}" for k in range(len(BANDS))))

    # ---- write correlations.csv ----
    corr_csv = args.results_dir / "correlations.csv"
    with open(corr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# inter-band Pearson correlation"])
        w.writerow(["band"] + [f"b{i}" for i in BANDS])
        for k, i in enumerate(BANDS):
            w.writerow([f"b{i}"] + [f"{v:.4f}" for v in inter[k]])
        w.writerow([])
        w.writerow(["# band <-> HR magnetic correlation"])
        w.writerow(["band", "name", "corr_hr"])
        for k, i in enumerate(BANDS):
            w.writerow([f"b{i}", BAND_NAMES[i], f"{band_hr[k]:.4f}"])
        w.writerow([])
        w.writerow(["# PCA: explained variance ratio and PC<->HR correlation"])
        w.writerow(["component", "explained_var_ratio", "corr_hr"])
        for k in range(len(BANDS)):
            w.writerow([f"PC{k + 1}", f"{evr[k]:.4f}", f"{pc_hr[k]:.4f}"])
    print(f"-> {corr_csv}")

    # ---- figures: inter-band heatmap + band<->HR bars ----
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(inter, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(BANDS)), band_labels, fontsize=8)
    ax.set_yticks(range(len(BANDS)), band_labels, fontsize=8)
    for r in range(len(BANDS)):
        for c in range(len(BANDS)):
            ax.text(
                c,
                r,
                f"{inter[r, c]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(inter[r, c]) > 0.5 else "black",
            )
    ax.set_title("Inter-band Pearson correlation (redundancy)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    interband_png = args.out_dir / "interband_corr.png"
    fig.savefig(interband_png, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:red" if v >= 0 else "tab:blue" for v in band_hr]
    ax.bar([f"b{i}\n{BAND_NAMES[i]}" for i in BANDS], band_hr, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Pearson r vs HR magnetic")
    ax.set_title(f"Band correlation with HR magnetic ({cfg.hr_products[0]})")
    fig.tight_layout()
    bandhr_png = args.out_dir / "band_vs_hr.png"
    fig.savefig(bandhr_png, dpi=130)
    plt.close(fig)
    print(f"-> {interband_png}\n-> {bandhr_png}")

    # ---- visual panels: HR magnetic | b1..b7 | true-color RGB ----
    for n, t in enumerate(panel_patches):
        hr = hr_src.read_window(t.row_px, t.col_px, P, P)
        full = {  # native 30 m bands for the panels (no pooling) so RGB stays crisp
            i: band_src[i].read_window(t.row_px * ms_scale, t.col_px * ms_scale, P * ms_scale, P * ms_scale)
            for i in BANDS
        }
        fig, axes = plt.subplots(3, 3, figsize=(11, 11))
        axes = axes.ravel()
        lo, hi = robust_clim(hr)
        axes[0].imshow(hr, cmap="RdBu_r", vmin=lo, vmax=hi)
        axes[0].set_title(f"HR magnetic ({cfg.hr_products[0]})", fontsize=9)
        for k, i in enumerate(BANDS):
            lo, hi = robust_clim(full[i])
            axes[k + 1].imshow(full[i], cmap="viridis", vmin=lo, vmax=hi)
            axes[k + 1].set_title(f"b{i} {BAND_NAMES[i]}", fontsize=9)
        rgb = np.dstack([full[b] for b in RGB_BANDS])  # (2P, 2P, 3)
        lo, hi = robust_clim(rgb)
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
        rgb = np.nan_to_num(rgb)
        axes[8].imshow(rgb)
        axes[8].set_title("RGB (b4/b3/b2)", fontsize=9)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            f"{args.split} patch {n}  src={t.source_id}  row={t.row_px} col={t.col_px}  "
            f"valid_frac={t.valid_frac:.2f}",
            fontsize=10,
        )
        fig.tight_layout()
        panel_png = args.out_dir / f"panel_{n:02d}.png"
        fig.savefig(panel_png, dpi=110)
        plt.close(fig)
    print(f"-> {args.out_dir}/panel_*.png ({len(panel_patches)} panels)")

    clear_source_cache()


if __name__ == "__main__":
    main()

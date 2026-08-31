"""Remove the dominant brightness component (PC1) from the 7 Landsat bands and inspect
the RESIDUAL spectral structure (PC2, PC3, ...) over fold-3 test patches, next to the
HR magnetic field.

PC1 = ~96% of band variance = overall albedo / illumination — the common factor that
makes the 7 bands 0.92-0.99 collinear and that carried no SR signal. Removing it leaves
the inter-band mineralogical contrasts (iron in VNIR, clay/carbonate in SWIR), where any
real geology<->magnetics coupling would live.

Per test patch it renders: HR magnetic | PC1 (removed) | PC2 | PC3 | true-color RGB |
PC1-removed false-color (PC2,PC3,PC4); and it reports how PC1/PC2/PC3 correlate with the
magnetic field and its high-frequency detail, aggregated over the test patches.

The PCA basis is fit GLOBALLY on a sample of valid test-region band pixels (z-scored),
then applied to every patch, so the components are consistent across patches. Reuses
open_raster / PatchIndex / pool2x2_nanmean — no shared library code is touched.

Usage:
    uv run python experiments/multispectral/inspect_pc_residual.py
    uv run python experiments/multispectral/inspect_pc_residual.py --n-panels 10 --n-basis 300
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from experiments._archive.multispectral.inspect_multispectral import (
    BAND_NAMES,
    BANDS,
    band_path,
    pool2x2_nanmean,
    robust_clim,
)
from magsr import ROOT_FOLDER
from magsr.datasets import KSAAlignedConfig
from magsr.datasets.io import clear_source_cache, open_raster
from magsr.datasets.patching import PatchIndex

EPS = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT_FOLDER / "data/processed/ksa_aligned/patch_indices_cellgrid8_fold3",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--n-basis",
        type=int,
        default=250,
        help="Patches used to fit the global PCA basis + aggregate corr.",
    )
    p.add_argument("--n-panels", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT_FOLDER / "figures/multispectral/pc_residual")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    cfg = KSAAlignedConfig.from_yaml()
    ms_scale = cfg.dem_scale  # 30 m bands = 2x the 60 m HR grid
    hr_src = open_raster(cfg.hr_product_path(cfg.hr_products[0]))
    band_src = {i: open_raster(band_path(cfg, i)) for i in BANDS}

    idx = PatchIndex.load(args.index_dir / f"{args.split}.json")
    P = idx.spec.patch_px
    patches = list(idx.patches)
    rng.shuffle(patches)
    basis_patches = patches[: args.n_basis]
    panel_patches = patches[: args.n_panels]
    print(
        f"{args.split}: {len(idx.patches)} patches; basis {len(basis_patches)}, panels {len(panel_patches)} "
        f"(patch_px={P}@60m, ms_scale={ms_scale})"
    )

    def read_bands_60m(t):
        """7-band stack pooled to the 60 m grid, shape (P, P, 7)."""
        return np.stack(
            [
                pool2x2_nanmean(
                    band_src[i].read_window(
                        t.row_px * ms_scale, t.col_px * ms_scale, P * ms_scale, P * ms_scale
                    )
                )
                for i in BANDS
            ],
            axis=-1,
        )

    # ---- fit global PCA basis on 60 m band pixels over the basis patches ----
    cols = []
    for t in basis_patches:
        b = read_bands_60m(t).reshape(-1, 7)
        b = b[np.isfinite(b).all(1)]
        if b.size:
            cols.append(b)
    X = np.concatenate(cols)
    mu, sd = X.mean(0), X.std(0) + EPS
    Z = (X - mu) / sd
    _, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    evr = (S**2) / (S**2).sum()
    print(f"\nPCA basis on {X.shape[0]:,} test-region pixels:")
    for k in range(4):
        print(
            f"  PC{k+1}: EVR {evr[k]:6.3f}  loadings "
            + " ".join(f"b{BANDS[i]}={Vt[k,i]:+.2f}" for i in range(7))
        )

    def project(stack):  # (.., 7) -> PCs (.., 7)
        return ((stack - mu) / sd) @ Vt.T

    # ---- aggregate correlations: PC1/PC2/PC3 vs magnetic field & detail ----
    pcs_acc = {1: [], 2: [], 3: []}
    hr_acc, det_acc = [], []
    for t in basis_patches:
        hr = hr_src.read_window(t.row_px, t.col_px, P, P)
        bands = read_bands_60m(t)
        m = np.isfinite(hr) & np.isfinite(bands).all(-1)
        if m.sum() < 50:
            continue
        pc = project(bands)
        det = hr - ndimage.gaussian_filter(
            np.nan_to_num(hr, nan=float(np.nanmean(hr))), 2.0
        )  # high-pass detail
        for k in (1, 2, 3):
            pcs_acc[k].append(pc[..., k][m])
        hr_acc.append(hr[m])
        det_acc.append(det[m])
    HR = np.concatenate(hr_acc)
    DET = np.concatenate(det_acc)
    print(f"\nAggregate over {len(hr_acc)} test patches ({HR.size:,} px) — Pearson r:")
    print(f"{'comp':>6} {'vs HR field':>12} {'vs HR detail':>13}")
    for k in (1, 2, 3):
        v = np.concatenate(pcs_acc[k])
        print(f"{'PC'+str(k):>6} {np.corrcoef(v, HR)[0,1]:>12.4f} {np.corrcoef(v, DET)[0,1]:>13.4f}")

    # ---- panels ----
    for n, t in enumerate(panel_patches):
        hr = hr_src.read_window(t.row_px, t.col_px, P, P)
        full = {
            i: band_src[i].read_window(t.row_px * ms_scale, t.col_px * ms_scale, P * ms_scale, P * ms_scale)
            for i in BANDS
        }
        stack = np.stack([full[i] for i in BANDS], axis=-1)  # (2P,2P,7)
        pc = project(stack)  # (2P,2P,7)

        fig, axes = plt.subplots(2, 3, figsize=(13, 9))
        ax = axes.ravel()
        lo, hi = robust_clim(hr)
        ax[0].imshow(hr, cmap="RdBu_r", vmin=lo, vmax=hi)
        ax[0].set_title(f"HR magnetic ({cfg.hr_products[0]})", fontsize=9)
        for j, (k, lbl) in enumerate([(0, "PC1 (removed — brightness)"), (1, "PC2"), (2, "PC3")], start=1):
            lo, hi = robust_clim(pc[..., k])
            ax[j].imshow(pc[..., k], cmap="cividis", vmin=lo, vmax=hi)
            ax[j].set_title(lbl, fontsize=9)
        rgb = np.dstack([full[4], full[3], full[2]])
        lo, hi = robust_clim(rgb)
        ax[4].imshow(np.nan_to_num(np.clip((rgb - lo) / (hi - lo + EPS), 0, 1)))
        ax[4].set_title("true-color RGB (b4/b3/b2)", fontsize=9)
        res = np.dstack([pc[..., 1], pc[..., 2], pc[..., 3]])  # PC1-removed residual structure
        lo, hi = robust_clim(res)
        ax[5].imshow(np.nan_to_num(np.clip((res - lo) / (hi - lo + EPS), 0, 1)))
        ax[5].set_title("PC1-removed false-color (PC2,PC3,PC4)", fontsize=9)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle(
            f"{args.split} patch {n}  src={t.source_id}  row={t.row_px} col={t.col_px}", fontsize=10
        )
        fig.tight_layout()
        fig.savefig(args.out_dir / f"pc_residual_{n:02d}.png", dpi=110)
        plt.close(fig)
    print(f"\n-> {args.out_dir}/pc_residual_*.png ({len(panel_patches)} panels)")
    clear_source_cache()


if __name__ == "__main__":
    main()

"""Reconstruct each KSA-aligned block's held-out test region by patchwise SR + blending.

Writes one HR GeoTIFF per block, a region-level metrics CSV (RDN++ vs Bicubic),
and a 2×2 magnetic-only panel PNG (LR / Bicubic / HR-Target / RDN++) per block.

Usage:
    uv run python experiments/ksa_aligned_reconstruct.py \
        --checkpoint checkpoints/<run>/<run>_best.pt \
        --stride-lr-px 22 --out-dir figures/recon_ksa

    uv run python experiments/ksa_aligned_reconstruct.py \
        --checkpoint checkpoints/rdnpp_x3_16mp_last.pt \
        --blocks 1 2 3 --stride-lr-px 22 --out-dir figures/recon_ksa
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.windows import from_bounds

from magsr.cli import add_channel_args
from magsr.datasets import KSAAlignedConfig
from magsr.datasets.ksa_shield_aligned import KSAShieldAlignedDataset
from magsr.metrics import per_patch_msssim, per_patch_rmse, per_patch_ssim
from magsr.models import BicubicModel, load_checkpoint
from magsr.reconstruct import (
    Normalizer,
    PatchPlan,
    build_aux_lr_channels,
    load_rect_json,
    plan_lr_patches,
    reconstruct_region_multichannel,
    write_reconstruction_geotiff,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--blocks", type=int, nargs="+", default=(1, 2, 3))
    p.add_argument("--lr-product", default="RTP")
    p.add_argument("--hr-truth-product", default="AMF_RTP")
    p.add_argument("--stride-lr-px", type=int, default=None)
    p.add_argument("--blend-kind", default="auto", choices=("auto", "linear", "cosine", "gaussian", "ones"))
    p.add_argument("--batch-size", type=int, default=8)
    add_channel_args(p)
    p.add_argument("--out-dir", type=Path, default=Path("figures/recon_ksa"))
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="CSV path. Default: <out-dir>/metrics.csv. Rows are appended (header on first write).",
    )
    return p.parse_args()


def read_lr_array(lr_path: Path, plan: PatchPlan) -> np.ndarray:
    """Full LR window read as float32 nT, NoData -> NaN."""
    with rasterio.open(lr_path) as src:
        arr = src.read(1, window=plan.lr_window).astype(np.float32, copy=False)
        nd = src.nodata
    if nd is not None and not np.isnan(nd):
        arr = np.where(arr == nd, np.float32(np.nan), arr)
    return arr


def read_hr_truth_aligned(
    truth_path: Path,
    plan: PatchPlan,
    *,
    clip_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """HR truth on the (plan.hr_size) grid; NaN where source missing or NoData.

    Pass `clip_range=(vmin, vmax)` to clip finite pixels into the same `[vmin, vmax]` window
    the predictions are clamped to — keeps the comparison in one agreed range.
    """
    bbox = plan.world_bbox
    hr_h, hr_w = plan.hr_size
    with rasterio.open(truth_path) as src:
        win = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(
            1,
            window=win,
            out_shape=(hr_h, hr_w),
            boundless=True,
            fill_value=np.nan,
        ).astype(np.float32, copy=False)
        nd = src.nodata
    if nd is not None and not np.isnan(nd):
        arr = np.where(arr == nd, np.float32(np.nan), arr)
    arr[np.abs(arr) > 1e5] = np.nan
    if clip_range is not None:
        vmin, vmax = clip_range
        finite = np.isfinite(arr)
        arr[finite] = np.clip(arr[finite], vmin, vmax)
    return arr


def bicubic_upsample(lr_arr: np.ndarray, normalizer: Normalizer, scale: int) -> np.ndarray:
    """Single-shot bicubic LR -> HR in nT, mirroring SR's normalize / clamp / denormalize path."""
    fill = 0.5 * (normalizer.vmin + normalizer.vmax)
    lr_filled = np.nan_to_num(lr_arr, nan=fill)
    lr_norm = normalizer.normalize(lr_filled)
    lr_t = torch.from_numpy(lr_norm)[None, None].float()
    sr_norm = BicubicModel(upscale_factor=scale)(lr_t).clamp(0.0, 1.0).squeeze().numpy()
    return normalizer.denormalize(sr_norm)


def region_metrics(sr: np.ndarray, hr: np.ndarray, *, data_range: float) -> dict[str, float]:
    """RMSE / SSIM / MS-SSIM on the full stitched region. Single scalar each (no std).

    Mask on `sr_finite & hr_finite` jointly so the NaN border that `reconstruct_region` writes
    where blend weight sums to zero (corners / edges) doesn't poison the metrics.
    """
    sr_t = torch.from_numpy(sr).float()[None, None]
    hr_t = torch.from_numpy(hr).float()[None, None]
    mask = (sr_t.isfinite() & hr_t.isfinite()).float()
    sr_m = sr_t.nan_to_num(0.0) * mask
    hr_m = hr_t.nan_to_num(0.0) * mask

    rmse = per_patch_rmse(sr_m, hr_m, mask).item()
    ssim = per_patch_ssim(sr_m, hr_m, data_range=data_range).item()
    msssim = per_patch_msssim(sr_m, hr_m, patch_size=min(sr.shape), data_range=data_range).item()

    return {"RMSE": rmse, "SSIM": ssim, "MS-SSIM": msssim}


def append_metrics_csv(
    csv_path: Path,
    *,
    checkpoint: str,
    block: str,
    method: str,
    metrics: dict[str, float],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["checkpoint", "block", "method", "RMSE", "SSIM", "MS-SSIM"])
        w.writerow(
            [
                checkpoint,
                block,
                method,
                f"{metrics['RMSE']:.6f}",
                f"{metrics['SSIM']:.6f}",
                f"{metrics['MS-SSIM']:.6f}",
            ]
        )


def plot_block_panel(
    *,
    label: str,
    plan: PatchPlan,
    lr_arr: np.ndarray,
    sr_arr: np.ndarray,
    bicubic_arr: np.ndarray,
    truth_arr: np.ndarray,
    m_sr: dict[str, float],
    m_bic: dict[str, float],
    save_path: Path,
    suptitle: str | None = None,
) -> plt.Figure:
    """2×2 magnetic-data panel with normal km axis labels (local coords, SW origin).

    Layout:
        row 0: LR (Input)   |  Bicubic + metrics
        row 1: HR (Target)  |  RDN++   + metrics
    """
    finite_hr = truth_arr[np.isfinite(truth_arr)]
    if finite_hr.size:
        clim = {"vmin": float(finite_hr.min()), "vmax": float(finite_hr.max())}
    else:
        clim = {"vmin": -1.0, "vmax": 1.0}

    ext = plan.world_extent  # [left, right, bottom, top] in m
    w_km = (ext[1] - ext[0]) / 1000.0
    h_km = (ext[3] - ext[2]) / 1000.0
    extent_km = [0.0, w_km, 0.0, h_km]

    fig, axes = plt.subplots(2, 2, figsize=(11, 11), sharex=True, sharey=True)

    def _title(name: str, m: dict[str, float]) -> str:
        return f"{name}\nRMSE {m['RMSE']:.2f}   SSIM {m['SSIM']:.3f}   MS-SSIM {m['MS-SSIM']:.3f}"

    img_kw = dict(extent=extent_km, origin="upper", cmap="gray", interpolation="nearest", **clim)
    im_data = axes[0, 0].imshow(lr_arr, **img_kw)
    axes[0, 0].set_title("LR (Input)", fontweight="bold")
    axes[0, 1].imshow(bicubic_arr, **img_kw)
    axes[0, 1].set_title(_title("Bicubic", m_bic), fontweight="bold")
    axes[1, 0].imshow(truth_arr, **img_kw)
    axes[1, 0].set_title("HR (Target)", fontweight="bold")
    axes[1, 1].imshow(sr_arr, **img_kw)
    axes[1, 1].set_title(_title("RDN++", m_sr), fontweight="bold")

    for ax in axes.flatten():
        ax.set_aspect("equal")
    for ax in axes[1, :]:
        ax.set_xlabel("X (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Y (km)")

    fig.subplots_adjust(right=0.88, wspace=0.04, hspace=0.10)
    cbar = fig.colorbar(im_data, cax=fig.add_axes([0.90, 0.15, 0.02, 0.7]))
    cbar.set_label("Magnetic Field (nT)", fontweight="bold")

    fig.suptitle(suptitle or label, fontweight="bold", fontsize=16)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    multichannel = bool(args.use_dem or args.lr_aux)
    cfg = KSAAlignedConfig.from_yaml(
        load_dem=bool(args.use_dem),
        dem_mode=args.dem_mode,
        lr_aux_products=tuple(args.lr_aux),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_lr = cfg.patch_px // cfg.lr_scale
    stride_lr = args.stride_lr_px if args.stride_lr_px is not None else patch_lr
    lr_path = cfg.lr_product_path(args.lr_product)
    truth_path = cfg.hr_product_path(args.hr_truth_product) if args.hr_truth_product else None

    aux_dataset = None
    lr_aux_paths: dict[str, Path] | None = None
    if multichannel:
        aux_dataset = KSAShieldAlignedDataset.for_channels(cfg, product=args.lr_product)
        lr_aux_paths = {p: cfg.lr_product_path(p) for p in cfg.lr_aux_products}

    normalizer = Normalizer.from_json(cfg.normalization_path)
    data_range = normalizer.data_range

    csv_path = args.metrics_csv if args.metrics_csv is not None else args.out_dir / "metrics_test_recon.csv"
    ckpt_stem = args.checkpoint.stem

    model = load_checkpoint(args.checkpoint, device=device)

    for block in args.blocks:
        rect = load_rect_json(cfg.patch_index_dir / f"test_rect_B{block}.json")
        plan = plan_lr_patches(
            polygon_world=rect,
            lr_path=lr_path,
            patch_lr_px=patch_lr,
            stride_lr_px=stride_lr,
            scale=cfg.lr_scale,
        )

        lr_arr = read_lr_array(lr_path, plan)
        aux_channels = (
            build_aux_lr_channels(
                plan,
                dataset=aux_dataset,
                lr_aux_paths=lr_aux_paths,
                dem_path=cfg.dem_path,
            )
            if multichannel
            else None
        )
        sr_arr = reconstruct_region_multichannel(
            model,
            lr_path,
            plan,
            normalizer=normalizer,
            aux_channels=aux_channels,
            blend_kind=args.blend_kind,
            batch_size=args.batch_size,
            device=device,
            block_id=block,
        )
        bicubic_arr = bicubic_upsample(lr_arr, normalizer, cfg.lr_scale)

        out_tif = args.out_dir / f"recon_B{block}.tif"
        write_reconstruction_geotiff(sr_arr, plan, out_tif)
        print(f"B{block}: {len(plan.positions)} patches → {out_tif}")

        if truth_path is None:
            continue

        truth_arr = read_hr_truth_aligned(truth_path, plan, clip_range=(normalizer.vmin, normalizer.vmax))
        m_sr = region_metrics(sr_arr, truth_arr, data_range=data_range)
        m_bic = region_metrics(bicubic_arr, truth_arr, data_range=data_range)
        print(
            f"  RDN++   RMSE={m_sr['RMSE']:.4f} SSIM={m_sr['SSIM']:.4f} MS-SSIM={m_sr['MS-SSIM']:.4f}\n"
            f"  Bicubic RMSE={m_bic['RMSE']:.4f} SSIM={m_bic['SSIM']:.4f} MS-SSIM={m_bic['MS-SSIM']:.4f}"
        )
        append_metrics_csv(csv_path, checkpoint=ckpt_stem, block=f"B{block}", method="RDN++", metrics=m_sr)
        append_metrics_csv(
            csv_path, checkpoint=ckpt_stem, block=f"B{block}", method="Bicubic", metrics=m_bic
        )

        panel_path = args.out_dir / f"recon_B{block}_panel.png"
        plot_block_panel(
            label=f"B{block}",
            plan=plan,
            lr_arr=lr_arr,
            sr_arr=sr_arr,
            bicubic_arr=bicubic_arr,
            truth_arr=truth_arr,
            m_sr=m_sr,
            m_bic=m_bic,
            save_path=panel_path,
            suptitle=f"Block B{block} — blend={args.blend_kind}, stride_lr={stride_lr}",
        )
        plt.close("all")
        print(f"  panel → {panel_path}")

    if truth_path is not None:
        print(f"metrics → {csv_path}")


if __name__ == "__main__":
    main()

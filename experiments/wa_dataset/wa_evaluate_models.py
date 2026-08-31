"""Evaluation over test set of the WA dataset.

Reproduces the Smith et al. (2022) accuracy table: per-patch RMSE / SSIM / MS-SSIM
reported as mean ± std across all test patches, for RDN++ and a bicubic baseline.
"""

from __future__ import annotations

import argparse
from os import name
from pathlib import Path
from unicodedata import name

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from magsr import ROOT_FOLDER
from magsr.datasets import build_wa_datasets, pool_collate, worker_init_fn
from magsr.metrics import (
    apply_hr_nan_mask,
    make_rmse,
    per_patch_msssim,
    per_patch_rmse,
    per_patch_ssim,
)
from magsr.models import BicubicModel, rdnpp_default_x4

DEFAULT_CKPT = Path("checkpoints/rdnpp_x4_wa_baseline_best.pt")  # sweep_configs/wa.yaml baseline


def build_wa_loaders(batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders sharing one `build_wa_datasets()` call."""
    return {
        split: DataLoader(
            ds,
            collate_fn=pool_collate,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            persistent_workers=num_workers > 0,
        )
        for split, ds in build_wa_datasets().items()
    }


def load_rdn_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = rdnpp_default_x4().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | None = None,
    desc: str = "WA",
) -> dict[str, tuple[float, float]]:
    """Run `model` over `loader`; return per-metric (mean, std) across patches."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = int(next(iter(loader))["hr"].shape[-1])
    vmin, vmax = loader.dataset.vmin, loader.dataset.vmax
    data_range = vmax - vmin

    batch_rmse = make_rmse().to(device)

    store: dict[str, list[torch.Tensor]] = {"rmse": [], "rmse_batch": [], "ssim": [], "msssim": []}

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, ncols=100):
            lr_t = batch["lr"].to(device, non_blocking=True)
            hr_t = batch["hr"].to(device, non_blocking=True)

            sr = model(lr_t).clamp(0, 1)
            mask = hr_t.isfinite().to(sr.dtype)
            sr_m, hr_m = apply_hr_nan_mask(sr, hr_t)

            # Denormalize to nanoTesla (nT) units
            sr_nt = (sr_m * data_range) + (mask * vmin)
            hr_nt = (hr_m * data_range) + (mask * vmin)

            store["rmse"].append(per_patch_rmse(sr_nt, hr_nt, mask).cpu())
            store["rmse_batch"].append(batch_rmse(sr_nt, hr_nt).reshape(1).cpu())

            fully_valid = mask.flatten(1).all(dim=1)
            if fully_valid.any():
                sr_v = sr_nt[fully_valid]
                hr_v = hr_nt[fully_valid]
                store["ssim"].append(per_patch_ssim(sr_v, hr_v, data_range=data_range).cpu())
                store["msssim"].append(
                    per_patch_msssim(sr_v, hr_v, patch_size=patch_size, data_range=data_range).cpu()
                )

    return {
        k: (torch.cat(v).mean().item(), torch.cat(v).std(unbiased=True).item()) for k, v in store.items()
    }


def format_table(results: dict[str, dict[str, tuple[float, float]]]) -> str:
    methods = list(results.keys())
    metrics = ["rmse", "rmse_batch", "ssim", "msssim"]
    col_w = 24
    lines = [
        f"{'':<10}" + "".join(f"{m:<{col_w}}" for m in methods),
        f"{'':<10}" + "".join(f"{'mean':<12}{'std':<12}" for _ in methods),
    ]
    lines.append("-" * len(lines[0]))
    for metric in metrics:
        row = f"{metric:<10}"
        for method in methods:
            mean, std = results[method][metric]
            row += f"{mean:<12.4f}{std:<12.4f}"
        lines.append(row)
    return "\n".join(lines)


def print_table(results: dict[str, dict[str, tuple[float, float]]]) -> None:
    print()
    print(format_table(results))


def _patch_metrics(
    sr: torch.Tensor,
    hr: torch.Tensor,
    data_range: float,
    vmin: float,
    patch_size: int,
) -> dict[str, float | None]:
    """Scalar RMSE (always) and SSIM / MS-SSIM (only if patch is fully valid)."""
    mask = hr.isfinite().to(sr.dtype)
    sr_m, hr_m = apply_hr_nan_mask(sr, hr)
    sr_nt = sr_m * data_range + mask * vmin
    hr_nt = hr_m * data_range + mask * vmin

    out: dict[str, float | None] = {
        "RMSE": per_patch_rmse(sr_nt, hr_nt, mask).item(),
        "SSIM": None,
        "MS-SSIM": None,
    }
    if bool(mask.all()):
        out["SSIM"] = per_patch_ssim(sr_nt, hr_nt, data_range=data_range).item()
        out["MS-SSIM"] = per_patch_msssim(sr_nt, hr_nt, patch_size=patch_size, data_range=data_range).item()
    return out


def peek_reconstructions(ckpt_path: Path = DEFAULT_CKPT):
    """Helper to peek at reconstructions for a few patches."""
    from matplotlib.colors import Normalize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rdn = load_rdn_model(ckpt_path, device)
    bicubic = BicubicModel(upscale_factor=4).to(device)

    save_dir = ROOT_FOLDER / "figures" / "wa_reconstructions" / ckpt_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    indices = [7, 8, 78, 780, 1340, 300]
    ds = build_wa_datasets()["test"]

    for idx in indices:
        sample = ds[idx]
        hr = sample["hr"]["MAG"].to(device)
        lr = sample["lr"]["MAG"].to(device)

        # Add Batch and Channel dims: [H, W] -> [1, 1, H, W]
        hr = hr[None, None]
        lr = lr[None, None]

        name = sample["meta"]["name"]
        patch_size = int(hr.shape[-1])

        vmin, vmax = ds.vmin, ds.vmax
        data_range = vmax - vmin

        with torch.no_grad():
            sr_rdn = rdn(lr).clamp(0, 1)
            sr_bic = bicubic(lr).clamp(0, 1)

        # Metrics
        m_bic = _patch_metrics(sr_bic, hr, data_range, vmin, patch_size)
        m_rdn = _patch_metrics(sr_rdn, hr, data_range, vmin, patch_size)

        # Convert everything to nT units for visualization and error
        def to_nt_raw(t):
            return t.squeeze().detach().cpu().numpy() * data_range + vmin

        img_lr = to_nt_raw(lr)
        img_hr = to_nt_raw(hr)
        img_rdn = to_nt_raw(sr_rdn)
        img_bic = to_nt_raw(sr_bic)

        err_bic = img_hr - img_bic
        err_rdn = img_hr - img_rdn

        # Define Layout: 2 rows, 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(14, 10))

        # Per-patch grayscale range; symmetric linear error range shared by both methods.
        # Keyed to the p99 of |error| (robust to a single hot pixel) so the linear scale
        # isn't washed out by outliers.
        clim = dict(vmin=float(np.nanmin(img_hr)), vmax=float(np.nanmax(img_hr)))
        abs_err = np.abs(np.concatenate([err_bic.ravel(), err_rdn.ravel()]))
        abs_err = abs_err[np.isfinite(abs_err)]
        max_err = float(np.percentile(abs_err, 99)) if abs_err.size else 1e-3
        max_err = max(max_err, 1e-3)
        err_norm = Normalize(vmin=-max_err, vmax=max_err)

        def _title(name: str, m: dict[str, float | None]) -> str:
            line = f"RMSE {m['RMSE']:.2f}"
            if m["SSIM"] is not None:
                line += f"   SSIM {m['SSIM']:.3f}   MS-SSIM {m['MS-SSIM']:.3f}"
            return f"{name}\n{line}"

        # Column 1: Input / Target
        im_data = axes[0, 0].imshow(img_lr, cmap="gray", **clim)
        axes[0, 0].set_title("LR (Input)", fontweight="bold")
        axes[1, 0].imshow(img_hr, cmap="gray", **clim)
        axes[1, 0].set_title("HR (Target)", fontweight="bold")

        # Column 2: Reconstructions with metrics
        axes[0, 1].imshow(img_bic, cmap="gray", **clim)
        axes[0, 1].set_title(_title("Bicubic", m_bic), fontweight="bold")
        axes[1, 1].imshow(img_rdn, cmap="gray", **clim)
        axes[1, 1].set_title(_title("RDN++", m_rdn), fontweight="bold")

        # Column 3: Signed error, standalone. Shared symlog-normalized scale.
        im_err = axes[0, 2].imshow(err_bic, cmap="RdBu_r", norm=err_norm)
        axes[0, 2].set_title("Bicubic Error", fontweight="bold")
        axes[1, 2].imshow(err_rdn, cmap="RdBu_r", norm=err_norm)
        axes[1, 2].set_title("RDN++ Error", fontweight="bold")

        # Hide ticks everywhere, but keep a thin black frame on the error panels
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in (axes[0, 2], axes[1, 2]):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("black")
                spine.set_linewidth(1.0)
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Tighten spacing between subplots; leave room on the right for colorbars
        fig.subplots_adjust(right=0.85, wspace=0.02, hspace=0.08)

        # 1. Grayscale colorbar (Full vertical)
        cbar_ax_data = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar_data = fig.colorbar(im_data, cax=cbar_ax_data)
        cbar_data.set_label("Magnetic Field (nT)", fontweight="bold")

        # 2. Error colorbar (Full vertical, further right)
        cbar_ax_err = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar_err = fig.colorbar(im_err, cax=cbar_ax_err)
        cbar_err.set_label("Error (nT)", fontweight="bold")

        fig.suptitle(f"Patch #{idx}: {name}", fontweight="bold", fontsize=16)

        save_path = save_dir / f"patch_{idx}_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved detailed comparison: {save_path}")
        plt.close(fig)


def evaluate_models() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["train", "val", "test"],
        help="Which WA splits to evaluate.",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Instead of the metrics table, render the 6-panel per-patch comparison figures.",
    )
    args = parser.parse_args()

    if args.peek:
        peek_reconstructions(args.ckpt)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rdn = load_rdn_model(args.ckpt, device)
    bicubic = BicubicModel(upscale_factor=4).to(device)

    loaders = build_wa_loaders(args.batch_size, args.num_workers)
    save_dir = ROOT_FOLDER / "results" / "wa_evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{args.ckpt.stem}_results.csv"
    csv_new = not csv_path.exists()

    print(f"\nCheckpoint: {args.ckpt}")
    with open(csv_path, "a") as csv_f:
        if csv_new:
            csv_f.write("checkpoint,split,method,metric,mean,std\n")
        for split in args.splits:
            results = {
                "RDN++": evaluate_loader(rdn, loaders[split], device=device, desc=f"RDN++ {split}"),
                "Bicubic": evaluate_loader(bicubic, loaders[split], device=device, desc=f"Bicubic {split}"),
            }
            table = format_table(results)
            print(f"\n=== WA {split} ===")
            print(table)

            txt_path = save_dir / f"{args.ckpt.stem}_{split}_results.txt"
            txt_path.write_text(f"Checkpoint: {args.ckpt}\nSplit: {split}\n{table}\n")
            for method, metrics in results.items():
                for metric, (mean, std) in metrics.items():
                    csv_f.write(f"{args.ckpt.stem},{split},{method},{metric},{mean:.6f},{std:.6f}\n")
            print(f"Saved: {txt_path}")
    print(f"Appended CSV rows: {csv_path}")


if __name__ == "__main__":
    evaluate_models()

"""Peek at random KSA-aligned test patches: HR/LR + Bicubic/RDN++ + linear-scale errors.

Mirrors `wa_evaluate_models.peek_reconstructions` for the KSA-aligned test split.
For each sampled patch, writes one 2×3 PNG:

    row 0: LR (Input)   |  Bicubic + metrics  |  Bicubic Error  (linear scale)
    row 1: HR (Target)  |  RDN++   + metrics  |  RDN++ Error    (linear scale)

Usage:
    uv run python experiments/ksa_aligned_peek.py \
        --checkpoint checkpoints/ksa_2gpu_b32/ksa_2gpu_b32_last.pt

    uv run python experiments/ksa_aligned_peek.py \
        --checkpoint checkpoints/ksa_2gpu_b32/ksa_2gpu_b32_last.pt \
        --n-samples 6 --seed 7 --out-dir figures/ksa_peek
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from magsr.datasets import build_ksa_aligned_datasets
from magsr.metrics import apply_hr_nan_mask, per_patch_msssim, per_patch_rmse, per_patch_ssim
from magsr.models import BicubicModel, load_checkpoint
from magsr.normalize import Normalizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hr-product", default="AMF_RTP")
    p.add_argument("--lr-product", default="RTP")
    p.add_argument("--out-dir", type=Path, default=Path("figures/ksa_peek"))
    return p.parse_args()


def patch_metrics(
    sr_nt: torch.Tensor,
    hr_nt: torch.Tensor,
    *,
    data_range: float,
    patch_size: int,
) -> dict[str, float | None]:
    """Scalar RMSE (always) and SSIM / MS-SSIM (only if patch is fully valid). Inputs in nT."""
    mask = hr_nt.isfinite().to(sr_nt.dtype)
    sr_m, hr_m = apply_hr_nan_mask(sr_nt, hr_nt)
    out: dict[str, float | None] = {
        "RMSE": per_patch_rmse(sr_m, hr_m, mask).item(),
        "SSIM": None,
        "MS-SSIM": None,
    }
    if bool(mask.all()):
        out["SSIM"] = per_patch_ssim(sr_m, hr_m, data_range=data_range).item()
        out["MS-SSIM"] = per_patch_msssim(sr_m, hr_m, patch_size=patch_size, data_range=data_range).item()
    return out


def plot_patch(
    *,
    idx: int,
    name: str,
    block: int,
    extent_km: list[float],
    img_lr: np.ndarray,
    img_hr: np.ndarray,
    img_bic: np.ndarray,
    img_rdn: np.ndarray,
    m_bic: dict[str, float | None],
    m_rdn: dict[str, float | None],
    save_path: Path,
) -> None:
    err_bic = img_hr - img_bic
    err_rdn = img_hr - img_rdn

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharex=True, sharey=True)

    clim = dict(vmin=float(np.nanmin(img_hr)), vmax=float(np.nanmax(img_hr)))
    max_err = float(
        max(
            np.nanmax(np.abs(err_bic)) if np.isfinite(err_bic).any() else 0.0,
            np.nanmax(np.abs(err_rdn)) if np.isfinite(err_rdn).any() else 0.0,
        )
    )
    img_kw = dict(extent=extent_km, origin="upper", interpolation="nearest")
    data_kw = dict(cmap="gray", **clim, **img_kw)
    err_kw = dict(cmap="RdBu_r", vmin=-max_err, vmax=max_err, **img_kw)  # linear, symmetric

    def _title(label: str, m: dict[str, float | None]) -> str:
        line = f"RMSE {m['RMSE']:.2f}"
        if m["SSIM"] is not None:
            line += f"   SSIM {m['SSIM']:.3f}   MS-SSIM {m['MS-SSIM']:.3f}"
        return f"{label}\n{line}"

    im_data = axes[0, 0].imshow(img_lr, **data_kw)
    axes[0, 0].set_title("LR (Input)", fontweight="bold")
    axes[1, 0].imshow(img_hr, **data_kw)
    axes[1, 0].set_title("HR (Target)", fontweight="bold")

    axes[0, 1].imshow(img_bic, **data_kw)
    axes[0, 1].set_title(_title("Bicubic", m_bic), fontweight="bold")
    axes[1, 1].imshow(img_rdn, **data_kw)
    axes[1, 1].set_title(_title("RDN++", m_rdn), fontweight="bold")

    im_err = axes[0, 2].imshow(err_bic, **err_kw)
    axes[0, 2].set_title("Bicubic Error", fontweight="bold")
    axes[1, 2].imshow(err_rdn, **err_kw)
    axes[1, 2].set_title("RDN++ Error", fontweight="bold")

    for ax in axes.flatten():
        ax.set_aspect("equal")
    for ax in axes[1, :]:
        ax.set_xlabel("X (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Y (km)")

    fig.subplots_adjust(right=0.85, wspace=0.05, hspace=0.10)
    fig.colorbar(im_data, cax=fig.add_axes([0.88, 0.15, 0.02, 0.7])).set_label(
        "Magnetic Field (nT)", fontweight="bold"
    )
    fig.colorbar(im_err, cax=fig.add_axes([0.94, 0.15, 0.02, 0.7])).set_label(
        "Error (nT)", fontweight="bold"
    )

    fig.suptitle(f"Patch #{idx} — B{block}: {name}", fontweight="bold", fontsize=16)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = build_ksa_aligned_datasets()
    test_ds = splits["test"]
    cfg = test_ds.config
    if test_ds.norm_stats is None:
        raise RuntimeError(
            f"{cfg.normalization_path} not found. Run "
            "scripts/build_ksa_dataset/04_compute_ksa_aligned_normalization.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rdn = load_checkpoint(args.checkpoint, device=device).eval()

    bicubic = BicubicModel(upscale_factor=cfg.lr_scale).to(device).eval()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(test_ds), size=args.n_samples, replace=False).tolist()
    print(f"Sampled patch indices: {indices}")

    data_range = Normalizer.from_stats(test_ds.norm_stats["global"]).data_range

    for idx in indices:
        sample = test_ds[int(idx)]
        meta = sample["meta"]
        block = int(meta["block"])
        name = meta["name"]
        w_km = (float(meta["right"]) - float(meta["left"])) / 1000.0
        h_km = (float(meta["top"]) - float(meta["bottom"])) / 1000.0
        extent_km = [0.0, w_km, 0.0, h_km]

        hr_t = sample["hr"][args.hr_product][None, None].to(device).float()
        lr_t = sample["lr"][args.lr_product][None, None].to(device).float()
        patch_size = int(hr_t.shape[-1])

        hr_norm = test_ds.normalize(hr_t, blocks=[block])
        lr_norm = test_ds.normalize(lr_t, blocks=[block])

        with torch.no_grad():
            sr_rdn_norm = rdn(lr_norm).clamp(0, 1)
            sr_bic_norm = bicubic(lr_norm).clamp(0, 1)

        sr_rdn_nt = test_ds.denormalize(sr_rdn_norm, blocks=[block])
        sr_bic_nt = test_ds.denormalize(sr_bic_norm, blocks=[block])
        hr_nt = test_ds.denormalize(hr_norm, blocks=[block])
        lr_nt = test_ds.denormalize(lr_norm, blocks=[block])

        m_bic = patch_metrics(sr_bic_nt, hr_nt, data_range=data_range, patch_size=patch_size)
        m_rdn = patch_metrics(sr_rdn_nt, hr_nt, data_range=data_range, patch_size=patch_size)

        save_path = args.out_dir / f"patch_{int(idx):05d}.png"
        plot_patch(
            idx=int(idx),
            name=name,
            block=block,
            extent_km=extent_km,
            img_lr=lr_nt.squeeze().cpu().numpy(),
            img_hr=hr_nt.squeeze().cpu().numpy(),
            img_bic=sr_bic_nt.squeeze().cpu().numpy(),
            img_rdn=sr_rdn_nt.squeeze().cpu().numpy(),
            m_bic=m_bic,
            m_rdn=m_rdn,
            save_path=save_path,
        )
        ssim_str = f"{m_rdn['SSIM']:.3f}" if m_rdn["SSIM"] is not None else "—"
        print(
            f"  patch #{idx} (B{block}): RDN++ RMSE={m_rdn['RMSE']:.2f} SSIM={ssim_str}  "
            f"Bicubic RMSE={m_bic['RMSE']:.2f}  → {save_path}"
        )


if __name__ == "__main__":
    main()

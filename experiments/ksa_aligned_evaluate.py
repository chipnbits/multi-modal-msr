"""Evaluation over the KSA-aligned dataset (train / val / test splits).

Per-patch RMSE (nT) / SSIM / MS-SSIM reported as mean ± std across all patches
in each split, for RDN++ x3 and a bicubic baseline. Also breaks the table
down per SGS survey block (B1 / B2 / B3).

Notes:
    - HR is multi-product on KSA (default `AMF_RTP`); we slice the first
      channel and treat that as the SR target.
    - Normalization is delegated to `KSAShieldAlignedDataset.normalize/denormalize`
      using the per-batch `blocks` list pulled from `meta`. This works for
      both `global` and `blockwise` `norm_mode`.
    - SSIM / MS-SSIM are computed on the [0, 1]-normalized tensors (data_range=1.0).
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from magsr import ROOT_FOLDER
from magsr.datasets import build_ksa_aligned_datasets, pool_collate, worker_init_fn
from magsr.metrics import (
    apply_hr_nan_mask,
    mean_std,
    per_patch_msssim,
    per_patch_rmse,
    per_patch_ssim,
)
from magsr.models import BicubicModel, load_checkpoint

DEFAULT_CKPT = Path("checkpoints/rdnpp_x3_nb23_noadapt_cellgrid8_f3_best.pt")  # fold-3 baseline
# The canonical fold-3 benchmark split — always used. This eval never falls back to the legacy
# rect-holdout cut, whose test cells are NOT the fold-3 ones the reported numbers are computed on.
FOLD3_INDEX = Path("data/processed/ksa_aligned/patch_indices_cellgrid8_fold3")


def build_ksa_loaders(
    batch_size: int, num_workers: int, index_dir: Path = FOLD3_INDEX
) -> tuple[dict, dict[str, DataLoader]]:
    """Build train/val/test DataLoaders for the fold-based benchmark split (no legacy holdout)."""
    splits = build_ksa_aligned_datasets(index_dir=index_dir)
    aligned_cfg = next(iter(splits.values())).config
    collate = partial(
        pool_collate,
        hr_products=aligned_cfg.hr_products,
        lr_products=aligned_cfg.lr_products,
    )
    loaders = {
        split: DataLoader(
            ds,
            collate_fn=collate,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            persistent_workers=num_workers > 0,
        )
        for split, ds in splits.items()
    }
    return splits, loaders


METRIC_ROWS = ("rmse", "ssim", "msssim")
BLOCK_KEYS_DEFAULT: tuple[int | str, ...] = (1, 2, 3, "Net")


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | None = None,
    desc: str = "KSA",
    criterion: nn.Module | None = None,
    predict=None,
    to_nt=None,
    patch_size: int | None = None,
) -> dict[str, dict[int | str, tuple[float, float]]]:
    """Run `model` over `loader`; return `{metric: {block_or 'Net': (mean, std)}}`.

    `predict(batch, i) -> (sr01, hr01, blocks)` gets the [0,1]-normalized SR + HR (HR may hold NaN)
    and the per-patch block list; `to_nt(x01, blocks)` denormalizes to nT. Both default to the KSA
    real-LR / per-block-denorm path, so existing callers are unchanged — other modes (synthetic LR,
    WA global span, tanh corrector) just pass their own closures instead of copying the loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = loader.dataset
    patch_size = patch_size if patch_size is not None else int(ds.config.patch_px)
    multi = getattr(model, "num_domains", 1) > 1

    if predict is None:

        def predict(batch, _i):
            lr_t = batch["lr"].to(device, non_blocking=True)
            hr_t = batch["hr"][:, :1].to(device, non_blocking=True)
            blocks = [m["block"] for m in batch["meta"]]
            lr_n = ds.assemble_lr_input(
                lr_t,
                blocks,
                dem=batch["dem"].to(device) if ds.config.load_dem else None,
                lr_aux=batch["lr_aux"].to(device) if ds.lr_aux_products else None,
                ms=batch["ms"].to(device) if (ds.ms_bands or ds.ms_features) else None,
            )
            hr_n = ds.normalize(hr_t, blocks=blocks)
            blocks_t = torch.tensor(blocks, dtype=torch.long, device=device)
            sr_n = model(lr_n, blocks_t).clamp(0, 1) if multi else model(lr_n).clamp(0, 1)
            return sr_n, hr_n, blocks

    if to_nt is None:

        def to_nt(x, blocks):
            return ds.denormalize(x, blocks=blocks)

    patch_vals: dict[str, list[torch.Tensor]] = {"rmse": [], "ssim": [], "msssim": []}
    patch_blocks: dict[str, list[torch.Tensor]] = {"rmse": [], "ssim": [], "msssim": []}
    # Pooled normalized SSE / valid-pixel count per block, for the global-RMS RMSE
    # (and PSNR) over [0,1] data — the SAME quantity the training val loop reports, so
    # eval lines up with the val curve (distinct from the per-patch-mean nT "rmse" row).
    sse_norm: dict[int, float] = {}
    cnt_norm: dict[int, float] = {}
    loss_sum, n_loss = (
        0.0,
        0,
    )  # optional MaskedL1 over the split (matches training val_loss) when criterion given

    with torch.no_grad():
        for _i, batch in enumerate(tqdm(loader, desc=desc, ncols=100)):
            sr_n, hr_n, blocks = predict(batch, _i)
            blocks_t = torch.tensor(blocks, dtype=torch.long)

            if criterion is not None:
                # clamped SR vs normalized HR (with NaNs) — same as the training val loss
                loss_sum += criterion(sr_n, hr_n).item()
                n_loss += 1

            mask = hr_n.isfinite().to(sr_n.dtype)
            sr_m, hr_m = apply_hr_nan_mask(sr_n, hr_n)

            sr_nt = to_nt(sr_m, blocks) * mask
            hr_nt = to_nt(hr_m, blocks) * mask

            patch_vals["rmse"].append(per_patch_rmse(sr_nt, hr_nt, mask).cpu())
            patch_blocks["rmse"].append(blocks_t)

            # Pooled normalized SSE / valid-pixel count, accumulated per block.
            sse_p = ((sr_m - hr_m) ** 2).flatten(1).sum(dim=1)
            cnt_p = mask.flatten(1).sum(dim=1)
            for b, e, c in zip(blocks, sse_p.tolist(), cnt_p.tolist()):
                sse_norm[b] = sse_norm.get(b, 0.0) + e
                cnt_norm[b] = cnt_norm.get(b, 0.0) + c

            fully_valid = mask.flatten(1).all(dim=1)
            if fully_valid.any():
                sr_v = sr_m[fully_valid]
                hr_v = hr_m[fully_valid]
                blk_v = blocks_t[fully_valid.cpu()]
                patch_vals["ssim"].append(per_patch_ssim(sr_v, hr_v, data_range=1.0).cpu())
                patch_vals["msssim"].append(
                    per_patch_msssim(sr_v, hr_v, patch_size=patch_size, data_range=1.0).cpu()
                )
                patch_blocks["ssim"].append(blk_v)
                patch_blocks["msssim"].append(blk_v)

    seen_blocks = sorted({int(x) for v in patch_blocks["rmse"] for x in v.tolist()})

    table: dict[str, dict[int | str, tuple[float, float]]] = {m: {} for m in METRIC_ROWS}
    for metric in ("rmse", "ssim", "msssim"):
        if not patch_vals[metric]:
            continue
        vals = torch.cat(patch_vals[metric])
        blks = torch.cat(patch_blocks[metric])
        for b in seen_blocks:
            sel = blks == b
            if sel.any():
                table[metric][b] = mean_std(vals[sel])
        table[metric]["Net"] = mean_std(vals)

    # Pooled normalized RMSE + PSNR (global-RMS over valid pixels), matching the
    # training val metric. Reported alongside the per-patch-mean nT "rmse" row.
    if cnt_norm:
        import math

        table["rmse_norm"] = {}
        table["psnr"] = {}
        for b in (*seen_blocks, "Net"):
            if b == "Net":
                s, c = sum(sse_norm.values()), sum(cnt_norm.values())
            else:
                s, c = sse_norm.get(b, 0.0), cnt_norm.get(b, 0.0)
            if c > 0:
                r = math.sqrt(s / c)
                table["rmse_norm"][b] = (r, float("nan"))
                table["psnr"][b] = (10.0 * math.log10(1.0 / (r + 1e-12)), float("nan"))

    if criterion is not None and n_loss:
        table["loss"] = {"Net": (loss_sum / n_loss, float("nan"))}

    return table


def format_split_table(results: dict[str, dict], blocks: tuple[int | str, ...] = BLOCK_KEYS_DEFAULT) -> str:
    """One wide table: rows = metrics, columns = method × block × (mean, std)."""
    methods = list(results.keys())
    label_w = 12
    cell_w = 9  # width per scalar — fits e.g. "12345.67" with 4 decimals via .4g

    def fmt(v: float) -> str:
        return f"{v:<{cell_w}.4g}" if v == v else f"{'-':<{cell_w}}"

    block_w = 2 * cell_w
    method_w = block_w * len(blocks)

    line_top = " " * label_w + "".join(f"{str(m):<{method_w}}" for m in methods)
    line_blk = " " * label_w + "".join(
        "".join(f"{f'B{b}' if isinstance(b, int) else b:<{block_w}}" for b in blocks) for _ in methods
    )
    line_ms = " " * label_w + "".join(
        "".join(f"{'mean':<{cell_w}}{'std':<{cell_w}}" for _ in blocks) for _ in methods
    )
    width = max(len(line_top), len(line_blk), len(line_ms))
    lines = [line_top, line_blk, line_ms, "-" * width]
    for metric in METRIC_ROWS:
        row = f"{metric:<{label_w}}"
        for method in methods:
            mtab = results[method].get(metric, {})
            for b in blocks:
                if b in mtab:
                    mean, std = mtab[b]
                    row += fmt(mean) + fmt(std)
                else:
                    row += " " * cell_w + " " * cell_w
        lines.append(row)
    return "\n".join(lines)


def _patch_metrics(
    sr_norm: torch.Tensor,
    hr_norm: torch.Tensor,
    ds,
    blocks: list[int],
    patch_size: int,
) -> dict[str, float | None]:
    """Per-patch RMSE (nT, always) + SSIM/MS-SSIM (only if fully valid)."""
    mask = hr_norm.isfinite().to(sr_norm.dtype)
    sr_m, hr_m = apply_hr_nan_mask(sr_norm, hr_norm)
    sr_nt = ds.denormalize(sr_m, blocks=blocks) * mask
    hr_nt = ds.denormalize(hr_m, blocks=blocks) * mask

    out: dict[str, float | None] = {
        "RMSE": per_patch_rmse(sr_nt, hr_nt, mask).item(),
        "SSIM": None,
        "MS-SSIM": None,
    }
    if bool(mask.all()):
        out["SSIM"] = per_patch_ssim(sr_m, hr_m, data_range=1.0).item()
        out["MS-SSIM"] = per_patch_msssim(sr_m, hr_m, patch_size=patch_size, data_range=1.0).item()
    return out


def _resolve_blocks(results: dict[str, dict]) -> tuple[int | str, ...]:
    seen = sorted({b for r in results.values() for mtab in r.values() for b in mtab if isinstance(b, int)})
    return (*seen, "Net")


def peek_reconstructions(
    ckpt_path: Path, run_name: str, indices_per_block: int = 2, index_dir: Path = FOLD3_INDEX
) -> None:
    """Save HR/LR/SR + signed-error PNGs for a few patches per block from the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rdn = load_checkpoint(ckpt_path, device=device)
    multi = getattr(rdn, "num_domains", 1) > 1
    bicubic = BicubicModel(upscale_factor=3).to(device)

    save_dir = ROOT_FOLDER / "figures" / "ksa_reconstructions" / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = build_ksa_aligned_datasets(index_dir=index_dir)["test"]
    patch_size = int(ds.config.patch_px)

    by_block: dict[int, list[int]] = {}
    for i, t in enumerate(ds.index.patches):
        b = ds._block_from_patch(t)
        by_block.setdefault(b, []).append(i)
    pick: list[int] = []
    for b, idxs in sorted(by_block.items()):
        step = max(1, len(idxs) // (indices_per_block + 1))
        pick.extend(idxs[step::step][:indices_per_block])

    for idx in pick:
        sample = ds[idx]
        hr = sample["hr"]["AMF_RTP"][None, None].to(device)
        lr = sample["lr"]["RTP"][None, None].to(device)
        block = int(sample["meta"]["block"])
        name = sample["meta"]["name"]
        blocks = [block]

        lr_n = ds.normalize(lr, blocks=blocks).nan_to_num(0.0)
        hr_n = ds.normalize(hr, blocks=blocks)

        with torch.no_grad():
            blocks_t = torch.tensor(blocks, device=device, dtype=torch.long) if multi else None
            sr_rdn = (rdn(lr_n, blocks_t) if multi else rdn(lr_n)).clamp(0, 1)
            sr_bic = bicubic(lr_n).clamp(0, 1)

        m_bic = _patch_metrics(sr_bic, hr_n, ds, blocks, patch_size)
        m_rdn = _patch_metrics(sr_rdn, hr_n, ds, blocks, patch_size)

        def to_nt(t_norm: torch.Tensor) -> np.ndarray:
            return ds.denormalize(t_norm, blocks=blocks).squeeze().detach().cpu().numpy()

        img_lr = to_nt(lr_n)
        img_hr = to_nt(hr_n)
        img_rdn = to_nt(sr_rdn)
        img_bic = to_nt(sr_bic)
        err_bic = img_hr - img_bic
        err_rdn = img_hr - img_rdn

        fig, axes = plt.subplots(2, 3, figsize=(14, 10))

        clim = dict(vmin=float(np.nanmin(img_hr)), vmax=float(np.nanmax(img_hr)))
        max_err = float(
            max(
                np.nanmax(np.abs(err_bic)) if np.isfinite(err_bic).any() else 0.0,
                np.nanmax(np.abs(err_rdn)) if np.isfinite(err_rdn).any() else 0.0,
            )
        )

        def _title(label: str, m: dict[str, float | None]) -> str:
            line = f"RMSE {m['RMSE']:.2f}"
            if m["SSIM"] is not None:
                line += f"   SSIM {m['SSIM']:.3f}   MS-SSIM {m['MS-SSIM']:.3f}"
            return f"{label}\n{line}"

        im_data = axes[0, 0].imshow(img_lr, cmap="gray", **clim)
        axes[0, 0].set_title("LR (Input)", fontweight="bold")
        axes[1, 0].imshow(img_hr, cmap="gray", **clim)
        axes[1, 0].set_title("HR (Target)", fontweight="bold")

        axes[0, 1].imshow(img_bic, cmap="gray", **clim)
        axes[0, 1].set_title(_title("Bicubic", m_bic), fontweight="bold")
        axes[1, 1].imshow(img_rdn, cmap="gray", **clim)
        axes[1, 1].set_title(_title("RDN++", m_rdn), fontweight="bold")

        im_err = axes[0, 2].imshow(err_bic, cmap="RdBu_r", vmin=-max_err, vmax=max_err)
        axes[0, 2].set_title("Bicubic Error", fontweight="bold")
        axes[1, 2].imshow(err_rdn, cmap="RdBu_r", vmin=-max_err, vmax=max_err)
        axes[1, 2].set_title("RDN++ Error", fontweight="bold")

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

        fig.subplots_adjust(right=0.85, wspace=0.02, hspace=0.08)
        cbar_ax_data = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar_data = fig.colorbar(im_data, cax=cbar_ax_data)
        cbar_data.set_label("Magnetic Field (nT)", fontweight="bold")
        cbar_ax_err = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar_err = fig.colorbar(im_err, cax=cbar_ax_err)
        cbar_err.set_label("Error (nT)", fontweight="bold")

        fig.suptitle(f"B{block} Patch #{idx}: {name}", fontweight="bold", fontsize=16)

        save_path = save_dir / f"B{block}_patch_{idx}_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved detailed comparison: {save_path}")
        plt.close(fig)


def _ckpt_variant(stem: str) -> str:
    """Tag for the wandb summary namespace, derived from the checkpoint filename."""
    if stem.endswith("_best"):
        return "best"
    if stem.endswith("_last"):
        return "last"
    return "ckpt"


def _attach_to_wandb(
    ckpt_path: Path,
    ckpt_dict: dict,
    all_results: dict[str, dict],
    *,
    run_id_override: str | None,
) -> None:
    """Write RDN++ eval metrics into the training wandb run's summary."""
    run_id = run_id_override or ckpt_dict.get("wandb_run_id")
    if not run_id:
        print("No wandb_run_id in checkpoint and no --wandb-run-id given; skipping wandb summary.")
        return
    cfg = ckpt_dict.get("config") or {}
    project = cfg.get("wandb_project")
    entity = cfg.get("wandb_entity")
    if not project:
        print("Checkpoint config missing wandb_project; skipping wandb summary.")
        return

    variant = _ckpt_variant(ckpt_path.stem)
    summary: dict[str, float] = {}
    for split, results in all_results.items():
        rdn_table = results.get("RDN++", {})
        for metric, scopes in rdn_table.items():
            for scope, (mean, std) in scopes.items():
                scope_str = "all" if scope == "Net" else f"B{scope}"
                base = f"eval_{variant}/{split}/{metric}/{scope_str}"
                summary[base] = mean
                summary[f"{base}_std"] = std

    run = wandb.init(id=run_id, project=project, entity=entity, resume="must")
    try:
        run.summary.update(summary)
        print(f"Attached {len(summary)} metrics to wandb run {run_id} under eval_{variant}/*.")
    finally:
        run.finish()


def evaluate_models() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=FOLD3_INDEX,
        help="Patch index dir. Defaults to the fold-3 benchmark split, which is "
        "what the reported test numbers are computed on.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["train", "val", "test"],
        help="Which KSA-aligned splits to evaluate.",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Also save per-patch reconstruction PNGs for a few test-split patches per block.",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="Resume this wandb run and attach eval metrics to its summary. "
        "Defaults to the id stored in the checkpoint (if any).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rdn, ckpt_dict = load_checkpoint(args.ckpt, device=device, return_ckpt=True)
    bicubic = BicubicModel(upscale_factor=3).to(device)

    _, loaders = build_ksa_loaders(args.batch_size, args.num_workers, index_dir=args.index_dir)
    save_dir = ROOT_FOLDER / "results" / "ksa_aligned_evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{args.ckpt.stem}_results.csv"
    csv_new = not csv_path.exists()

    print(f"\nCheckpoint: {args.ckpt}")
    all_results: dict[str, dict] = {}
    with open(csv_path, "a") as csv_f:
        if csv_new:
            csv_f.write("checkpoint,split,method,scope,metric,mean,std\n")
        for split in args.splits:
            results = {
                "Bicubic": evaluate_loader(bicubic, loaders[split], device=device, desc=f"Bicubic {split}"),
                "RDN++": evaluate_loader(rdn, loaders[split], device=device, desc=f"RDN++ {split}"),
            }
            all_results[split] = results
            block_keys = _resolve_blocks(results)
            table_str = format_split_table(results, blocks=block_keys)
            print(f"\n=== KSA aligned {split} ===")
            print(table_str)

            txt_path = save_dir / f"{args.ckpt.stem}_{split}_results.txt"
            txt_path.write_text(f"Checkpoint: {args.ckpt}\nSplit: {split}\n\n{table_str}\n")
            for method, mtable in results.items():
                for metric, scopes in mtable.items():
                    for b, (mean, std) in scopes.items():
                        scope = f"B{b}" if isinstance(b, int) else str(b)
                        csv_f.write(
                            f"{args.ckpt.stem},{split},{method},{scope},{metric},{mean:.6f},{std:.6f}\n"
                        )
            print(f"Saved: {txt_path}")
    print(f"Appended CSV rows: {csv_path}")

    _attach_to_wandb(args.ckpt, ckpt_dict, all_results, run_id_override=args.wandb_run_id)

    if args.peek:
        peek_reconstructions(args.ckpt, args.ckpt.stem, index_dir=args.index_dir)


if __name__ == "__main__":
    evaluate_models()

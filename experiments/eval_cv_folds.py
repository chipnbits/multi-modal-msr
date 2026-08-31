"""Evaluate the cellgrid8 5-fold CV checkpoints on their matching splits.

For every fold f in {0..4} and every checkpoint variant in
{best (best val SSIM), best_rmse (best val RMSE), last (final epoch)}, run
inference over that fold's own train / val / test splits and record per-patch
RMSE (nT) / SSIM / MS-SSIM (mean +/- std, per SGS block B1/B2/B3 and overall
"Net"). That is 5 folds * 3 variants * 3 splits = 45 model evaluations.

Outputs (under results/cv_folds_eval/):
    - <prefix>_inference.csv: long-format table, one row per
      (fold, variant, split, method, scope, metric).
    - <prefix>_epoch_tracking.csv: one row per fold summarizing which epoch hit
      the best val SSIM vs best val RMSE (read from the full per-epoch `hist`
      stored in the _last.pt checkpoint) and the val metrics there.

Reuses `evaluate_loader` from ksa_aligned_evaluate. Each fold's DataLoaders are
built once and shared across the 3 variants. A bicubic baseline is evaluated
once per (fold, split) for reference.

Note on RMSE units: the eval RMSE here is per-patch RMSE in nT (denormalized),
whereas the `val_rmse` stored in the checkpoint / `hist` is the normalized
[0,1] validation RMSE used for model selection. They are different quantities;
epoch selection uses the normalized hist, the inference table reports nT.

Usage:
    uv run python experiments/eval_cv_folds.py
    uv run python experiments/eval_cv_folds.py --folds 0 1 2 --splits val test
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

import numpy as np
import torch
from ksa_aligned_evaluate import evaluate_loader
from torch.utils.data import DataLoader

from magsr import ROOT_FOLDER
from magsr.cli import add_channel_args
from magsr.datasets import build_ksa_aligned_datasets, pool_collate, worker_init_fn
from magsr.models import SOUP_VARIANT, BicubicModel, ensure_soup, load_checkpoint

VARIANTS = ("best", "best_rmse", "last", SOUP_VARIANT)
VARIANT_DESC = {
    "best": "best val SSIM",
    "best_rmse": "best val RMSE",
    "last": "final epoch",
    SOUP_VARIANT: "0.5*best_ssim + 0.5*best_rmse soup",
}


def build_fold_loaders(
    index_dir: Path,
    batch_size: int,
    num_workers: int,
    *,
    load_dem: bool = False,
    dem_mode: str = "grad",
    lr_aux: tuple[str, ...] = (),
) -> dict[str, DataLoader]:
    """train/val/test DataLoaders for one fold's pre-baked split JSONs.

    Pass load_dem / lr_aux to evaluate a multi-channel (in=4 combo) model — evaluate_loader
    concatenates the DEM-gradient / aux channels in the same order as training.
    """
    splits = build_ksa_aligned_datasets(
        index_dir=index_dir, load_dem=load_dem, dem_mode=dem_mode, lr_aux_products=lr_aux
    )
    aligned_cfg = next(iter(splits.values())).config
    collate = partial(
        pool_collate,
        hr_products=aligned_cfg.hr_products,
        lr_products=aligned_cfg.lr_products,
    )
    return {
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


def epoch_tracking_row(fold: int, ckpt_dir: Path, prefix: str) -> dict:
    """Derive best-val epochs for a fold from the full hist in its _last.pt."""
    last = torch.load(ckpt_dir / f"{prefix}{fold}_last.pt", map_location="cpu", weights_only=False)
    hist = last["hist"]
    epochs = hist["epochs"]
    ssim = hist["val_ssim"]
    rmse = hist["val_rmse"]
    # Validation runs every few epochs; non-validated epochs are NaN in hist.
    i_ssim = int(np.nanargmax(ssim))  # best (max) val SSIM
    i_rmse = int(np.nanargmin(rmse))  # best (min) val RMSE
    return {
        "fold": fold,
        "n_epochs": epochs[-1],
        "best_ssim_epoch": epochs[i_ssim],
        "best_ssim_val_ssim": ssim[i_ssim],
        "best_ssim_val_rmse": rmse[i_ssim],
        "best_rmse_epoch": epochs[i_rmse],
        "best_rmse_val_rmse": rmse[i_rmse],
        "best_rmse_val_ssim": ssim[i_rmse],
        # negative => RMSE-best happens earlier than SSIM-best
        "rmse_minus_ssim_epoch": epochs[i_rmse] - epochs[i_ssim],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--prefix", default="rdnpp_x3_16mp_cellgrid8_f")
    p.add_argument(
        "--index-root",
        type=Path,
        default=Path("data/processed/ksa_aligned"),
        help="Holds patch_indices_cellgrid8_fold{f}/ per-fold split JSONs.",
    )
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"]
    )
    p.add_argument("--variants", nargs="+", default=["best", "best_rmse", "last"], choices=list(VARIANTS))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--no-bicubic", action="store_true", help="Skip the bicubic baseline.")
    add_channel_args(p)
    p.add_argument(
        "--append",
        action="store_true",
        help="Append to the existing inference CSV instead of overwriting, "
        "and skip rewriting the epoch-tracking CSV.",
    )
    p.add_argument(
        "--out-tag",
        default="",
        help="Write to a complementary results dir results/cv_folds_<tag>_eval/ "
        "instead of results/cv_folds_eval/ (e.g. --out-tag soups for the "
        "soup models). Also skips the (training-derived) epoch-tracking CSV.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub = f"cv_folds_{args.out_tag}_eval" if args.out_tag else "cv_folds_eval"
    out_dir = ROOT_FOLDER / "results" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    inf_csv = out_dir / f"{args.prefix}inference.csv"
    epoch_csv = out_dir / f"{args.prefix}epoch_tracking.csv"

    bicubic = None if args.no_bicubic else BicubicModel(upscale_factor=3).to(device)

    # Per-fold epoch tracking (a property of training, independent of which
    # checkpoint we score). Skipped in --append mode and for complementary
    # out-tag runs (the main cv_folds_eval run already captured it).
    if not args.append and not args.out_tag:
        epoch_rows = [epoch_tracking_row(f, args.ckpt_dir, args.prefix) for f in args.folds]
        with open(epoch_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
            w.writeheader()
            w.writerows(epoch_rows)
        print(f"\nWrote epoch tracking: {epoch_csv}")
        print(
            f"{'fold':>4}  {'n_ep':>4}  {'ssim@':>6} {'(val_ssim)':>10}  "
            f"{'rmse@':>6} {'(val_rmse)':>11}  {'rmse-ssim':>9}"
        )
        for r in epoch_rows:
            print(
                f"{r['fold']:>4}  {r['n_epochs']:>4}  {r['best_ssim_epoch']:>6} "
                f"{r['best_ssim_val_ssim']:>10.4f}  {r['best_rmse_epoch']:>6} "
                f"{r['best_rmse_val_rmse']:>11.5f}  {float(r['rmse_minus_ssim_epoch']):>+9.2f}"
            )

    fields = [
        "fold",
        "variant",
        "variant_desc",
        "ckpt_epoch",
        "split",
        "method",
        "scope",
        "metric",
        "mean",
        "std",
    ]
    with open(inf_csv, "a" if args.append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not args.append:
            w.writeheader()

        for fold in args.folds:
            index_dir = args.index_root / f"patch_indices_cellgrid8_fold{fold}"
            print(f"\n{'='*70}\nFOLD {fold}  ({index_dir})\n{'='*70}")
            loaders = build_fold_loaders(
                index_dir,
                args.batch_size,
                args.num_workers,
                load_dem=args.use_dem,
                dem_mode=args.dem_mode,
                lr_aux=tuple(args.lr_aux),
            )

            # Bicubic baseline: depends only on (fold, split), not the variant.
            if bicubic is not None:
                for split in args.splits:
                    tab = evaluate_loader(
                        bicubic, loaders[split], device=device, desc=f"f{fold} bicubic {split}"
                    )
                    for metric, scopes in tab.items():
                        for scope, (mean, std) in scopes.items():
                            sc = f"B{scope}" if isinstance(scope, int) else str(scope)
                            w.writerow(
                                {
                                    "fold": fold,
                                    "variant": "bicubic",
                                    "variant_desc": "bicubic baseline",
                                    "ckpt_epoch": "",
                                    "split": split,
                                    "method": "Bicubic",
                                    "scope": sc,
                                    "metric": metric,
                                    "mean": f"{mean:.6f}",
                                    "std": f"{std:.6f}",
                                }
                            )
                    f.flush()

            for variant in args.variants:
                if variant == SOUP_VARIANT:  # derived: build from _best + _best_rmse if absent
                    ensure_soup(args.ckpt_dir, f"{args.prefix}{fold}")
                ckpt = args.ckpt_dir / f"{args.prefix}{fold}_{variant}.pt"
                rdn, ck = load_checkpoint(ckpt, device=device, return_ckpt=True)
                epoch = ck["epoch"]
                print(f"\n--- fold {fold} / {variant} ({VARIANT_DESC[variant]}, " f"epoch {epoch}) ---")
                for split in args.splits:
                    tab = evaluate_loader(
                        rdn, loaders[split], device=device, desc=f"f{fold} {variant} {split}"
                    )
                    for metric, scopes in tab.items():
                        for scope, (mean, std) in scopes.items():
                            sc = f"B{scope}" if isinstance(scope, int) else str(scope)
                            w.writerow(
                                {
                                    "fold": fold,
                                    "variant": variant,
                                    "variant_desc": VARIANT_DESC[variant],
                                    "ckpt_epoch": epoch,
                                    "split": split,
                                    "method": "RDN++",
                                    "scope": sc,
                                    "metric": metric,
                                    "mean": f"{mean:.6f}",
                                    "std": f"{std:.6f}",
                                }
                            )
                    f.flush()
                    net = tab.get("rmse", {}).get("Net")
                    ssim = tab.get("ssim", {}).get("Net")
                    if net and ssim:
                        print(f"    {split:>5}: Net RMSE {net[0]:.3f} nT   " f"SSIM {ssim[0]:.4f}")
                del rdn
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print(f"\nWrote inference table: {inf_csv}")


if __name__ == "__main__":
    main()

"""Score the nb=13 multi-modal stride-12 runs against the recent baselines, all on
ONE common test set: the canonical fold-3 benchmark split
(`patch_indices_cellgrid8_fold3`, stride 66 / 2309 test patches). The multi-modal
models were *trained* on the dense stride-12 repatch of the SAME cells, so evaluating
on the original (sparser) test patches over those identical held-out cells is a clean,
directly-comparable held-out score — and it's the exact set the trunk-depth / adapter
baselines were scored on.

Each run's auxiliary input channels (DEM and/or LR aux products like 1VD/ANS) are
reconstructed from its checkpoint's stored training config, so the dataset is built with
the matching channels and `evaluate_loader` concatenates them just as in training.
Handles both the pre-refactor `use_1vd` flag and the generalized `lr_aux` list.

Per-patch RMSE (nT) / SSIM / MS-SSIM (mean +/- std, per block B1/B2/B3 and Net) go to
results/f3_multimodal_eval/inference.csv, with a Net-RMSE leaderboard printed at the end.

Usage:
    uv run python experiments/eval_f3_multimodal.py
    uv run python experiments/eval_f3_multimodal.py --splits val test --variants best best_rmse last
"""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

import torch
from ksa_aligned_evaluate import evaluate_loader
from torch.utils.data import DataLoader

from magsr import ROOT_FOLDER
from magsr.datasets import build_ksa_aligned_datasets, pool_collate, worker_init_fn
from magsr.models import SOUP_VARIANT, BicubicModel, ensure_soup, load_checkpoint

# (run_name, friendly label) — ordered baseline first, then mag-only, then modalities.
RUNS: list[tuple[str, str]] = [
    ("rdnpp_x3_ksa_f3_nb13", "baseline nb13 (plain, stride66)"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12", "mag only (dense s12)"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_dem", "mag + DEM"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_1vd", "mag + 1VD"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_ans", "mag + ANS"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_1vd_ans", "mag + 1VD + ANS"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_1vd_ans_dem", "mag + 1VD + ANS + DEM"),
    ("rdnpp_x3_ksa_f3_nb8_k0tail_dwd2e-3_s12_1vd", "mag + 1VD (nb8)"),
    ("rdnpp_x3_ksa_f3_nb8_k0tail_dwd2e-3_s12_1vd_ans_dem", "mag + 1VD + ANS + DEM (nb8)"),
    ("rdnpp_x3_ksa_f3_nb8_nf96_k0tail_dwd2e-3_s12_1vd_ans_dem", "full combo (nb8, nf96)"),
    ("rdnpp_x3_ksa_f3_nb18_nf44_k0tail_dwd2e-3_s12_1vd", "mag + 1VD (nb18, nf44 ~ nb13 params)"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_demgrad", "mag + DEMgrad alone (nb13)"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_ferrous", "mag + ferrous ratio b6/b5 (nb13)"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_1vd_demgrad", "mag + 1VD + DEMgrad (nb13)"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_fvd5", "mag only + FVD loss λ=5 (nb13)"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_fvd50", "mag only + FVD loss λ=50 (nb13)"),
    (
        "rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_1vd_demgrad_fvd5",
        "mag + 1VD + DEMgrad input + FVD loss λ=5 (nb13)",
    ),
    (
        "rdnpp_x3_ksa_f3_nb8_k0tail_dwd2e-3_s12_1vd_demgrad_fvd5",
        "mag + 1VD + DEMgrad input + FVD loss λ=5 (nb8)",
    ),
    (
        "rdnpp_x3_ksa_f3_nb18_k0tail_dwd2e-3_s12_1vd_demgrad_fvd5",
        "mag + 1VD + DEMgrad input + FVD loss λ=5 (nb18)",
    ),
    (
        "rdnpp_x3_ksa_f3_nb13_k1_dwd2e-3_s12_1vd_demgrad_fvd5",
        "mag + 1VD + DEMgrad input + FVD loss λ=5 (nb13, k1)",
    ),
    (
        "rdnpp_x3_ksa_f3_nb23_k0tail_dwd2e-3_s12_1vd_demgrad_fvd5",
        "mag + 1VD + DEMgrad input + FVD loss λ=5 (nb23)",
    ),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_lr66", "mag only, LR66/HR198 context (nb13)"),
    ("rdnpp_x3_ksa_f3_nb13_k0tail_dwd2e-3_s12_lr88", "mag only, LR88/HR264 context (nb13)"),
]


def run_complete(ckpt_dir: Path, run: str) -> bool:
    """True iff the run's _last.pt exists and reached its configured final epoch — so we
    skip runs still training (partial best ckpts, risk of reading a half-written file)."""
    last = ckpt_dir / f"{run}_last.pt"
    if not last.exists():
        return False
    try:
        ck = torch.load(last, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return int(ck.get("epoch", 0)) >= int((ck.get("config") or {}).get("num_epochs", 1))


def channels_from_cfg(cfg: dict):
    """Reconstruct (use_dem, dem_mode, lr_aux_products, ms_bands, ms_features) from a
    checkpoint config, handling both the pre-refactor `use_1vd` flag and the `lr_aux` list."""
    aux = list(cfg.get("lr_aux") or [])
    if cfg.get("use_1vd") and "1VD" not in aux:
        aux = ["1VD", *aux]
    ms = tuple(cfg.get("ms_bands") or [])
    msf = tuple(cfg.get("ms_features") or [])
    return bool(cfg.get("use_dem")), cfg.get("dem_mode") or "relief", tuple(aux), ms, msf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/processed/ksa_aligned/patch_indices_cellgrid8_fold3"),
        help="Common test set (canonical stride-66 fold-3 benchmark).",
    )
    p.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    p.add_argument(
        "--variants",
        nargs="+",
        default=["best", "best_rmse", "last", SOUP_VARIANT],
        help=f"Checkpoint variants. '{SOUP_VARIANT}' is built on the fly as the 50/50 "
        "weight average of _best (val SSIM) and _best_rmse (val RMSE).",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--no-bicubic", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT_FOLDER / "results" / "f3_multimodal_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    inf_csv = out_dir / "inference.csv"

    # Cache loaders by channel config (load_dem, dem_mode, lr_aux) so we open each raster set once.
    loader_cache: dict[tuple, dict[str, DataLoader]] = {}

    def get_loaders(use_dem, dem_mode, aux, ms, msf):
        key = (use_dem, dem_mode, aux, ms, msf)
        if key not in loader_cache:
            splits = build_ksa_aligned_datasets(
                index_dir=args.index_dir,
                load_dem=use_dem,
                dem_mode=dem_mode,
                lr_aux_products=aux,
                ms_bands=ms,
                ms_features=msf,
            )
            cfg0 = next(iter(splits.values())).config
            collate = partial(pool_collate, hr_products=cfg0.hr_products, lr_products=cfg0.lr_products)
            loader_cache[key] = {
                s: DataLoader(
                    ds,
                    collate_fn=collate,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    worker_init_fn=worker_init_fn,
                    shuffle=False,
                    persistent_workers=args.num_workers > 0,
                )
                for s, ds in splits.items()
            }
        return loader_cache[key]

    fields = [
        "run",
        "label",
        "variant",
        "ckpt_epoch",
        "in_channels",
        "split",
        "method",
        "scope",
        "metric",
        "mean",
        "std",
    ]
    rows: list[dict] = []
    # label, variant, nT per-patch RMSE, normalized pooled RMSE, PSNR, SSIM, MS-SSIM
    leaderboard: list[tuple] = []

    def net(tab: dict, metric: str) -> float:
        return tab.get(metric, {}).get("Net", (float("nan"),))[0]

    # Bicubic reference (single-channel; depends only on split).
    if not args.no_bicubic:
        bic = BicubicModel(upscale_factor=3).to(device)
        loaders = get_loaders(False, "relief", (), (), ())
        for split in args.splits:
            tab = evaluate_loader(bic, loaders[split], device=device, desc=f"bicubic {split}")
            for metric, scopes in tab.items():
                for scope, (mean, std) in scopes.items():
                    sc = f"B{scope}" if isinstance(scope, int) else str(scope)
                    rows.append(
                        {
                            "run": "bicubic",
                            "label": "bicubic",
                            "variant": "",
                            "ckpt_epoch": "",
                            "in_channels": 1,
                            "split": split,
                            "method": "Bicubic",
                            "scope": sc,
                            "metric": metric,
                            "mean": f"{mean:.6f}",
                            "std": f"{std:.6f}",
                        }
                    )
            if split == "test":
                leaderboard.append(
                    (
                        "bicubic",
                        "-",
                        net(tab, "rmse"),
                        net(tab, "rmse_norm"),
                        net(tab, "psnr"),
                        net(tab, "ssim"),
                        net(tab, "msssim"),
                    )
                )

    for run, label in RUNS:
        if not run_complete(args.ckpt_dir, run):
            print(f"[skip] {label} ({run}) — not finished training yet")
            continue
        for variant in args.variants:
            if variant == SOUP_VARIANT:
                ckpt = ensure_soup(args.ckpt_dir, run)
                if ckpt is None:
                    print(f"[skip] {run} {SOUP_VARIANT} — need _best and _best_rmse")
                    continue
            else:
                ckpt = args.ckpt_dir / f"{run}_{variant}.pt"
                if not ckpt.exists():
                    print(f"[skip] {ckpt.name} not found (run still training?)")
                    continue
            rdn, ck = load_checkpoint(ckpt, device=device, return_ckpt=True)
            use_dem, dem_mode, aux, ms, msf = channels_from_cfg(ck.get("config") or {})
            dem_ch = (2 if dem_mode == "grad" else 1) if use_dem else 0
            in_ch = 1 + dem_ch + len(aux) + len(ms) + len(msf)
            loaders = get_loaders(use_dem, dem_mode, aux, ms, msf)
            print(
                f"\n--- {label} / {variant} (epoch {ck.get('epoch')}, in={in_ch}, "
                f"dem={use_dem}{'/'+dem_mode if use_dem else ''}, aux={aux or '-'}, "
                f"ms={ms or '-'}, msf={msf or '-'}) ---"
            )
            for split in args.splits:
                tab = evaluate_loader(rdn, loaders[split], device=device, desc=f"{run} {variant} {split}")
                for metric, scopes in tab.items():
                    for scope, (mean, std) in scopes.items():
                        sc = f"B{scope}" if isinstance(scope, int) else str(scope)
                        rows.append(
                            {
                                "run": run,
                                "label": label,
                                "variant": variant,
                                "ckpt_epoch": ck.get("epoch"),
                                "in_channels": in_ch,
                                "split": split,
                                "method": "RDN++",
                                "scope": sc,
                                "metric": metric,
                                "mean": f"{mean:.6f}",
                                "std": f"{std:.6f}",
                            }
                        )
                if split == "test":
                    rn, ps = net(tab, "rmse_norm"), net(tab, "psnr")
                    leaderboard.append(
                        (label, variant, net(tab, "rmse"), rn, ps, net(tab, "ssim"), net(tab, "msssim"))
                    )
                    print(
                        f"    test: Net RMSE {net(tab,'rmse'):.2f} nT   norm-RMSE {rn:.4f}  "
                        f"PSNR {ps:.2f}   SSIM {net(tab,'ssim'):.4f}   MS-SSIM {net(tab,'msssim'):.4f}"
                    )
            del rdn
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with open(inf_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {inf_csv}")

    # Net test-RMSE leaderboard (ascending RMSE = best first).
    leaderboard.sort(key=lambda t: t[2])
    print("\n" + "=" * 78)
    print("TEST leaderboard (Net, canonical stride-66 fold-3 — lower RMSE better)")
    print("=" * 78)
    print(
        f"{'model':38} {'variant':10} {'RMSE(nT)':>9} {'nRMSE':>7} {'PSNR':>6} {'SSIM':>7} {'MS-SSIM':>8}"
    )
    print("-" * 90)
    for label, variant, r, rn, ps, s, m in leaderboard:
        print(f"{label:38} {variant:10} {r:9.2f} {rn:7.4f} {ps:6.2f} {s:7.4f} {m:8.4f}")


if __name__ == "__main__":
    main()

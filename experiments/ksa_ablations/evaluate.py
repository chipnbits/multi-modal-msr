"""End-to-end evaluation for the KSA ablation tables -> the complete results/ksa_ablations_eval.csv.

Three scorers, one command:
  1. model runs  -> sweep_all_models.py: every non-synuc checkpoint on its own fold/dataset (KSA
                    fold-N, WA x4), through the shared real-LR `evaluate_loader`.
  2. bicubic     -> BicubicModel through the same `evaluate_loader` (KSA folds 0-4 + WA).
  3. synuc grid  -> eval_synuc.score_synuc on the synthetic-operator checkpoints. These need the
                    seeded SYNTHETIC LR (not the real LR), written to the same CSV.

Run:  uv run python experiments/ksa_ablations/evaluate.py               # full: sweep + bicubic + synuc
      uv run python experiments/ksa_ablations/evaluate.py --skip-sweep  # reuse model rows (bicubic + synuc)
      uv run python experiments/ksa_ablations/evaluate.py --skip-synuc  # reuse synuc rows (sweep + bicubic)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from magsr import ROOT_FOLDER

sys.path.insert(0, str(ROOT_FOLDER / "experiments"))
from eval_synuc import score_synuc
from ksa_aligned_evaluate import evaluate_loader

from magsr.datasets import (
    build_ksa_aligned_datasets,
    build_wa_datasets,
    pool_collate,
    worker_init_fn,
)
from magsr.models import BicubicModel

# Master csv for eval results (sweep + bicubic + synuc).
OUT = ROOT_FOLDER / "results/ksa_ablations_eval.csv"

FIELDS = [
    "run",
    "variant",
    "model",
    "dataset",
    "fold",
    "in_ch",
    "use_dem",
    "lr_aux",
    "ckpt_epoch",
    "split",
    "scope",
    "metric",
    "mean",
    "std",
]

# KSA bicubic reference grids (per fold basis)
KSA_BICUBIC = {f"bicubic_ksa_f{n}": (f"patch_indices_cellgrid8_fold{n}", f"ksa_f{n}", n) for n in range(5)}


def _rows(run, dataset, fold, split, tab):
    out = []
    for metric, scopes in tab.items():
        for scope, ms in scopes.items():
            sc = f"B{scope}" if isinstance(scope, int) else str(scope)
            out.append(
                dict(
                    run=run,
                    variant="bicubic",
                    model="bicubic",
                    dataset=dataset,
                    fold=fold,
                    in_ch=1,
                    use_dem=False,
                    lr_aux="-",
                    ckpt_epoch="",
                    split=split,
                    scope=sc,
                    metric=metric,
                    mean=f"{ms[0]:.6f}",
                    std=f"{ms[1]:.6f}",
                )
            )
    return out


def _ksa_loaders(index_name, bs=128, nw=8):
    splits = build_ksa_aligned_datasets(index_dir=ROOT_FOLDER / "data/processed/ksa_aligned" / index_name)
    acfg = next(iter(splits.values())).config
    collate = partial(pool_collate, hr_products=acfg.hr_products, lr_products=acfg.lr_products)
    return {
        s: DataLoader(
            d,
            collate_fn=collate,
            batch_size=bs,
            num_workers=nw,
            worker_init_fn=worker_init_fn,
            shuffle=False,
        )
        for s, d in splits.items()
    }


def _wa_bicubic(loader, model, device, ps):
    """WA bicubic via the canonical evaluate_loader. WA is single-region (one pseudo-block) with a
    global nT span instead of per-block denorm, so it just passes a `predict` + span `to_nt`."""
    span = 717.32 - (-329.35)  # WA nT normalization span

    def predict(batch, _i):
        lr, hr = batch["lr"].to(device), batch["hr"].to(device)
        return model(lr).clamp(0, 1), hr, [0] * lr.shape[0]

    tab = evaluate_loader(
        model,
        loader,
        device=device,
        desc="bicubic_wa",
        predict=predict,
        to_nt=lambda x, _b: x * span,
        patch_size=ps,
    )
    return {m: {"Net": tab[m].get("Net", (float("nan"), float("nan")))} for m in ("rmse", "ssim", "msssim")}


def score_bicubic(device):
    """Bicubic reference rows for KSA folds 0-4 (x3) and WA (x4), via evaluate_loader."""
    rows = []
    bic3 = BicubicModel(upscale_factor=3).to(device)
    for run, (idx, ds, fold) in KSA_BICUBIC.items():
        ld = _ksa_loaders(idx)
        for split in ("val", "test"):
            rows += _rows(
                run, ds, fold, split, evaluate_loader(bic3, ld[split], device=device, desc=f"{run} {split}")
            )
    bic4 = BicubicModel(upscale_factor=4).to(device)
    wa = build_wa_datasets()
    for split in ("val", "test"):
        ld = DataLoader(
            wa[split],
            collate_fn=pool_collate,
            batch_size=128,
            num_workers=8,
            worker_init_fn=worker_init_fn,
            shuffle=False,
        )
        ps = int(wa[split][0]["hr"][wa[split].product].shape[-1])
        rows += _rows("bicubic_wa", "wa", "", split, _wa_bicubic(ld, bic4, device, ps))
    return rows


# Synthetic-operator (ideal) study: 12 checkpoints (flat + drape x 6 stacks), scored on seeded
# synthetic pairs -> exp_ideal table.
SYNUC_INDEX = ROOT_FOLDER / "data/processed/ksa_aligned/patch_indices_cellgrid8s12_p264_fold3"
SYNUC_STACKS = ["mag", "1vd", "demgrad", "1vd_demgrad", "demrelief", "relief_1vd"]


def score_synuc_grid() -> list[dict]:
    """Score the 12 synuc checkpoints via eval_synuc.score_synuc; Net rmse/ssim/msssim + per-block
    RMSE (the columns the ideal table reads)."""
    rows = []
    for pfx, drape in (("synuc", False), ("synucd", True)):
        for stack in SYNUC_STACKS:
            run = f"rdnpp_x3_ksa_f3_nb13_k0_p264_{pfx}_{stack}"
            ckpt = ROOT_FOLDER / f"checkpoints/{run}_best_rmse.pt"
            if not ckpt.exists():
                print(f"  synuc: no checkpoint for {run} — skipped (train sweep_configs/synuc.yaml)")
                continue
            table = score_synuc(ckpt, SYNUC_INDEX, drape=drape)
            for metric, scopes in table.items():
                for scope, (mean, _) in scopes.items():
                    if scope != "Net" and metric != "rmse":
                        continue  # per-block: RMSE only (matches the table)
                    sc = f"B{scope}" if isinstance(scope, int) else str(scope)
                    rows.append(
                        dict(
                            run=run,
                            variant="synuc",
                            model="rdn",
                            dataset="ksa_f3_synuc",
                            fold=3,
                            in_ch="",
                            use_dem="",
                            lr_aux="",
                            ckpt_epoch="",
                            split="test",
                            scope=sc,
                            metric=metric,
                            mean=f"{mean:.6f}",
                            std="",
                        )
                    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-sweep", action="store_true", help="reuse existing real-LR model rows")
    ap.add_argument("--skip-synuc", action="store_true", help="reuse existing synuc rows")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    existing = list(csv.DictReader(open(OUT))) if OUT.exists() else []

    if args.skip_sweep:
        model_rows = [r for r in existing if r["model"] != "bicubic" and r["variant"] != "synuc"]
    else:
        # Build in tmp csv then transfer to master copy
        tmp = ROOT_FOLDER / "results/_sweep_tmp.csv"
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                "experiments/sweep_all_models.py",
                "--splits",
                "val",
                "test",
                "--out",
                str(tmp),
            ],
            cwd=ROOT_FOLDER,
            check=True,
        )
        # drop synuc rows: the sweep feeds real LR, the wrong task for operator-consistent ckpts
        model_rows = [
            r for r in csv.DictReader(open(tmp)) if "synuc" not in r["run"] and r["model"] != "bicubic"
        ]
        tmp.unlink()

    print("Scoring bicubic references (KSA folds + WA)...")
    bic_rows = score_bicubic(device)

    if args.skip_synuc:
        synuc_rows = [r for r in existing if r["variant"] == "synuc"]
    else:
        print("Scoring synuc grid (ideal-operator study)...")
        synuc_rows = score_synuc_grid()

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(model_rows)
        w.writerows(bic_rows)
        w.writerows(synuc_rows)
    print(
        f"\nwrote {OUT}  ({len(model_rows)} model + {len(bic_rows)} bicubic + {len(synuc_rows)} synuc rows)"
    )


if __name__ == "__main__":
    main()

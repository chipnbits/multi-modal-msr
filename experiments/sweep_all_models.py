"""Sweep every (non-legacy) checkpoint in checkpoints/ and record val + test metrics.

Each checkpoint is classified from its stored config:
  - model type : RDN (rdnpp) vs U-Net (has 'build_kwargs')
  - dataset    : KSA fold N (index_dir matches cellgrid8[s12][_pNNN]_fold{N}) or WA (patch_dir/x4)
  - channels   : use_dem / dem_mode / lr_aux  (so multi-channel models eval with matching inputs)
KSA p132 models evaluate on the canonical stride-66 cellgrid8_fold{N} grid (same as eval_cv_folds);
p198/p264 patch-size models on their own cellgrid8s12_p{size}_fold3. Legacy (tile_indices_*, geographic
split, cellgrid10) are skipped.

Metrics are PER-PATCH nT RMSE / SSIM / MS-SSIM (mean +/- std), Net (+ per-block B1/B2/B3 for KSA RDN).
DataLoaders are cached by (eval-key, channels) and reused across variants.

Output: results/all_models_eval.csv  (one row per run x variant x split x scope x metric).

Usage:
    uv run python experiments/sweep_all_models.py                 # full sweep
    uv run python experiments/sweep_all_models.py --only RUNSUB   # only runs whose name contains RUNSUB
    uv run python experiments/sweep_all_models.py --splits val test --variants best best_rmse
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import traceback
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "wa_dataset"))

from ksa_aligned_evaluate import evaluate_loader  # RDN KSA (per-block scopes)  # noqa: E402

from magsr.datasets import (  # noqa: E402
    build_ksa_aligned_datasets,
    build_wa_datasets,
    pool_collate,
    worker_init_fn,
)
from magsr.models import SOUP_VARIANT, ensure_soup, load_checkpoint  # noqa: E402

CKPT_DIR = ROOT / "checkpoints"
VARIANTS = ("best", "best_rmse", "last", SOUP_VARIANT)
FOLD_RE = re.compile(r"cellgrid8(s12)?(?:_p(\d+))?_fold(\d)")


# --------------------------------------------------------------------------- classify
def model_in_channels(ck: dict) -> int | None:
    """Ground-truth input channel count = 2nd dim of the first 4D conv weight."""
    for v in ck.get("model", {}).values():
        if hasattr(v, "dim") and v.dim() == 4:
            return int(v.shape[1])
    return None


def channels_from_name(run: str) -> tuple[bool, str, tuple, bool]:
    """RDN run names systematically encode modalities (the stored config is unreliable for older
    runs). Returns (use_dem, dem_mode, lr_aux, is_multispectral). _demgrad=2ch grad, bare _dem=1ch
    relief, _1vd/_ans=lr_aux products, _ferrous=multispectral (separate ms path)."""
    aux = []
    if "_1vd" in run:
        aux.append("1VD")
    if "_ans" in run:
        aux.append("ANS")
    use_dem = "_dem" in run  # matches both _dem and _demgrad
    dem_mode = "grad" if "demgrad" in run else "relief"
    is_ms = "ferrous" in run or "multispec" in run
    return use_dem, dem_mode, tuple(aux), is_ms


def classify(run: str, ck: dict) -> dict | None:
    if "synuc" in run:
        return None  # operator-consistent synthetic pairs -> scored by eval_synuc, not real LR
    cfg = ck.get("config", {})
    is_unet = "build_kwargs" in ck
    idx = str(cfg.get("index_dir", ""))
    m = FOLD_RE.search(idx)
    dataset = fold = eval_index = None
    if m:
        psize, fold = m.group(2), int(m.group(3))
        dataset = f"ksa_f{fold}"
        eval_index = (
            f"patch_indices_cellgrid8s12_p{psize}_fold{fold}"
            if psize
            else f"patch_indices_cellgrid8_fold{fold}"
        )
    elif "x4" in run or str(cfg.get("patch_dir", "None")) != "None":
        dataset = "wa"
    if dataset is None:
        return None  # legacy -> skip

    if is_unet:  # newer trainers store reliable channel config
        use_dem = bool(cfg.get("use_dem", False))
        dem_mode = cfg.get("dem_mode", "grad")
        lr_aux = tuple(cfg.get("lr_aux", ()) or ())
        is_ms = False
    else:  # RDN: derive channels from the run name (config unreliable for older runs)
        use_dem, dem_mode, lr_aux, is_ms = channels_from_name(run)
    if is_ms:
        return None  # multispectral/ferrous uses the ms_bands path -> skip (documented negative)

    info = dict(
        model="unet" if is_unet else "rdn",
        dataset=dataset,
        fold=fold,
        eval_index=eval_index,
        use_dem=use_dem,
        dem_mode=dem_mode,
        lr_aux=lr_aux,
        class_cond=bool(cfg.get("class_cond", False)),
        in_ch=cfg.get("in_channels"),
    )
    # Validate channel inference against the model's actual first-conv in_channels.
    mic = model_in_channels(ck)
    if mic is not None:
        info["in_ch"] = mic
        exp = 1 + len(lr_aux) + ((2 if dem_mode == "grad" else 1) if use_dem else 0)
        if exp != mic:
            info["channel_mismatch"] = f"name={exp}ch vs model={mic}ch"
    return info


# --------------------------------------------------------------------------- loaders (cached)
_loader_cache: dict[tuple, tuple] = {}


def get_loaders(c: dict, batch_size: int, num_workers: int):
    """Return (loaders_dict, ds_dict_or_None). Cached by (dataset/index, channels)."""
    key = (c["dataset"], c["eval_index"], c["use_dem"], c["dem_mode"], c["lr_aux"])
    if key in _loader_cache:
        return _loader_cache[key]
    if c["dataset"] == "wa":
        splits = build_wa_datasets(lr_aux=c["lr_aux"], use_dem=c["use_dem"], dem_mode=c["dem_mode"])
        collate = pool_collate
    else:
        splits = build_ksa_aligned_datasets(
            index_dir=ROOT / "data/processed/ksa_aligned" / c["eval_index"],
            load_dem=c["use_dem"],
            dem_mode=c["dem_mode"],
            lr_aux_products=c["lr_aux"],
        )
        acfg = next(iter(splits.values())).config
        collate = partial(pool_collate, hr_products=acfg.hr_products, lr_products=acfg.lr_products)
    loaders = {
        sp: DataLoader(
            ds,
            collate_fn=collate,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            persistent_workers=num_workers > 0,
        )
        for sp, ds in splits.items()
    }
    _loader_cache[key] = (loaders, splits)
    return loaders, splits


# --------------------------------------------------------------------------- eval paths
# All three feed the ONE canonical evaluate_loader; they differ only in `predict` (how a batch maps
# to [0,1] SR/HR) and `to_nt` (per-block denorm for KSA, global span for WA). The U-Net helpers reuse
# the trainers' `build_hr_input`/`from_pm1` so scoring stays in lock-step with training, but the metric
# loop lives in exactly one place. (The trainers keep their own score_loader for in-training val.)
def eval_wa_rdn(model, loader, vmin, vmax, device, ps, desc="wa_rdn"):
    span = vmax - vmin

    def predict(batch, _i):
        lr, hr = batch["lr"].to(device), batch["hr"].to(device)
        return model(lr).clamp(0, 1), hr, [0] * lr.shape[0]

    tab = evaluate_loader(
        model,
        loader,
        device=device,
        desc=desc,
        predict=predict,
        to_nt=lambda z, _b: z * span,
        patch_size=ps,
    )
    return {m: {"Net": tab[m].get("Net", (float("nan"), float("nan")))} for m in ("rmse", "ssim", "msssim")}


def eval_unet_ksa(model, loader, ds, device, image_size, class_cond, desc):
    from ksa_aligned_unet_train import blocks_to_class, build_hr_input, from_pm1

    def predict(batch, _i):
        x, hr_n, blocks = build_hr_input(ds, batch, device, image_size)
        y = blocks_to_class(blocks, device) if class_cond else None
        sr01 = from_pm1(model(x, y).clamp(-1.0, 1.0))
        return sr01, from_pm1(hr_n), blocks

    tab = evaluate_loader(model, loader, device=device, desc=desc, predict=predict, patch_size=image_size)
    return {m: tab[m] for m in ("rmse", "ssim", "msssim")}


def eval_unet_wa(model, loader, vmin, vmax, device, image_size, desc):
    from wa_unet_train import build_hr_input, from_pm1

    span = vmax - vmin

    def predict(batch, _i):
        x, hr_n = build_hr_input(batch, device, image_size)
        sr01 = from_pm1(model(x, None).clamp(-1.0, 1.0))
        return sr01, from_pm1(hr_n), [0] * x.shape[0]

    tab = evaluate_loader(
        model,
        loader,
        device=device,
        desc=desc,
        predict=predict,
        to_nt=lambda z, _b: z * span + vmin,
        patch_size=image_size,
    )
    return {m: {"Net": tab[m].get("Net", (float("nan"), float("nan")))} for m in ("rmse", "ssim", "msssim")}


def to_rows(run, variant, c, epoch, split, tab):
    rows = []
    for metric, scopes in tab.items():
        for scope, ms in scopes.items():
            sc = f"B{scope}" if isinstance(scope, int) else str(scope)
            rows.append(
                dict(
                    run=run,
                    variant=variant,
                    model=c["model"],
                    dataset=c["dataset"],
                    fold=c["fold"],
                    in_ch=c["in_ch"],
                    use_dem=c["use_dem"],
                    lr_aux="+".join(c["lr_aux"]) or "-",
                    ckpt_epoch=epoch,
                    split=split,
                    scope=sc,
                    metric=metric,
                    mean=f"{ms[0]:.6f}",
                    std=f"{ms[1]:.6f}",
                )
            )
    return rows


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--only", default=None, help="substring filter on run name")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default="results/all_models_eval.csv")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # group checkpoint files by run name
    runs: dict[str, list[str]] = {}
    for f in sorted(CKPT_DIR.glob("*.pt")):
        b = f.stem
        mm = re.match(r"(.+?)_(best_rmse|best|last|soup_[a-z_]+)$", b)
        run, var = (mm.group(1), mm.group(2)) if mm else (b, "none")
        runs.setdefault(run, []).append(var)
    if a.only:
        runs = {r: v for r, v in runs.items() if a.only in r}

    from ksa_aligned_unet_train import MSRRegressionUNet

    out_path = ROOT / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    n_done = n_skip = n_err = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for ri, (run, variants) in enumerate(sorted(runs.items()), 1):
            ref = "best" if "best" in variants else variants[0]
            ref_path = CKPT_DIR / (f"{run}_{ref}.pt" if ref != "none" else f"{run}.pt")
            try:
                ck0 = torch.load(ref_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[{ri}/{len(runs)}] {run}: LOAD-ERR {e}")
                n_err += 1
                continue
            c = classify(run, ck0)
            if c is None:
                print(f"[{ri}/{len(runs)}] {run}: skip (legacy/multispectral)")
                n_skip += 1
                continue
            if c.get("channel_mismatch"):
                print(f"[{ri}/{len(runs)}] {run}: skip (channel {c['channel_mismatch']})")
                n_skip += 1
                continue
            # Soups are derived, not trained: build the missing ones here so every scored run
            # offers the same candidate set (make_tables selects by argmin val RMSE across them).
            if SOUP_VARIANT in a.variants and SOUP_VARIANT not in variants and ensure_soup(CKPT_DIR, run):
                print(f"[{ri}/{len(runs)}] {run}: built {SOUP_VARIANT}")
                variants.append(SOUP_VARIANT)
            try:
                loaders, splits = get_loaders(c, a.batch_size, a.num_workers)
            except Exception as e:
                print(f"[{ri}/{len(runs)}] {run}: LOADER-ERR {e}")
                n_err += 1
                continue
            ds0 = splits["train"] if "train" in splits else next(iter(splits.values()))
            print(
                f"[{ri}/{len(runs)}] {run}  ({c['model']}/{c['dataset']}/in{c['in_ch']}"
                f"/dem{int(c['use_dem'])}/aux{'+'.join(c['lr_aux']) or '-'})"
            )
            for variant in variants:
                if variant not in a.variants:
                    continue
                cp = CKPT_DIR / f"{run}_{variant}.pt"
                try:
                    if c["model"] == "rdn" and c["dataset"] == "wa":
                        # wa_rdn checkpoints lack the KSA 'model_spec'; rebuild x4 directly.
                        from magsr.models import rdnpp_default_x4

                        ck = torch.load(cp, map_location=device, weights_only=False)
                        dem_ch = (2 if c["dem_mode"] == "grad" else 1) if c["use_dem"] else 0
                        in_ch = 1 + dem_ch + len(c["lr_aux"])
                        model = rdnpp_default_x4(in_channels=in_ch).to(device)
                        model.load_state_dict(ck["model"])
                        model.eval()
                        epoch = ck.get("epoch", "")
                    elif c["model"] == "rdn":
                        model, ck = load_checkpoint(cp, device=device, return_ckpt=True)
                        epoch = ck.get("epoch", "")
                    else:
                        ck = torch.load(cp, map_location=device, weights_only=False)
                        model = MSRRegressionUNet(**ck["build_kwargs"]).to(device)
                        model.load_state_dict(ck["model"])
                        model.eval()
                        epoch = ck.get("epoch", "")
                    for split in a.splits:
                        if split not in loaders:
                            continue
                        if c["dataset"] == "wa":
                            vmin, vmax = ds0.vmin, ds0.vmax
                            ps = (
                                int(ds0[0]["hr"][ds0.product].shape[-1]) if hasattr(ds0, "product") else 128
                            )
                            if c["model"] == "rdn":
                                tab = eval_wa_rdn(
                                    model,
                                    loaders[split],
                                    vmin,
                                    vmax,
                                    device,
                                    ps,
                                    desc=f"{run[:20]} {split}",
                                )
                            else:
                                tab = eval_unet_wa(
                                    model,
                                    loaders[split],
                                    vmin,
                                    vmax,
                                    device,
                                    ps,
                                    desc=f"{run[:20]} {split}",
                                )
                        else:
                            if c["model"] == "rdn":
                                tab = evaluate_loader(
                                    model, loaders[split], device=device, desc=f"{run[:20]} {split}"
                                )
                            else:
                                tab = eval_unet_ksa(
                                    model,
                                    loaders[split],
                                    ds0,
                                    device,
                                    int(ds0.config.patch_px),
                                    c["class_cond"],
                                    desc=f"{run[:20]} {split}",
                                )
                        w.writerows(to_rows(run, variant, c, epoch, split, tab))
                        fh.flush()
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    n_done += 1
                except Exception:
                    print(f"    {variant}: EVAL-ERR\n{traceback.format_exc().splitlines()[-1]}")
                    n_err += 1
    print(f"\nDONE: {n_done} evals, {n_skip} legacy runs skipped, {n_err} errors -> {out_path}")


if __name__ == "__main__":
    main()

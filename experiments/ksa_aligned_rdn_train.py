"""Train RDN++ x3 on KSA Shield aeromagnetic patches (aligned snapshot) with Accelerate.

Loads the pre-snapped EPSG:32637 KSA aligned dataset (HR 60 m AMF_RTP / LR 180 m
RTP, both prefer the per-block zero-mean HR when present) and runs x3 super-resolution.
Train/val/test splits come from the JSON files written by
`scripts/build_ksa_dataset/06_assign_cell_splits.py` (point `--index-dir` at the
cell-grid cut, e.g. `patch_indices_cellgrid8_fold3/`).

Launch (single-GPU or multi-GPU via Accelerate):
    uv run accelerate launch experiments/ksa_aligned_rdn_train.py
    uv run accelerate launch --num_processes 4 experiments/ksa_aligned_rdn_train.py
Plain `uv run python experiments/ksa_aligned_rdn_train.py` also works (Accelerate
falls back to a single process).
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from magsr.cli import add_channel_args
from magsr.datasets import build_ksa_aligned_datasets, pool_collate, random_d4, worker_init_fn
from magsr.fourier import FVDLoss
from magsr.fourier.fvd import first_vertical_derivative
from magsr.fourier.synth import synth_uc_pair, synth_uc_pair_drape
from magsr.metrics import (
    MaskedL1Loss,
    make_msssim,
    make_rmse,
    make_ssim,
)
from magsr.models import RDNpp, git_sha


# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------
@dataclass
class TrainConfig:
    run_name: str = "rdnpp_x3_16mp_250ep_adamw2e-3"
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 16
    num_epochs: int = 250
    stop_epoch: int | None = None  # halt training after this epoch; scheduler still spans num_epochs
    lr_start: float = 3e-4
    grad_clip: float = 1.0
    pct_start: float = 0.05  # Percent of cycle spent increasing LR (OneCycleLR)
    anneal_strategy: str = "cos"  # "cos" or "linear" (OneCycleLR)
    mixed_precision: str = "bf16"  # one of: "no", "fp16", "bf16", "fp8"
    val_every: float = 5.0  # validate + checkpoint every N (fractional) epochs; 0.5 = twice/epoch
    ckpt_dir: Path = Path("checkpoints")
    wandb_project: str = "magsr-ksa"
    wandb_entity: str | None = None

    # Dataloader settings
    augment: bool = True  # random D4 (rot90 + flips) train-time augmentation; --no-augment to disable
    use_dem: bool = False  # feed the 30m DEM (pooled to 180m) as extra input channel(s)
    dem_mode: str = "relief"  # "relief" (1ch elevation) or "grad" (2ch slope dz/dx,dz/dy)
    lr_aux: tuple[
        str, ...
    ] = ()  # extra LR product channels (e.g. ("1VD", "ANS")), each normalized independently
    ms_bands: tuple[int, ...] = ()  # Landsat bands 1..7 to feed (pooled 30m->180m, per-band normalized)
    ms_features: tuple[str, ...] = ()  # derived band-ratio channels (e.g. ('ferrous',) = b6/b5)
    index_dir: Path | None = (
        None  # split JSON dir; None = rect cut (05), pass patch_indices_cellgrid for 04b/05b
    )

    # Model architecture
    nb: int = 23  # number of trunk RRDBs (paper default 23); lower = shorter trunk
    nf: int = 64  # trunk feature width (head Conv2d(in, nf)); the untested capacity axis
    gc: int = 32  # dense growth channels inside each RDB
    up_factors: list[int] | None = None  # per-stage upsample factors; None = model default [3,1] for x3.

    # Domain adapters
    num_domains: int = 3  # >1 enables per-block residual adapters + residual-delta tail
    domain_rrdbs: int = 5  # k final RRDB blocks carrying residual adapters
    domain_warmup_epochs: int = 20  # Number of domain epochs to freeze adapters
    weight_decay: float = 5e-4  # AdamW weight decay on shared backbone params
    domain_weight_decay: float = 2e-3  # AdamW weight decay on adapter + tail-delta params

    # Synthetic pure upward continuation training for problem analysis
    synthetic_uc: bool = False
    syn_drape: bool = False  # drape-aware generator (EL fit on -(60+DEM), forward to drape-dz)
    syn_dz_min: float = 200.0
    syn_dz_max: float = 250.0
    syn_noise_nt: float = 1.0
    syn_out_px: int = 132

    # Additional loss term experiments
    fvd_weight: float = 0.0  # weight of the |FVD(sr)-FVD(hr)| 1VD physics penalty (0 = off)
    grad_diag_every: int = 0  # every N steps, log per-term gradient norms/ratio/cosine (0 = off)


# -----------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------
def _make_lr_hr_sr_figure(
    lr: torch.Tensor,
    hr: torch.Tensor,
    sr: torch.Tensor,
    blocks: list[int],
    n_patches: int = 6,
    cmap: str = "viridis",
):
    """Side-by-side LR / HR / SR comparison for the first `n_patches` of a batch."""
    n = min(n_patches, sr.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    titles = ("LR", "HR", "SR")
    for i in range(n):
        patches = (lr[i, 0].numpy(), hr[i, 0].numpy(), sr[i, 0].numpy())
        for j, arr in enumerate(patches):
            axes[i, j].imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j])
        axes[i, 0].set_ylabel(f"B{blocks[i]}", rotation=0, labelpad=20, va="center")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------
# Data
# -----------------------------------------------------------------
def _grad_diag(l1_term, aux_terms, model: nn.Module) -> dict[str, float]:
    """Per-term gradient diagnostics for auxiliary regularizers (run on occasional draws)."""
    params = [p for p in model.parameters() if p.requires_grad]
    g1 = torch.autograd.grad(l1_term, params, retain_graph=True, allow_unused=True)

    def _norm(gs) -> torch.Tensor:
        return torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None) + 1e-20)

    n1 = _norm(g1)
    out: dict[str, float] = {"l1_grad": n1.item(), "l1_val": l1_term.item()}
    for name, term, lam in aux_terms:
        gf = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        nf = _norm(gf)
        dot = sum((a.float() * b.float()).sum() for a, b in zip(g1, gf) if a is not None and b is not None)
        out[f"{name}_grad_raw"] = nf.item()
        out[f"{name}_grad_weighted"] = (lam * nf).item()
        out[f"{name}_ratio"] = (lam * nf / n1).item()
        out[f"{name}_cos"] = float(dot / (n1 * nf))
        out[f"{name}_val"] = term.item()
    return out


def build_loaders(cfg: TrainConfig) -> tuple[dict, dict[str, DataLoader]]:
    splits = build_ksa_aligned_datasets(
        index_dir=cfg.index_dir,
        load_dem=cfg.use_dem or cfg.syn_drape,
        dem_mode=cfg.dem_mode,
        lr_aux_products=cfg.lr_aux,
        ms_bands=cfg.ms_bands,
        ms_features=cfg.ms_features,
    )
    aligned_cfg = next(iter(splits.values())).config
    collate = partial(
        pool_collate,
        hr_products=aligned_cfg.hr_products,
        lr_products=aligned_cfg.lr_products,
    )

    def make_loader(split: str, *, shuffle: bool) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        return DataLoader(
            splits[split],
            collate_fn=collate,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=shuffle,
            generator=g,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=True,
            drop_last=shuffle,  # drop last only for training
        )

    loaders = {
        "train": make_loader("train", shuffle=True),
        "val": make_loader("val", shuffle=False),
        "test": make_loader("test", shuffle=False),
    }
    return splits, loaders


# -----------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------
def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    cfg: TrainConfig,
    accelerator: Accelerator,
) -> dict:
    train_dl, val_dl = loaders["train"], loaders["val"]
    ds = build_ksa_aligned_datasets(
        index_dir=cfg.index_dir,
        load_dem=cfg.use_dem,
        dem_mode=cfg.dem_mode,
        lr_aux_products=cfg.lr_aux,
        ms_bands=cfg.ms_bands,
        ms_features=cfg.ms_features,
    )["train"]

    # Auxiliary physics penalty dx=dy=1 (per-pixel wavenumber).
    #   fvd   |FVD(pred)−FVD(hr)|          1VD via |k| FFT
    aux_fns: list[tuple[str, nn.Module, float]] = []
    if cfg.fvd_weight > 0:
        aux_fns.append(("fvd", FVDLoss(dx=1.0, dy=1.0), cfg.fvd_weight))

    # Give warning if dataset is doing perblock norm
    if ds.config.norm_mode == "per_block":
        warn("Dataset is using per-block normalization instead of global stats. Ensure this is intended.")

    # Capture the source commit ONCE at run start — not per checkpoint-save. The
    # working tree can change during a multi-hour run, so a save-time git_sha()
    # would mis-stamp the architecture actually held in memory.
    source_sha = git_sha()

    optim = AdamW(
        model.param_groups(
            base_weight_decay=cfg.weight_decay,
            domain_weight_decay=cfg.domain_weight_decay,
        ),
        lr=cfg.lr_start,
        fused=True,
    )

    model, optim, train_dl, val_dl = accelerator.prepare(model, optim, train_dl, val_dl)

    sched = OneCycleLR(
        optim,
        max_lr=cfg.lr_start,
        epochs=cfg.num_epochs,
        steps_per_epoch=len(train_dl),
        pct_start=cfg.pct_start,
        anneal_strategy=cfg.anneal_strategy,
    )

    device = accelerator.device
    is_main = accelerator.is_main_process

    wandb_cfg = {
        **asdict(cfg),
        "ckpt_dir": str(cfg.ckpt_dir),
        "index_dir": str(cfg.index_dir) if cfg.index_dir is not None else None,
        "model": type(accelerator.unwrap_model(model)).__name__,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "criterion": type(criterion).__name__,
        "optimizer": type(optim).__name__,
        "scheduler": "OneCycleLR",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "num_processes": accelerator.num_processes,
        "mixed_precision": str(accelerator.mixed_precision),
    }

    run = None
    if is_main:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.run_name,
            config=wandb_cfg,
        )

    patch_size = int(ds.index.spec.patch_px)  # actual HR patch size (index-driven, not config default)
    if cfg.synthetic_uc:
        patch_size = cfg.syn_out_px  # target is the centre crop of the oversized context
    block_ids = list(ds.config.blocks)

    def fresh_metrics() -> dict:
        return {
            "rmse": make_rmse().to(device),
            "ssim": make_ssim().to(device),
            "msssim": make_msssim(patch_size=patch_size).to(device),
        }

    val_metrics = {b: fresh_metrics() for b in block_ids}

    if is_main:
        cfg.ckpt_dir.mkdir(exist_ok=True, parents=True)
    hist = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_msssim": [],
        "epochs": [],
    }
    metric_names = ("rmse", "psnr", "ssim", "msssim")
    best_ssim = -float("inf")
    best_rmse = float("inf")
    global_step = 0
    multi_proc = accelerator.num_processes > 1
    log_every = 50  # how often to sync train loss to host for pbar/wandb

    # Domain group warmup: freeze adapter + tail_delta params for the first N
    # epochs so the shared backbone settles before per-domain perturbations
    # start updating. param_groups[1] is the domain group (when num_domains > 1).
    domain_params = (
        list(optim.param_groups[1]["params"])
        if cfg.num_domains > 1 and cfg.domain_warmup_epochs > 0
        else []
    )

    steps_per_epoch = max(1, len(train_dl))
    # val_every is in (fractional) epochs: 0.5 => validate twice per epoch, 1 => every epoch, etc.
    val_interval = max(1, round(cfg.val_every * steps_per_epoch))

    def synth_batch(batch, blocks, generator=None):
        """Oversized-context batch -> (hr_target, lr_syn, dem_crop, aux_syn) in RAW units.

        Replaces the real LR with the operator-consistent synthetic one; the 1VD aux
        channel (if configured) is recomputed FROM the synthetic LR so every input is
        consistent with the same generative operator; DEM is centre-cropped to match.
        """
        hr_full = batch["hr"][:, :1]
        B = hr_full.shape[0]
        if generator is None:
            dz = cfg.syn_dz_min + (cfg.syn_dz_max - cfg.syn_dz_min) * torch.rand(B, device=hr_full.device)
        else:
            dz = cfg.syn_dz_min + (cfg.syn_dz_max - cfg.syn_dz_min) * torch.rand(
                B, generator=generator, device=hr_full.device
            )
        if cfg.syn_drape:
            # DEM on the HR grid of the FULL context (30 m -> 60 m via 2x mean).
            dem_t = batch["dem"].float()
            if dem_t.ndim == 3:
                dem_t = dem_t.unsqueeze(1)
            dem_hr = torch.nn.functional.avg_pool2d(dem_t, 2, 2).squeeze(1)
            hr_c, lr_syn = synth_uc_pair_drape(
                hr_full, dem_hr, dz, noise_nt=cfg.syn_noise_nt, out_px=cfg.syn_out_px, generator=generator
            )
        else:
            hr_c, lr_syn = synth_uc_pair(
                hr_full, dz, noise_nt=cfg.syn_noise_nt, out_px=cfg.syn_out_px, generator=generator
            )
        dem_c = None
        if cfg.use_dem:
            dem = batch["dem"]
            offd = (dem.shape[-1] - cfg.syn_out_px * 2) // 2  # DEM at 30 m = 2x HR px
            dem_c = dem[..., offd : offd + cfg.syn_out_px * 2, offd : offd + cfg.syn_out_px * 2]
        aux_c = None
        if cfg.lr_aux:
            assert tuple(cfg.lr_aux) == ("1VD",), "synthetic mode supports lr_aux=('1VD',) only"
            aux_c = first_vertical_derivative(lr_syn[:, 0].float(), dx=180.0, dy=180.0).unsqueeze(1)
        return hr_c, lr_syn, dem_c, aux_c

    def validate(epoch_label: float, t_loss: float) -> None:
        """Run validation + checkpointing at a (possibly fractional) epoch point.

        Called on a fixed STEP interval (``val_interval``) so a long dense epoch can produce
        several eval points. Appends one aligned row to ``hist`` per call, logs to wandb at the
        current ``global_step``, and saves ``_last`` / ``_best`` (val SSIM) / ``_best_rmse``.
        """
        nonlocal best_ssim, best_rmse
        model.eval()
        v_loss = 0.0
        for b in block_ids:
            for m in val_metrics[b].values():
                m.reset()
        counts_per = {b: 0 for b in block_ids}
        first_batch_vis: tuple | None = None
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dl):
                blocks = [m["block"] for m in batch["meta"]]
                if cfg.synthetic_uc:
                    # Deterministic batch generator for consistency
                    gen = torch.Generator(device=batch["hr"].device)
                    gen.manual_seed(10_000 + batch_idx)
                    hr_t, lr_t, dem_in, aux_in = synth_batch(batch, blocks, generator=gen)
                else:
                    lr_t, hr_t = batch["lr"], batch["hr"][:, :1]
                    dem_in, aux_in = batch.get("dem"), batch.get("lr_aux")
                blocks_t = torch.tensor(blocks, device=lr_t.device, dtype=torch.long)
                hr_t = ds.normalize(hr_t, blocks=blocks)
                lr_t = ds.assemble_lr_input(
                    lr_t,
                    blocks,
                    dem=dem_in if cfg.use_dem else None,
                    lr_aux=aux_in if cfg.lr_aux else None,
                    ms=batch["ms"] if (cfg.ms_bands or cfg.ms_features) else None,
                )
                sr = (model(lr_t, blocks_t) if cfg.num_domains > 1 else model(lr_t)).clamp(0, 1)
                if batch_idx == 0 and is_main:
                    first_batch_vis = (
                        lr_t.detach().cpu(),
                        hr_t.detach().cpu(),
                        sr.detach().cpu(),
                        list(blocks),
                    )
                loss = criterion(sr, hr_t)
                v_loss += accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
                sr_g = accelerator.gather_for_metrics(sr)
                hr_g = accelerator.gather_for_metrics(hr_t)
                blk_g = accelerator.gather_for_metrics(blocks_t)
                msk_g = hr_g.isfinite().float()
                sr_m = sr_g * msk_g
                hr_m = hr_g.nan_to_num(0.0) * msk_g
                for b in block_ids:
                    sel = blk_g == b
                    n_b = int(sel.sum().item())
                    if n_b == 0:
                        continue
                    sr_b, hr_b = sr_m[sel], hr_m[sel]
                    for m in val_metrics[b].values():
                        m.update(sr_b, hr_b)
                    counts_per[b] += n_b
        v_loss /= len(val_dl)

        per_block: dict[int, dict[str, float]] = {}
        for b in block_ids:
            if counts_per[b] == 0:
                continue
            per_block[b] = {n: m.compute().item() for n, m in val_metrics[b].items()}
        total_n = sum(counts_per[b] for b in per_block) or 1
        v_rmse_val = math.sqrt(sum(counts_per[b] * per_block[b]["rmse"] ** 2 for b in per_block) / total_n)
        v_psnr = 10.0 * math.log10(1.0 / (v_rmse_val + 1e-12))
        v_ssim = sum(counts_per[b] * per_block[b]["ssim"] for b in per_block) / total_n
        v_msssim = sum(counts_per[b] * per_block[b]["msssim"] for b in per_block) / total_n
        totals = {"rmse": v_rmse_val, "psnr": v_psnr, "ssim": v_ssim, "msssim": v_msssim}

        hist["train_loss"].append(t_loss)
        hist["epochs"].append(epoch_label)
        hist["val_loss"].append(v_loss)
        hist["val_rmse"].append(v_rmse_val)
        hist["val_psnr"].append(v_psnr)
        hist["val_ssim"].append(v_ssim)
        hist["val_msssim"].append(v_msssim)

        if is_main:
            log: dict = {"epoch": epoch_label, "train/loss_epoch": t_loss, "val/loss": v_loss}
            for name in metric_names:
                log[f"val/{name}/all"] = totals[name]
            for b in block_ids:
                pm = per_block.get(b)
                if pm is None:
                    continue
                pm_psnr = 10.0 * math.log10(1.0 / (pm["rmse"] + 1e-12))
                log[f"val/rmse/B{b}"] = pm["rmse"]
                log[f"val/psnr/B{b}"] = pm_psnr
                log[f"val/ssim/B{b}"] = pm["ssim"]
                log[f"val/msssim/B{b}"] = pm["msssim"]
            if first_batch_vis is not None:
                fig = _make_lr_hr_sr_figure(*first_batch_vis)
                log["val/samples"] = wandb.Image(fig)
                plt.close(fig)
            wandb.log(log, step=global_step)

        accelerator.wait_for_everyone()
        if is_main:
            unwrapped = accelerator.unwrap_model(model)
            ckpt = {
                "epoch": epoch_label,
                "model_spec": {
                    "name": type(unwrapped).__name__,
                    "kwargs": dict(unwrapped.build_kwargs),
                    "source_sha": source_sha,
                },
                "model": unwrapped.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "hist": hist,
                "config": wandb_cfg,
                "wandb_run_id": run.id if run is not None else None,
                "val_rmse": v_rmse_val,
                "val_psnr": v_psnr,
                "val_ssim": v_ssim,
                "val_msssim": v_msssim,
            }
            torch.save(ckpt, cfg.ckpt_dir / f"{cfg.run_name}_last.pt")
            if v_ssim > best_ssim:
                best_ssim = v_ssim
                torch.save(ckpt, cfg.ckpt_dir / f"{cfg.run_name}_best.pt")
                wandb.run.summary["best/val_ssim"] = v_ssim
                wandb.run.summary["best/val_psnr"] = v_psnr
                wandb.run.summary["best/val_msssim"] = v_msssim
                wandb.run.summary["best/epoch"] = epoch_label
            if v_rmse_val < best_rmse:
                best_rmse = v_rmse_val
                torch.save(ckpt, cfg.ckpt_dir / f"{cfg.run_name}_best_rmse.pt")
                wandb.run.summary["best_rmse/val_rmse"] = v_rmse_val
                wandb.run.summary["best_rmse/val_psnr"] = v_psnr
                wandb.run.summary["best_rmse/val_ssim"] = v_ssim
                wandb.run.summary["best_rmse/epoch"] = epoch_label
        model.train()

    last_epoch = min(cfg.stop_epoch or cfg.num_epochs, cfg.num_epochs)
    for epoch in range(1, last_epoch + 1):
        if domain_params:
            unfreeze = epoch > cfg.domain_warmup_epochs
            for p in domain_params:
                p.requires_grad_(unfreeze)
            if is_main and (epoch == 1 or epoch == cfg.domain_warmup_epochs + 1):
                print(f"epoch {epoch}: domain params requires_grad={unfreeze}")

        # --- train ---
        model.train()
        loss_sum = torch.zeros((), device=device)
        n_steps = 0
        pbar = tqdm(
            train_dl,
            desc=f"[{cfg.run_name}] {epoch:03d}/{last_epoch}",
            leave=False,
            ncols=100,
            disable=not is_main,
        )
        for batch in pbar:
            blocks = [m["block"] for m in batch["meta"]]
            if cfg.synthetic_uc:
                hr_t, lr_t, dem_in, aux_in = synth_batch(batch, blocks)
            else:
                lr_t, hr_t = batch["lr"], batch["hr"][:, :1]
                dem_in, aux_in = batch.get("dem"), batch.get("lr_aux")
            hr_t = ds.normalize(hr_t, blocks=blocks)
            lr_t = ds.assemble_lr_input(
                lr_t,
                blocks,
                dem=dem_in if cfg.use_dem else None,
                lr_aux=aux_in if cfg.lr_aux else None,
                ms=batch["ms"] if (cfg.ms_bands or cfg.ms_features) else None,
            )
            if cfg.augment:
                lr_t, hr_t = random_d4(lr_t, hr_t)

            optim.zero_grad(set_to_none=True)
            if cfg.num_domains > 1:
                blocks_t = torch.tensor(blocks, device=lr_t.device, dtype=torch.long)
                pred = model(lr_t, blocks_t)
            else:
                pred = model(lr_t)
            l1_term = criterion(pred, hr_t)
            loss = l1_term
            if aux_fns:
                aux_terms = []
                for name, fn, w in aux_fns:
                    term = fn(pred, hr_t)
                    loss = loss + w * term
                    aux_terms.append((name, term, w))
                # Regularizer diagnostics on occasional draws: compare each aux term's
                # gradient magnitude w.r.t. the actual weights (not just loss values) and its
                # alignment with L1, so we can see whether each penalty stays in-ballpark with
                # L1 as training converges and whether it conflicts with the main objective.
                if is_main and cfg.grad_diag_every and global_step % cfg.grad_diag_every == 0:
                    gd = _grad_diag(l1_term, aux_terms, model)
                    wandb.log({f"grad/{k}": v for k, v in gd.items()} | {"epoch": epoch}, step=global_step)
                    parts = "  ".join(
                        f"{name}:ratio={gd[f'{name}_ratio']:.2f} cos={gd[f'{name}_cos']:+.2f} "
                        f"val={gd[f'{name}_val']:.4f}"
                        for name, _, _ in aux_terms
                    )
                    print(
                        f"  [grad-diag step {global_step}] |∇L1|={gd['l1_grad']:.2e} "
                        f"L1={gd['l1_val']:.4f} | {parts}",
                        flush=True,
                    )

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()

            # Accumulate on-device; avoid per-step host sync.
            loss_sum += loss.detach()
            n_steps += 1
            if is_main and (global_step % log_every == 0):
                loss_t = loss.detach()
                if multi_proc:
                    loss_t = accelerator.gather(loss_t.unsqueeze(0)).mean()
                loss_avg = loss_t.item()
                current_lr = sched.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{loss_avg:.4f}", "lr": f"{current_lr:.2e}"})
                wandb.log(
                    {"train/loss_step": loss_avg, "train/lr": current_lr, "epoch": epoch},
                    step=global_step,
                )
            global_step += 1
            if global_step % val_interval == 0:
                ls = accelerator.reduce(loss_sum.clone(), reduction="mean") if multi_proc else loss_sum
                validate(epoch - 1 + n_steps / steps_per_epoch, (ls / max(n_steps, 1)).item())
    # Final validation if the last step did not land on a val_interval boundary.
    if global_step % val_interval != 0:
        validate(float(last_epoch), (loss_sum / max(n_steps, 1)).item())

    if is_main and run is not None:
        run.finish()
    return hist


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(
        description="Train RDN++ x3 on KSA Shield aeromagnetic patches.",
    )
    p.add_argument("--run-name", default=defaults.run_name)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--num-workers", type=int, default=defaults.num_workers)
    p.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    p.add_argument(
        "--stop-epoch",
        type=int,
        default=defaults.stop_epoch,
        help="Halt training after this epoch while the LR schedule still spans --num-epochs "
        "(e.g. --num-epochs 250 --stop-epoch 150 trains 150 epochs of a 250-epoch OneCycle).",
    )
    p.add_argument("--lr-start", type=float, default=defaults.lr_start)
    p.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    p.add_argument(
        "--pct-start",
        type=float,
        default=defaults.pct_start,
        help="OneCycleLR warmup fraction (portion of the schedule spent ramping LR up).",
    )
    p.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16", "fp8"),
        default=defaults.mixed_precision,
    )
    p.add_argument(
        "--val-every",
        type=float,
        default=defaults.val_every,
        help="Validate + checkpoint every N (fractional) epochs; 0.5 = twice per epoch",
    )
    p.add_argument("--ckpt-dir", type=Path, default=defaults.ckpt_dir)
    p.add_argument("--wandb-project", default=defaults.wandb_project)
    p.add_argument("--wandb-entity", default=defaults.wandb_entity)
    p.add_argument(
        "--nb",
        type=int,
        default=defaults.nb,
        help="Number of trunk RRDBs (paper default 23). Lower shrinks the trunk for capacity/overfitting studies.",
    )
    p.add_argument(
        "--nf",
        type=int,
        default=defaults.nf,
        help="Trunk feature width (head Conv2d(in, nf), all RRDBs run at nf). Widen to give the "
        "multi-channel input more fusion capacity. Paper default 64.",
    )
    p.add_argument(
        "--gc",
        type=int,
        default=defaults.gc,
        help="Dense growth channels inside each RDB. Paper default 32.",
    )
    p.add_argument(
        "--up-factors",
        type=int,
        nargs="+",
        default=defaults.up_factors,
        help="Per-stage upsample factors. Pass '--up-factors 3' for the single-stage wxbe3eck baseline; "
        "default (omit) uses the model's [3,1] for x3.",
    )
    p.add_argument(
        "--num-domains",
        type=int,
        default=defaults.num_domains,
        help="If >1, enable per-block residual adapters and residual-delta tail (Rebuffi 2017).",
    )
    p.add_argument(
        "--domain-rrdbs",
        type=int,
        default=defaults.domain_rrdbs,
        help="Number of trailing RRDBs that carry a per-domain adapter (only used when --num-domains > 1).",
    )
    p.add_argument(
        "--domain-warmup-epochs",
        type=int,
        default=defaults.domain_warmup_epochs,
        help="Freeze adapter+tail_delta params for the first N epochs so the shared backbone settles first.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.weight_decay,
        help="AdamW weight decay on shared backbone params. 0 reproduces plain Adam.",
    )
    p.add_argument(
        "--domain-weight-decay",
        type=float,
        default=defaults.domain_weight_decay,
        help="AdamW weight decay on adapter + tail-delta params (only used when --num-domains > 1).",
    )
    add_channel_args(p, defaults=defaults)
    p.add_argument(
        "--use-1vd",
        action="store_true",
        help="Back-compat alias for --lr-aux 1VD.",
    )
    p.add_argument(
        "--ms-bands",
        type=int,
        nargs="+",
        default=list(defaults.ms_bands),
        metavar="N",
        help="Landsat bands (1..7) to feed as extra channels, pooled 30m->180m and per-band "
        "normalized. e.g. --ms-bands 1 2 3 4 5 6 7 (all) or --ms-bands 6 (SWIR1).",
    )
    p.add_argument(
        "--syn-drape",
        action="store_true",
        help="Drape-aware synthetic generator (EL fit on the survey drape; needs DEM loaded).",
    )
    p.add_argument(
        "--synthetic-uc",
        action="store_true",
        help="Operator-consistent synthetic pairs from an oversized-patch index "
        "(LR = decimate3(UC(HR ctx, dz~U[syn-dz-min,max])) + noise).",
    )
    p.add_argument("--syn-dz-min", type=float, default=defaults.syn_dz_min)
    p.add_argument("--syn-dz-max", type=float, default=defaults.syn_dz_max)
    p.add_argument("--syn-noise-nt", type=float, default=defaults.syn_noise_nt)
    p.add_argument(
        "--fvd-weight",
        type=float,
        default=defaults.fvd_weight,
        help="Weight λ of the physics penalty λ·|FVD(pred)−FVD(hr)| (1VD via |k| FFT, "
        "per-pixel wavenumber). 0 disables. FVD is high-pass: its loss VALUE is tiny "
        "(~0.14×L1) but its GRADIENT is large (~0.6-0.9×L1 grad at λ=1), so balance on "
        "GRADIENT not value — at λ=1 the param-space FVD grad is ~1 pct of L1, so meaningful "
        "λ is ~5 (modest) to ~50 (strong); calibrate with --grad-diag-every.",
    )
    p.add_argument(
        "--grad-diag-every",
        type=int,
        default=defaults.grad_diag_every,
        help="Every N steps, log per-term gradient norms (|∇L1| vs λ|∇FVD|), their ratio "
        "and cosine alignment — to monitor a new regularizer. 0 disables. Costs 2 extra "
        "backward passes on those steps; ~200 is a good 'occasional draw'.",
    )
    p.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        default=defaults.augment,
        help="Disable random D4 train-time augmentation (rot90 + flips).",
    )
    p.add_argument(
        "--ms-features",
        nargs="+",
        default=list(defaults.ms_features),
        metavar="NAME",
        help="Derived multispectral band-ratio channels: ferrous (b6/b5), clay (b6/b7), "
        "ironoxide (b4/b2). e.g. --ms-features ferrous.",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=defaults.index_dir,
        help="Split JSON dir. Default: the rect cut (05); pass "
        "data/processed/ksa_aligned/patch_indices_cellgrid for the cell-grid cut (04b/05b).",
    )
    a = p.parse_args()
    lr_aux = list(a.lr_aux)
    if a.use_1vd and "1VD" not in lr_aux:
        lr_aux = ["1VD", *lr_aux]
    return TrainConfig(
        run_name=a.run_name,
        seed=a.seed,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        num_epochs=a.num_epochs,
        stop_epoch=a.stop_epoch,
        lr_start=a.lr_start,
        grad_clip=a.grad_clip,
        pct_start=a.pct_start,
        mixed_precision=a.mixed_precision,
        val_every=a.val_every,
        ckpt_dir=a.ckpt_dir,
        wandb_project=a.wandb_project,
        wandb_entity=a.wandb_entity,
        nb=a.nb,
        nf=a.nf,
        gc=a.gc,
        up_factors=a.up_factors,
        num_domains=a.num_domains,
        domain_rrdbs=a.domain_rrdbs,
        domain_warmup_epochs=a.domain_warmup_epochs,
        weight_decay=a.weight_decay,
        domain_weight_decay=a.domain_weight_decay,
        use_dem=a.use_dem,
        dem_mode=a.dem_mode,
        lr_aux=tuple(lr_aux),
        ms_bands=tuple(a.ms_bands),
        ms_features=tuple(a.ms_features),
        fvd_weight=a.fvd_weight,
        synthetic_uc=a.synthetic_uc,
        syn_drape=a.syn_drape,
        syn_dz_min=a.syn_dz_min,
        syn_dz_max=a.syn_dz_max,
        syn_noise_nt=a.syn_noise_nt,
        grad_diag_every=a.grad_diag_every,
        augment=a.augment,
        index_dir=a.index_dir,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    splits, loaders = build_loaders(cfg)
    if accelerator.is_main_process:
        print(
            f"splits: train={len(splits['train'])}  "
            f"val={len(splits['val'])}  test={len(splits['test'])}"
        )

    dem_ch = (2 if cfg.dem_mode == "grad" else 1) if cfg.use_dem else 0
    model = RDNpp(
        in_channels=1 + dem_ch + len(cfg.lr_aux) + len(cfg.ms_bands) + len(cfg.ms_features),
        out_channels=1,
        nb=cfg.nb,
        nf=cfg.nf,
        gc=cfg.gc,
        upscale=3,
        num_domains=cfg.num_domains,
        domain_rrdbs=cfg.domain_rrdbs,
        up_factors=cfg.up_factors,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        print(
            f"RDN++ x3 has {n_params:,} trainable parameters on {accelerator.device} "
            f"({accelerator.num_processes} process(es))"
        )

    criterion = MaskedL1Loss()
    train_model(model, loaders, criterion, cfg, accelerator)


if __name__ == "__main__":
    main()

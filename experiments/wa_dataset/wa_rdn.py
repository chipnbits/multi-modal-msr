"""Train RDN++ on Western Australia aeromagnetic patches (Smith et al., 2022).

Reproducibility:
    For fully deterministic DataLoader shuffling, set PYTHONHASHSEED=0 before
    launching, e.g. `PYTHONHASHSEED=0 uv run python experiments/wa_dataset/wa_rdn.py`.
    cuDNN is pinned to deterministic (benchmark disabled) inside set_seed().
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from magsr.cli import add_channel_args
from magsr.datasets import build_wa_datasets, pool_collate, random_d4, worker_init_fn
from magsr.fourier import FVDLoss
from magsr.metrics import (
    MaskedL1Loss,
    apply_hr_nan_mask,
    make_msssim,
    make_rmse,
    make_ssim,
    rmse_to_psnr,
)
from magsr.models import rdnpp_default_x4


# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------
@dataclass
class TrainConfig:
    run_name: str = "rdnpp_x4"
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 4
    # Smith et al. (2022) hyperparameters
    num_epochs: int = 602
    lr_start: float = 3e-4
    grad_clip: float = 1.0
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    ckpt_dir: Path = Path("checkpoints")
    wandb_project: str = "magsr-wa"
    wandb_entity: str | None = None
    # --- extra input modalities + physics loss (mirrors the KSA combo) ---
    lr_aux: tuple[str, ...] = ()  # e.g. ("1VD",) — adds the 1VD input channel
    use_dem: bool = False  # add the 2-channel DEM slope gradient input
    dem_mode: str = "grad"
    fvd_weight: float = 0.0  # λ for the |FVD(pred)-FVD(hr)| 1VD physics penalty (0 = off)
    patch_dir: Path | None = None  # override the dataset patch dir (default: configs/datasets.yaml)


# -----------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------
# Data
# -----------------------------------------------------------------
def build_loaders(cfg: TrainConfig) -> tuple[dict, dict[str, DataLoader]]:
    ds_overrides: dict = {"lr_aux": cfg.lr_aux, "use_dem": cfg.use_dem, "dem_mode": cfg.dem_mode}
    if cfg.patch_dir is not None:
        ds_overrides["patch_dir"] = cfg.patch_dir
    splits = build_wa_datasets(**ds_overrides)

    def make_loader(split: str, *, shuffle: bool) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        return DataLoader(
            splits[split],
            collate_fn=pool_collate,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=shuffle,
            generator=g,
            persistent_workers=cfg.num_workers > 0,
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
    device: torch.device,
) -> dict:
    train_dl, val_dl = loaders["train"], loaders["val"]

    # Physics penalty on the magnetic output: |FVD(pred) − FVD(hr)| (1VD via |k| FFT, dx=dy=1,
    # self-consistent — exactly 0 at pred==hr). Operates on channel 0 (mag) regardless of input
    # channel count. Balance λ on the param-space gradient (KSA: ~λ5 is a modest push).
    fvd_fn = FVDLoss(dx=1.0, dy=1.0).to(device) if cfg.fvd_weight > 0 else None

    optim = Adam(model.parameters(), lr=cfg.lr_start)
    sched = OneCycleLR(
        optim,
        max_lr=cfg.lr_start,
        epochs=cfg.num_epochs,
        steps_per_epoch=len(train_dl),
        pct_start=cfg.pct_start,
        anneal_strategy=cfg.anneal_strategy,
    )

    wandb_cfg = {
        **asdict(cfg),
        "ckpt_dir": str(cfg.ckpt_dir),
        "model": type(model).__name__,
        "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "criterion": type(criterion).__name__,
        "optimizer": "Adam",
        "scheduler": "OneCycleLR",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config=wandb_cfg,
    )
    wandb.watch(model, log="gradients", log_freq=200)

    patch_size = int(next(iter(train_dl))["hr"].shape[-1])
    val_rmse = make_rmse().to(device)
    val_ssim = make_ssim().to(device)
    val_msssim = make_msssim(patch_size=patch_size).to(device)

    cfg.ckpt_dir.mkdir(exist_ok=True, parents=True)
    hist = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_msssim": [],
    }
    best_ssim = -float("inf")
    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        # --- train ---
        model.train()
        t_loss = 0.0
        pbar = tqdm(
            train_dl,
            desc=f"[{cfg.run_name}] {epoch:03d}/{cfg.num_epochs}",
            leave=False,
            ncols=100,
        )
        for batch in pbar:
            lr_t = batch["lr"].to(device, non_blocking=True)
            hr_t = batch["hr"].to(device, non_blocking=True)
            lr_t, hr_t = random_d4(lr_t, hr_t)

            optim.zero_grad(set_to_none=True)
            pred = model(lr_t)
            loss = criterion(pred, hr_t)
            if fvd_fn is not None:
                loss = loss + cfg.fvd_weight * fvd_fn(pred, hr_t)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()

            t_loss += loss.item()
            current_lr = sched.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})
            wandb.log(
                {"train/loss_step": loss.item(), "train/lr": current_lr, "epoch": epoch},
                step=global_step,
            )
            global_step += 1
        t_loss /= len(train_dl)

        # --- validate ---
        model.eval()
        v_loss = 0.0
        val_rmse.reset()
        val_ssim.reset()
        val_msssim.reset()
        with torch.no_grad():
            for batch in val_dl:
                lr_t = batch["lr"].to(device, non_blocking=True)
                hr_t = batch["hr"].to(device, non_blocking=True)

                sr = model(lr_t).clamp(0, 1)
                v_loss += criterion(sr, hr_t).item()

                sr_m, hr_m = apply_hr_nan_mask(sr, hr_t)
                val_rmse.update(sr_m, hr_m)
                val_ssim.update(sr_m, hr_m)
                val_msssim.update(sr_m, hr_m)
        v_loss /= len(val_dl)
        v_rmse_val = val_rmse.compute().item()
        v_psnr = rmse_to_psnr(v_rmse_val)
        v_ssim = val_ssim.compute().item()
        v_msssim = val_msssim.compute().item()

        hist["train_loss"].append(t_loss)
        hist["val_loss"].append(v_loss)
        hist["val_rmse"].append(v_rmse_val)
        hist["val_psnr"].append(v_psnr)
        hist["val_ssim"].append(v_ssim)
        hist["val_msssim"].append(v_msssim)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss_epoch": t_loss,
                "val/loss": v_loss,
                "val/rmse": v_rmse_val,
                "val/psnr": v_psnr,
                "val/ssim": v_ssim,
                "val/msssim": v_msssim,
            },
            step=global_step,
        )

        # --- checkpoints ---
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "hist": hist,
            "config": wandb_cfg,
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
            wandb.run.summary["best/epoch"] = epoch

    run.finish()
    return hist


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(
        description="Train RDN++ on WA aeromagnetic patches (Smith et al., 2022).",
    )
    p.add_argument("--run-name", default=defaults.run_name)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--num-workers", type=int, default=defaults.num_workers)
    p.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    p.add_argument("--lr-start", type=float, default=defaults.lr_start)
    p.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    p.add_argument("--ckpt-dir", type=Path, default=defaults.ckpt_dir)
    p.add_argument("--wandb-project", default=defaults.wandb_project)
    p.add_argument("--wandb-entity", default=defaults.wandb_entity)
    add_channel_args(p, defaults=defaults, dem_modes=("grad",))
    p.add_argument(
        "--patch-dir",
        type=Path,
        default=defaults.patch_dir,
        help="Override the dataset patch dir (default from configs/datasets.yaml).",
    )
    p.add_argument(
        "--fvd-weight",
        type=float,
        default=defaults.fvd_weight,
        help="λ for the 1VD physics penalty λ·|FVD(pred)−FVD(hr)| on the mag output (0=off). "
        "Balance on the param-space gradient (KSA: ~5 is a modest push).",
    )
    a = p.parse_args()
    return TrainConfig(
        run_name=a.run_name,
        seed=a.seed,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        num_epochs=a.num_epochs,
        lr_start=a.lr_start,
        grad_clip=a.grad_clip,
        ckpt_dir=a.ckpt_dir,
        wandb_project=a.wandb_project,
        wandb_entity=a.wandb_entity,
        lr_aux=tuple(a.lr_aux),
        use_dem=a.use_dem,
        dem_mode=a.dem_mode,
        fvd_weight=a.fvd_weight,
        patch_dir=a.patch_dir,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, loaders = build_loaders(cfg)
    print(f"splits: train={len(splits['train'])}  " f"val={len(splits['val'])}  test={len(splits['test'])}")

    # Derive in_channels from the actual collated LR tensor (MAG [+1VD] [+DEMGX,DEMGY]).
    in_channels = int(next(iter(loaders["train"]))["lr"].shape[1])
    model = rdnpp_default_x4(in_channels=in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"RDN++ x4 in_channels={in_channels} (lr_aux={cfg.lr_aux} use_dem={cfg.use_dem} "
        f"fvd_weight={cfg.fvd_weight}) — {n_params:,} trainable params on {device}"
    )

    criterion = MaskedL1Loss().to(device)
    train_model(model, loaders, criterion, cfg, device)


if __name__ == "__main__":
    main()

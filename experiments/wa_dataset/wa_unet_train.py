"""Train the MSR residual-corrector U-Net on Western Australia aeromagnetic patches (x4).

WA counterpart of experiments/ksa_aligned_unet_train.py. Reuses the exact same model
(MSRRegressionUNet — HR-resolution residual corrector on a bicubic base) and helpers,
but swaps the data path to the WA goldfields dataset:
  - x4 (LR 32 -> HR 128), not KSA's x3
  - no SGS survey blocks (single survey -> no class conditioning)
  - WA patches arrive already normalized to [0,1] (mag clipped to q1/q99 vmin/vmax,
    NaN nodata preserved); nT denorm is the simple affine x*(vmax-vmin)+vmin.

Baseline = in=1 (mag only), L1, cosine LR. Mirrors the KSA small-U-Net probe setup
(base32 / nrb2 / (1,1,2) = 1.05M, bs128, cosine 1e-3->1e-5) but at 603 epochs to match
the WA RDN Smith-default schedule.

Usage:
    uv run python experiments/wa_dataset/wa_unet_train.py \
        --run-name unet_x4_wa_baseline_112 \
        --base-channels 32 --num-res-blocks 2 --channel-mult 1 1 2 \
        --num-epochs 603 --batch-size 128 --sched cosine --lr-start 1e-3 --lr-end 1e-5
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from magsr.cli import add_channel_args
from magsr.datasets import build_wa_datasets, pool_collate, random_d4, worker_init_fn
from magsr.fourier import FVDLoss
from magsr.metrics import (
    MaskedL1Loss,
    apply_hr_nan_mask,
    mean_std,
    per_patch_msssim,
    per_patch_rmse,
    per_patch_ssim,
)
from magsr.normalize import from_pm1, to_pm1

# Reuse the identical model + helpers from the KSA U-Net trainer (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ksa_aligned_unet_train import (  # noqa: E402
    MSRRegressionUNet,
    adamw_param_groups,
)


@dataclass
class TrainConfig:
    run_name: str = "unet_x4_wa_baseline_112"
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 8
    num_epochs: int = 603
    lr_start: float = 1e-3
    lr_end: float = 1e-5  # cosine floor (eta_min); only used with --sched cosine
    sched: str = "cosine"  # "cosine" (warmup->lr_start, decay->lr_end) or "onecycle"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    pct_start: float = 0.05
    val_every: int = 1
    ckpt_dir: Path = Path("checkpoints")
    wandb_project: str = "magsr-wa"
    wandb_entity: str | None = None
    no_wandb: bool = False
    augment: bool = True
    # model
    base_channels: int = 32
    num_res_blocks: int = 2
    channel_mult: tuple[int, ...] = (1, 1, 2)  # 1.05M @ base32 (depth=len; base must stay div by 32)
    # extra modalities + physics loss (baseline leaves these off)
    lr_aux: tuple[str, ...] = ()
    use_dem: bool = False
    dem_mode: str = "grad"
    fvd_weight: float = 0.0
    patch_dir: Path | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_loaders(cfg: TrainConfig):
    overrides: dict = {"lr_aux": cfg.lr_aux, "use_dem": cfg.use_dem, "dem_mode": cfg.dem_mode}
    if cfg.patch_dir is not None:
        overrides["patch_dir"] = cfg.patch_dir
    splits = build_wa_datasets(**overrides)

    def make(split: str, shuffle: bool) -> DataLoader:
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
            pin_memory=True,
            drop_last=shuffle,
        )

    return splits, {"train": make("train", True), "val": make("val", False), "test": make("test", False)}


def build_hr_input(batch: dict, device: torch.device, image_size: int):
    """HR-resolution [-1,1] input stack + HR target (NaN preserved for masking).

    Every LR channel (mag [+1VD] [+DEM-grad]) is bicubic-upsampled to the HR grid — they are
    all LR-sourced, so the U-Net adds the HR detail as a learned residual on the bicubic base.
    WA patches are already normalized to [0,1]; only NaN nodata needs neutralizing on the input.
    """
    lr_t = batch["lr"].to(device, non_blocking=True)
    hr_t = batch["hr"][:, :1].to(device, non_blocking=True)
    sz = (image_size, image_size)
    up = F.interpolate(lr_t.nan_to_num(0.5), size=sz, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    x = to_pm1(up)
    hr_n = to_pm1(hr_t)  # NaN survives -> masked in loss/metrics
    return x, hr_n


@torch.no_grad()
def score_loader(model, loader, vmin, vmax, device, image_size, desc):
    """Per-patch RMSE(nT)/SSIM/MS-SSIM + per-patch normalized RMSE over a split.

    Three RMSE flavors:
      - rmse        : per-patch nT, mean over patches (HEADLINE + selection; SR-literature standard,
                      matches the RDN evaluate_loader / report table).
      - rmse_norm   : per-patch normalized = rmse / (vmax-vmin) (exact scaling of the headline).
      - rmse_pooled : globally pooled normalized RMSE over ALL valid pixels = EXACTLY the old WA RDN
                      `val/rmse` (torchmetrics MeanSquaredError(squared=False)). Logged as val/rmse so
                      the U-Net curves overlay the RDN runs in the same wandb project.
    """
    model.eval()
    span = vmax - vmin  # affine [0,1] -> nT denorm
    vals = {m: [] for m in ("rmse", "rmse_norm", "ssim", "msssim")}
    sse = cnt = 0.0
    for batch in tqdm(loader, desc=desc, ncols=100):
        x, hr_n = build_hr_input(batch, device, image_size)
        sr01 = from_pm1(model(x, None).clamp(-1.0, 1.0))
        hr01 = from_pm1(hr_n)  # keeps NaN
        mask = hr01.isfinite().to(sr01.dtype)
        sr_m, hr_m = apply_hr_nan_mask(sr01, hr01)
        sr_nt = (sr_m * span + vmin) * mask
        hr_nt = (hr_m * span + vmin) * mask
        vals["rmse"].append(per_patch_rmse(sr_nt, hr_nt, mask).cpu())
        vals["rmse_norm"].append(per_patch_rmse(sr_m, hr_m, mask).cpu())
        sse += ((sr_m - hr_m) ** 2).sum().item()  # pooled (RDN-style) accumulation
        cnt += mask.sum().item()
        fully = mask.flatten(1).all(dim=1)
        if fully.any():
            sv, hv = sr_m[fully], hr_m[fully]
            vals["ssim"].append(per_patch_ssim(sv, hv, data_range=1.0).cpu())
            vals["msssim"].append(per_patch_msssim(sv, hv, patch_size=image_size, data_range=1.0).cpu())
    model.train()
    out = {
        m: mean_std(torch.cat(vals[m])) if vals[m] else (float("nan"), float("nan"))
        for m in ("rmse", "rmse_norm", "ssim", "msssim")
    }
    out["rmse_pooled"] = (math.sqrt(sse / max(cnt, 1.0)), float("nan"))  # == old RDN val/rmse
    return out


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, loaders = build_loaders(cfg)
    ds = splits["train"]
    vmin, vmax = ds.vmin, ds.vmax
    image_size = int(next(iter(loaders["train"]))["hr"].shape[-1])
    in_channels = int(next(iter(loaders["train"]))["lr"].shape[1])
    print(f"splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(
        f"in_channels={in_channels} (aux={cfg.lr_aux or '-'} dem={cfg.use_dem})  HR patch={image_size}  "
        f"x4  mag vmin/vmax={vmin:.1f}/{vmax:.1f} nT"
    )

    model = MSRRegressionUNet(
        in_channels=in_channels,
        image_size=image_size,
        base_channels=cfg.base_channels,
        num_res_blocks=cfg.num_res_blocks,
        channel_mult=cfg.channel_mult,
        class_cond=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MSRRegressionUNet (x4 WA): {n_params:,} trainable params")

    criterion = MaskedL1Loss()
    fvd = FVDLoss(dx=1.0, dy=1.0).to(device) if cfg.fvd_weight > 0 else None
    optim = AdamW(adamw_param_groups(model, cfg.weight_decay), lr=cfg.lr_start, fused=True)
    steps_per_epoch = len(loaders["train"])
    total_steps = cfg.num_epochs * steps_per_epoch
    if cfg.sched == "cosine":
        warmup_steps = max(1, round(cfg.pct_start * total_steps))
        warmup = LinearLR(optim, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optim, T_max=total_steps - warmup_steps, eta_min=cfg.lr_end)
        sched = SequentialLR(optim, [warmup, cosine], milestones=[warmup_steps])
    else:
        sched = OneCycleLR(
            optim,
            max_lr=cfg.lr_start,
            epochs=cfg.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.pct_start,
            anneal_strategy="cos",
        )

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if not cfg.no_wandb:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.run_name,
            config={
                **asdict(cfg),
                "in_channels": in_channels,
                "num_params": n_params,
                "image_size": image_size,
                "model": "MSRRegressionUNet",
                "scale": 4,
            },
        )

    def save_ckpt(tag: str, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "build_kwargs": model.build_kwargs,
                "config": {
                    **asdict(cfg),
                    "patch_dir": str(cfg.patch_dir),
                    "in_channels": in_channels,
                    "vmin": vmin,
                    "vmax": vmax,
                },
                **metrics,
            },
            cfg.ckpt_dir / f"{cfg.run_name}_{tag}.pt",
        )

    def validate(epoch: int) -> dict:
        tab = score_loader(model, loaders["val"], vmin, vmax, device, image_size, desc=f"val e{epoch}")
        return {
            "val_rmse_nt": tab["rmse"][0],  # per-patch nT (headline + selection)
            "val_rmse_norm": tab["rmse_norm"][0],  # per-patch normalized (= nt/span)
            "val_rmse_pooled": tab["rmse_pooled"][0],  # pooled normalized == old RDN val/rmse
            "val_ssim": tab["ssim"][0],
            "val_msssim": tab["msssim"][0],
        }

    best_ssim, best_rmse = -math.inf, math.inf
    global_step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"[{cfg.run_name}] {epoch:03d}/{cfg.num_epochs}", ncols=100)
        for batch in pbar:
            x, hr_n = build_hr_input(batch, device, image_size)
            if cfg.augment:
                x, hr_n = random_d4(x, hr_n)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                pred = model(x, None)
                loss = criterion(pred, hr_n)
                if fvd is not None:
                    loss = loss + cfg.fvd_weight * fvd(pred, hr_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            if global_step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                if run is not None:
                    wandb.log(
                        {"train/loss": loss.item(), "train/lr": sched.get_last_lr()[0], "epoch": epoch},
                        step=global_step,
                    )
            global_step += 1

        if epoch % cfg.val_every == 0 or epoch == cfg.num_epochs:
            m = validate(epoch)
            print(
                f"  e{epoch}: val RMSE {m['val_rmse_nt']:.2f} nT (per-patch)  "
                f"val/rmse(pooled,=RDN) {m['val_rmse_pooled']:.4f}  "
                f"SSIM {m['val_ssim']:.4f}  MS-SSIM {m['val_msssim']:.4f}"
            )
            if run is not None:
                # RDN-aligned keys so curves overlay the wa_rdn runs in the same wandb project.
                wandb.log(
                    {
                        "val/rmse": m["val_rmse_pooled"],  # == RDN val/rmse (pooled normalized)
                        "val/rmse_nt": m["val_rmse_nt"],  # per-patch nT (headline + selection)
                        "val/rmse_norm": m["val_rmse_norm"],
                        "val/ssim": m["val_ssim"],
                        "val/msssim": m["val_msssim"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            save_ckpt("last", epoch, m)
            if m["val_ssim"] > best_ssim:
                best_ssim = m["val_ssim"]
                save_ckpt("best", epoch, m)
            if m["val_rmse_nt"] < best_rmse:
                best_rmse = m["val_rmse_nt"]
                save_ckpt("best_rmse", epoch, m)

    print("\n=== TEST (WA held-out) ===")
    for tag in ("best", "best_rmse"):
        ck_path = cfg.ckpt_dir / f"{cfg.run_name}_{tag}.pt"
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        em = MSRRegressionUNet(**ck["build_kwargs"]).to(device)
        em.load_state_dict(ck["model"])
        tab = score_loader(em, loaders["test"], vmin, vmax, device, image_size, desc=f"test/{tag}")
        print(
            f"  {tag:9s} (e{ck['epoch']}): Net RMSE {tab['rmse'][0]:.2f} nT  "
            f"nRMSE {tab['rmse_norm'][0]:.4f}  SSIM {tab['ssim'][0]:.4f}  MS-SSIM {tab['msssim'][0]:.4f}"
        )
        if run is not None:
            wandb.run.summary[f"test/{tag}/rmse_nt"] = tab["rmse"][0]
            wandb.run.summary[f"test/{tag}/ssim"] = tab["ssim"][0]

    if run is not None:
        run.finish()


def parse_args() -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--lr-start", type=float, default=d.lr_start)
    p.add_argument(
        "--lr-end", type=float, default=d.lr_end, help="Cosine floor (eta_min); --sched cosine only."
    )
    p.add_argument("--sched", choices=["onecycle", "cosine"], default=d.sched)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--pct-start", type=float, default=d.pct_start)
    p.add_argument("--val-every", type=int, default=d.val_every)
    p.add_argument("--ckpt-dir", type=Path, default=d.ckpt_dir)
    p.add_argument("--wandb-project", default=d.wandb_project)
    p.add_argument("--wandb-entity", default=d.wandb_entity)
    p.add_argument("--no-wandb", action="store_true", default=d.no_wandb)
    p.add_argument("--no-augment", dest="augment", action="store_false", default=d.augment)
    p.add_argument("--base-channels", type=int, default=d.base_channels)
    p.add_argument("--num-res-blocks", type=int, default=d.num_res_blocks)
    p.add_argument(
        "--channel-mult",
        type=int,
        nargs="+",
        default=list(d.channel_mult),
        help="Per-level width multipliers, e.g. --channel-mult 1 1 2 (depth = count).",
    )
    add_channel_args(p, defaults=d, dem_modes=("grad",))
    p.add_argument(
        "--fvd-weight",
        type=float,
        default=d.fvd_weight,
        help="λ for |FVD(pred)-FVD(hr)| physics penalty on the mag output (0=off).",
    )
    p.add_argument("--patch-dir", type=Path, default=d.patch_dir)
    a = p.parse_args()
    return TrainConfig(
        run_name=a.run_name,
        seed=a.seed,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        num_epochs=a.num_epochs,
        lr_start=a.lr_start,
        lr_end=a.lr_end,
        sched=a.sched,
        weight_decay=a.weight_decay,
        grad_clip=a.grad_clip,
        pct_start=a.pct_start,
        val_every=a.val_every,
        ckpt_dir=a.ckpt_dir,
        wandb_project=a.wandb_project,
        wandb_entity=a.wandb_entity,
        no_wandb=a.no_wandb,
        augment=a.augment,
        base_channels=a.base_channels,
        num_res_blocks=a.num_res_blocks,
        channel_mult=tuple(a.channel_mult),
        lr_aux=tuple(a.lr_aux),
        use_dem=a.use_dem,
        dem_mode=a.dem_mode,
        fvd_weight=a.fvd_weight,
        patch_dir=a.patch_dir,
    )


if __name__ == "__main__":
    main()

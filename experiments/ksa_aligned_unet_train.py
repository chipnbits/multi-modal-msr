"""Train an guided-diffusion style U-Net aeromagnetic patches with L1 regression.

LR -> Bicubic Upsample -> Learned U-net Corrector -> SR Output.

Launch (single GPU):
    # baseline — magnetic only, in=1
    uv run python experiments/ksa_aligned_unet_train.py \
        --run-name unet_x3_ksa_f3_baseline \
        --index-dir data/processed/ksa_aligned/patch_indices_cellgrid8s12_fold3

    # multimodal — mag + 1VD + DEM-gradient, in=4 (matches the winning RDN combo inputs)
    uv run python experiments/ksa_aligned_unet_train.py \
        --run-name unet_x3_ksa_f3_1vd_demgrad \
        --index-dir data/processed/ksa_aligned/patch_indices_cellgrid8s12_fold3 \
        --use-dem --dem-mode grad --lr-aux 1VD
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from magsr.datasets import build_ksa_aligned_datasets, pool_collate, random_d4, worker_init_fn
from magsr.fourier import FVDLoss
from magsr.metrics import (
    MaskedL1Loss,
    apply_hr_nan_mask,
    mean_std,
    per_patch_msssim,
    per_patch_rmse,
    per_patch_ssim,
)
from magsr.models.unet import UNetModelWrapper
from magsr.normalize import from_pm1, to_pm1


# -----------------------------------------------------------------
# Model
# -----------------------------------------------------------------
class MSRRegressionUNet(nn.Module):
    """U-Net adapted for L1 MSR regression (HR-resolution corrector + residual learning).

    Input takes upsample HR-resolution channel stack (B, C, H, W) with input modalities normed to
    [-1, 1]. Unet time embed disabled (use_timestep=False). Structured as a learned residual on top of the
    classical upsampling interpolation. Final conv set to zero to begin with identity SR.
    """

    def __init__(
        self,
        in_channels: int,
        image_size: int,
        out_channels: int = 1,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: tuple[int, ...] = (1, 2, 4),  # ~15M params, lean full-res (64ch) stage
        attention_resolutions: str | None = None,  # None -> attention only at the bottleneck stage
        dropout: float = 0.0,
        class_cond: bool = False,
        num_classes: int = 3,
        use_timestep: bool = False,  # regression: time step is off.
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.num_domains = 1

        # Check image size matches
        bottleneck_ds = 2 ** (len(channel_mult) - 1)
        if self.image_size % bottleneck_ds != 0:
            raise ValueError(
                f"image_size {self.image_size} not divisible by "
                f"{bottleneck_ds} for channel_mult {channel_mult}"
            )
        # Parse None attn into bottleneck-only
        attention_resolutions = attention_resolutions or str(self.image_size // bottleneck_ds)

        self.unet = UNetModelWrapper(
            dim=(self.in_channels, self.image_size, self.image_size),
            num_channels=base_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            learn_sigma=False,
            class_cond=class_cond,
            num_classes=num_classes,
            use_timestep=use_timestep,
        )

        # Project to out channel size with 1x1 conv.
        self.final_proj = (
            nn.Conv2d(self.in_channels, out_channels, kernel_size=1)
            if self.in_channels != out_channels
            else nn.Identity()
        )

        # Identity-residual start: the wrapped UNet already zero-inits its OWN output
        # module, so h ~= 0 at init regardless of final_proj. Zero-initing final_proj's
        # WEIGHT too (when in!=out) puts two zero layers in series -> dL/dh = W_fp^T.dL/dout
        # = 0 permanently blocks gradient to the UNet, and dL/dW_fp ~ h ~= 0, so only the
        # bias can ever learn (model frozen at bicubic + a constant). Keep the weight at
        # default init so gradients flow; zero only the bias so the residual still starts
        # at ~0 (h ~= 0 => W_fp . h ~= 0). The in==out path uses Identity and is unaffected.
        if isinstance(self.final_proj, nn.Conv2d):
            nn.init.zeros_(self.final_proj.bias)

        # Stash constructor kwargs so checkpoints can rebuild the model for later eval.
        self.build_kwargs = dict(
            in_channels=self.in_channels,
            image_size=self.image_size,
            out_channels=out_channels,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=tuple(channel_mult),
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            class_cond=class_cond,
            num_classes=num_classes,
            use_timestep=use_timestep,
        )

    def forward(self, x_hr: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = self.unet(x_hr, t=None, y=y)
        return x_hr[:, :1] + self.final_proj(h)


# -----------------------------------------------------------------
# Config
# -----------------------------------------------------------------
@dataclass
class TrainConfig:
    run_name: str = "unet_x3_ksa_f3_baseline"
    index_dir: Path | None = None
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 16
    num_epochs: int = 250
    lr_start: float = 3e-4
    lr_end: float = 1e-5  # cosine floor (eta_min); only used when sched="cosine"
    sched: str = (
        "onecycle"  # "onecycle" (ramp to lr_start then ->~0) or "cosine" (start lr_start, floor lr_end)
    )
    weight_decay: float = 0.01  # AdamW default; applied to conv/linear weights only
    grad_clip: float = 1.0
    pct_start: float = 0.05
    val_every: float = 1.0  # validate + checkpoint every N epochs (fractional ok, e.g. 0.25 on dense s12)
    ckpt_dir: Path = Path("checkpoints")
    wandb_project: str = "magsr-ksa-unet"
    wandb_entity: str | None = None
    no_wandb: bool = False
    # model
    base_channels: int = 64
    num_res_blocks: int = 2
    channel_mult: tuple[int, ...] = (
        1,
        2,
        4,
    )  # per-level width multipliers (depth = len); base must stay div by 32
    class_cond: bool = False  # condition on survey block (3 classes) via the label embedding
    # inputs
    use_dem: bool = False
    dem_mode: str = "grad"
    lr_aux: tuple[str, ...] = ()
    fvd_weight: float = 0.0  # optional |FVD(pred)-FVD(hr)| 1VD physics penalty (0 = pure L1)
    augment: bool = True


def adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay only multi-dim weights (conv/linear); exclude biases + GroupNorm params (ndim<2)
    and the survey-class embedding, which weight decay would just shrink toward zero."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "label_emb" in name else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------
# Data
# -----------------------------------------------------------------
def build_loaders(cfg: TrainConfig) -> tuple[dict, dict[str, DataLoader]]:
    splits = build_ksa_aligned_datasets(
        index_dir=cfg.index_dir,
        load_dem=cfg.use_dem,
        dem_mode=cfg.dem_mode,
        lr_aux_products=cfg.lr_aux,
    )
    cfg0 = next(iter(splits.values())).config
    collate = partial(pool_collate, hr_products=cfg0.hr_products, lr_products=cfg0.lr_products)

    def make(split: str, shuffle: bool) -> DataLoader:
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
            drop_last=shuffle,
        )

    loaders = {
        "train": make("train", True),
        "val": make("val", False),
        "test": make("test", False),
    }
    return splits, loaders


def blocks_to_class(blocks: list[int], device: torch.device) -> torch.Tensor:
    """KSA survey block ids (1,2,3) -> 0-based class indices for the label embedding."""
    return torch.tensor([b - 1 for b in blocks], device=device, dtype=torch.long)


def build_hr_input(ds, batch: dict, device: torch.device, image_size: int):
    """Build the HR-resolution [-1,1] input stack + HR target (NaN preserved for masking).

    Channel layout all at `image_size`:
      ch0  magnetic   — LR normalized [0,1] then bicubic-upsampled (the residual base)
      ...  DEM-grad   — HR-NATIVE: pooled 30 m -> 60 m, gradient on the HR grid (2ch)
      ...  LR aux     — 1VD etc., LR [0,1] bicubic-upsampled (LR-sourced, no new HR detail)
    """
    lr_t = batch["lr"].to(device, non_blocking=True)
    hr_t = batch["hr"][:, :1].to(device, non_blocking=True)
    blocks = [m["block"] for m in batch["meta"]]
    sz = (image_size, image_size)

    def up(x):  # bicubic to HR grid, kept in [0,1]
        return F.interpolate(x, size=sz, mode="bicubic", align_corners=False).clamp(0.0, 1.0)

    mag = up(ds.normalize(lr_t, blocks=blocks).nan_to_num(0.5))  # input NaNs -> patch-neutral 0.5
    chans = [mag]
    if ds.config.load_dem:
        if ds.config.dem_mode != "grad":
            raise NotImplementedError("HR corrector currently supports --dem-mode grad only")
        chans.append(ds.dem_grad_to_hr(batch["dem"].to(device)))  # already HR-native [0,1]
    if ds.lr_aux_products:
        chans.append(up(ds.lr_aux_to_channels(batch["lr_aux"].to(device))))
    x = to_pm1(torch.cat(chans, dim=1))
    hr_n = to_pm1(ds.normalize(hr_t, blocks=blocks))  # NaN survives -> masked in loss/metrics
    return x, hr_n, blocks


@torch.no_grad()
def score_loader(model, loader, ds, device, image_size, desc, class_cond=False):
    """Net RMSE(nT)/SSIM/MS-SSIM + per-patch normalized RMSE over a split, using the shared
    per-patch metric primitives. Mirrors evaluate_loader but on the HR [-1,1] corrector path.
    All metrics are per-patch then mean-aggregated (report convention); rmse_norm is the
    per-patch normalized RMSE, not a separately-pooled quantity."""
    model.eval()
    vals = {m: [] for m in ("rmse", "rmse_norm", "ssim", "msssim")}
    blks = {m: [] for m in ("rmse", "ssim", "msssim")}
    for batch in tqdm(loader, desc=desc, ncols=100):
        x, hr_n, blocks = build_hr_input(ds, batch, device, image_size)
        blocks_t = torch.tensor(blocks, dtype=torch.long)
        y = blocks_to_class(blocks, device) if class_cond else None
        sr01 = from_pm1(model(x, y).clamp(-1.0, 1.0))  # [0,1] for metric/denormalize parity with RDN
        hr01 = from_pm1(hr_n)  # keeps NaN
        mask = hr01.isfinite().to(sr01.dtype)
        sr_m, hr_m = apply_hr_nan_mask(sr01, hr01)
        sr_nt = ds.denormalize(sr_m, blocks=blocks) * mask
        hr_nt = ds.denormalize(hr_m, blocks=blocks) * mask
        vals["rmse"].append(per_patch_rmse(sr_nt, hr_nt, mask).cpu())
        vals["rmse_norm"].append(per_patch_rmse(sr_m, hr_m, mask).cpu())
        blks["rmse"].append(blocks_t)
        fully = mask.flatten(1).all(dim=1)
        if fully.any():
            sv, hv, bv = sr_m[fully], hr_m[fully], blocks_t[fully.cpu()]
            vals["ssim"].append(per_patch_ssim(sv, hv, data_range=1.0).cpu())
            vals["msssim"].append(per_patch_msssim(sv, hv, patch_size=image_size, data_range=1.0).cpu())
            blks["ssim"].append(bv)
            blks["msssim"].append(bv)
    model.train()
    out = {
        m: mean_std(torch.cat(vals[m])) if vals[m] else (float("nan"), float("nan"))
        for m in ("rmse", "rmse_norm", "ssim", "msssim")
    }
    # Per-block (B1/B2/B3) scopes for parity with the RDN evaluate_loader, kept under
    # "{metric}_blocks" so out[metric] stays the Net (mean,std) tuple the trainer expects.
    for m in ("rmse", "ssim", "msssim"):
        if vals[m]:
            allv, allb = torch.cat(vals[m]), torch.cat(blks[m])
            out[f"{m}_blocks"] = {int(b): mean_std(allv[allb == b]) for b in allb.unique().tolist()}
    return out


# -----------------------------------------------------------------
# Train
# -----------------------------------------------------------------
def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, loaders = build_loaders(cfg)
    ds = splits["train"]
    image_size = int(ds.config.patch_px)
    dem_ch = (2 if cfg.dem_mode == "grad" else 1) if cfg.use_dem else 0
    in_channels = 1 + dem_ch + len(cfg.lr_aux)
    print(f"splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    num_classes = len(ds.config.blocks)
    print(
        f"in_channels={in_channels} (dem={cfg.use_dem}/{cfg.dem_mode if cfg.use_dem else '-'}, "
        f"aux={cfg.lr_aux or '-'})  HR patch={image_size}  "
        f"class_cond={cfg.class_cond} ({num_classes} survey blocks)"
    )

    model = MSRRegressionUNet(
        in_channels=in_channels,
        image_size=image_size,
        base_channels=cfg.base_channels,
        num_res_blocks=cfg.num_res_blocks,
        channel_mult=cfg.channel_mult,
        class_cond=cfg.class_cond,
        num_classes=num_classes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MSRRegressionUNet: {n_params:,} trainable params")

    criterion = MaskedL1Loss()
    fvd = FVDLoss(dx=1.0, dy=1.0).to(device) if cfg.fvd_weight > 0 else None
    optim = AdamW(adamw_param_groups(model, cfg.weight_decay), lr=cfg.lr_start, fused=True)
    steps_per_epoch = len(loaders["train"])
    total_steps = cfg.num_epochs * steps_per_epoch
    if cfg.sched == "cosine":
        # Short linear warmup (avoid blasting full LR into a cold U-Net) then cosine
        # decay from lr_start down to lr_end (a floor, not ~0 — keeps the model moving
        # instead of memorizing the train set at a vanishing LR).
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
            },
        )

    def save_ckpt(tag: str, epoch: int, metrics: dict) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "build_kwargs": model.build_kwargs,
                "config": {**asdict(cfg), "index_dir": str(cfg.index_dir), "in_channels": in_channels},
                **metrics,
            },
            cfg.ckpt_dir / f"{cfg.run_name}_{tag}.pt",
        )

    def validate(epoch_label: float) -> dict:
        tab = score_loader(
            model,
            loaders["val"],
            ds,
            device,
            image_size,
            desc=f"val e{epoch_label:.2f}",
            class_cond=cfg.class_cond,
        )
        return {
            "val_rmse_nt": tab["rmse"][0],
            "val_rmse_norm": tab["rmse_norm"][0],
            "val_ssim": tab["ssim"][0],
            "val_msssim": tab["msssim"][0],
        }

    best_ssim, best_rmse = -math.inf, math.inf
    global_step = 0
    val_interval = max(1, round(cfg.val_every * steps_per_epoch))  # step-based: fractional epochs ok

    def run_validation(epoch_label: float) -> None:
        nonlocal best_ssim, best_rmse
        m = validate(epoch_label)
        print(
            f"  e{epoch_label:.2f}: val RMSE {m['val_rmse_nt']:.2f} nT  nRMSE {m['val_rmse_norm']:.4f}  "
            f"SSIM {m['val_ssim']:.4f}  MS-SSIM {m['val_msssim']:.4f}"
        )
        if run is not None:
            wandb.log({f"val/{k}": v for k, v in m.items()} | {"epoch": epoch_label}, step=global_step)
        save_ckpt("last", epoch_label, m)
        if m["val_ssim"] > best_ssim:
            best_ssim = m["val_ssim"]
            save_ckpt("best", epoch_label, m)
        if m["val_rmse_nt"] < best_rmse:
            best_rmse = m["val_rmse_nt"]
            save_ckpt("best_rmse", epoch_label, m)

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"[{cfg.run_name}] {epoch:03d}/{cfg.num_epochs}", ncols=100)
        for i, batch in enumerate(pbar, 1):
            x, hr_n, blocks = build_hr_input(ds, batch, device, image_size)
            if cfg.augment:
                x, hr_n = random_d4(x, hr_n)
            y = blocks_to_class(blocks, device) if cfg.class_cond else None
            optim.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                pred = model(x, y)
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
            if global_step % val_interval == 0:
                run_validation(epoch - 1 + i / steps_per_epoch)

    # Final validation/checkpoint if the last step didn't land on a val boundary.
    if global_step % val_interval != 0:
        run_validation(float(cfg.num_epochs))

    # Final held-out test scoring of the two selected checkpoints.
    print("\n=== TEST (fold-3 held-out) ===")
    for tag in ("best", "best_rmse"):
        ck_path = cfg.ckpt_dir / f"{cfg.run_name}_{tag}.pt"
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        em = MSRRegressionUNet(**ck["build_kwargs"]).to(device)
        em.load_state_dict(ck["model"])
        tab = score_loader(
            em, loaders["test"], ds, device, image_size, desc=f"test/{tag}", class_cond=cfg.class_cond
        )
        print(
            f"  {tag:9s} (e{ck['epoch']}): Net RMSE {tab['rmse'][0]:.2f} nT  "
            f"nRMSE {tab['rmse_norm'][0]:.4f}  SSIM {tab['ssim'][0]:.4f}  "
            f"MS-SSIM {tab['msssim'][0]:.4f}"
        )
        if run is not None:
            wandb.run.summary[f"test/{tag}/rmse_nt"] = tab["rmse"][0]
            wandb.run.summary[f"test/{tag}/ssim"] = tab["ssim"][0]

    if run is not None:
        run.finish()


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def parse_args() -> TrainConfig:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train an L1-regression U-Net on KSA fold-3 patches.")
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument(
        "--index-dir",
        type=Path,
        default=d.index_dir,
        help="Split JSON dir, e.g. data/processed/ksa_aligned/patch_indices_cellgrid8s12_fold3",
    )
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--lr-start", type=float, default=d.lr_start)
    p.add_argument(
        "--lr-end",
        type=float,
        default=d.lr_end,
        help="Cosine floor (eta_min); only used with --sched cosine.",
    )
    p.add_argument(
        "--sched",
        choices=["onecycle", "cosine"],
        default=d.sched,
        help="LR schedule: onecycle (default) or cosine (warmup->lr_start, decay->lr_end).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=d.weight_decay,
        help="AdamW weight decay on conv/linear weights (norms/biases/embedding excluded).",
    )
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--pct-start", type=float, default=d.pct_start)
    p.add_argument(
        "--val-every",
        type=float,
        default=d.val_every,
        help="Validate every N epochs; fractional ok (e.g. 0.25 on dense s12).",
    )
    p.add_argument("--ckpt-dir", type=Path, default=d.ckpt_dir)
    p.add_argument("--wandb-project", default=d.wandb_project)
    p.add_argument("--wandb-entity", default=d.wandb_entity)
    p.add_argument("--no-wandb", action="store_true", default=d.no_wandb)
    p.add_argument("--base-channels", type=int, default=d.base_channels)
    p.add_argument("--num-res-blocks", type=int, default=d.num_res_blocks)
    p.add_argument(
        "--channel-mult",
        type=int,
        nargs="+",
        default=list(d.channel_mult),
        help="Per-level width multipliers, e.g. --channel-mult 1 2 2 (depth = count).",
    )
    p.add_argument(
        "--class-cond",
        action="store_true",
        default=d.class_cond,
        help="Condition the U-Net on the survey block (3 classes) via the label embedding.",
    )
    add_channel_args(p, defaults=d)
    p.add_argument(
        "--fvd-weight",
        type=float,
        default=d.fvd_weight,
        help="Weight of the optional |FVD(pred)-FVD(hr)| physics penalty (0 = pure L1).",
    )
    p.add_argument("--no-augment", dest="augment", action="store_false", default=d.augment)
    a = p.parse_args()
    return TrainConfig(
        run_name=a.run_name,
        index_dir=a.index_dir,
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
        base_channels=a.base_channels,
        num_res_blocks=a.num_res_blocks,
        channel_mult=tuple(a.channel_mult),
        class_cond=a.class_cond,
        use_dem=a.use_dem,
        dem_mode=a.dem_mode,
        lr_aux=tuple(a.lr_aux),
        fvd_weight=a.fvd_weight,
        augment=a.augment,
    )


if __name__ == "__main__":
    main()

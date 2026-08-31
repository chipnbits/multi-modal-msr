"""Evaluate synthetic-UC-trained checkpoints on seeded operator-consistent test pairs.

This evaluation flow handles the production of the synthetic LR pair via the UC

    uv run python experiments/eval_synuc.py \
        --ckpt checkpoints/rdnpp_..._p264_synuc_mag_best_rmse.pt \
        --index-dir data/processed/ksa_aligned/patch_indices_cellgrid8s12_p264_fold3
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from ksa_aligned_evaluate import evaluate_loader  # the canonical eval loop
from torch.utils.data import DataLoader

from magsr.datasets import build_ksa_aligned_datasets, pool_collate, worker_init_fn
from magsr.fourier.fvd import first_vertical_derivative
from magsr.fourier.synth import synth_uc_pair, synth_uc_pair_drape
from magsr.models import load_checkpoint


def score_synuc(
    ckpt_path: Path,
    index_dir: Path,
    *,
    drape: bool = False,
    split: str = "test",
    dz_min: float = 200.0,
    dz_max: float = 250.0,
    noise_nt: float = 1.0,
    out_px: int = 132,
    seed_base: int = 20_000,
    batch_size: int = 64,
    num_workers: int = 12,
) -> dict:
    """
    Score one synuc checkpoint on seeded synthetic pairs -> {metric: {scope: (mean, std)}}.


    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(ckpt_path, device=dev, return_ckpt=True)
    cfg = ckpt.get("config", {}) or {}
    use_dem = bool(cfg.get("use_dem", False))
    lr_aux = tuple(cfg.get("lr_aux", ()) or ())
    multi = int(cfg.get("num_domains", 1) or 1) > 1
    model.eval()
    print(f"{ckpt_path.name}: use_dem={use_dem} lr_aux={lr_aux} multi={multi}")

    splits = build_ksa_aligned_datasets(
        index_dir=index_dir,
        load_dem=use_dem or drape,
        dem_mode=cfg.get("dem_mode", "grad"),
        lr_aux_products=lr_aux,
    )
    ds = splits[split]
    collate = partial(pool_collate, hr_products=ds.config.hr_products, lr_products=ds.config.lr_products)
    loader = DataLoader(
        ds,
        collate_fn=collate,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )

    # Synthetic pairs are the ONLY thing that differs from the real-LR eval: `predict` regenerates
    # the seeded LR (and 1VD from it) per batch, then the shared evaluate_loader does the metrics.
    def predict(batch, bi):
        blocks = [m["block"] for m in batch["meta"]]
        hr_full = batch["hr"][:, :1].to(dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed_base + bi)
        dz = dz_min + (dz_max - dz_min) * torch.rand(hr_full.shape[0], generator=gen, device=dev)
        if drape:
            dem_t = batch["dem"].to(dev).float()
            if dem_t.ndim == 3:
                dem_t = dem_t.unsqueeze(1)
            dem_hr = torch.nn.functional.avg_pool2d(dem_t, 2, 2).squeeze(1)
            hr_c, lr_syn = synth_uc_pair_drape(
                hr_full, dem_hr, dz, noise_nt=noise_nt, out_px=out_px, generator=gen
            )
        else:
            hr_c, lr_syn = synth_uc_pair(hr_full, dz, noise_nt=noise_nt, out_px=out_px, generator=gen)
        dem_crop = None
        if use_dem:
            dem = batch["dem"].to(dev)
            offd = (dem.shape[-1] - out_px * 2) // 2
            dem_crop = dem[..., offd : offd + out_px * 2, offd : offd + out_px * 2]
        aux = (
            first_vertical_derivative(lr_syn[:, 0].float(), dx=180.0, dy=180.0).unsqueeze(1)
            if lr_aux
            else None
        )
        inp = ds.assemble_lr_input(lr_syn, blocks, dem=dem_crop, lr_aux=aux)
        blk_t = torch.tensor(blocks, dtype=torch.long, device=dev)
        sr = (model(inp, blk_t) if multi else model(inp)).clamp(0, 1)
        return sr, ds.normalize(hr_c, blocks=blocks), blocks

    # patch_size = the cropped synthetic output (out_px), NOT the oversized p264 index patch — MS-SSIM
    # scale-count depends on it (out_px 132 < 161 -> 4-scale, matching the original `ps = pred.shape[-1]`).
    table = evaluate_loader(
        model,
        loader,
        device=torch.device(dev),
        desc=f"eval-synuc {split}",
        predict=predict,
        patch_size=out_px,
    )
    table = {m: table[m] for m in ("rmse", "ssim", "msssim") if m in table}
    print(
        f"net per-patch nT RMSE: {table['rmse']['Net'][0]:.2f} ± {table['rmse']['Net'][1]:.2f}"
        f"  SSIM {table['ssim']['Net'][0]:.3f}  MS-SSIM {table['msssim']['Net'][0]:.3f}"
    )
    for x in sorted(k for k in table["rmse"] if isinstance(k, int)):
        print(f"  B{x}: {table['rmse'][x][0]:.2f}")
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--dz-min", type=float, default=200.0)
    p.add_argument("--dz-max", type=float, default=250.0)
    p.add_argument("--noise-nt", type=float, default=1.0)
    p.add_argument("--out-px", type=int, default=132)
    p.add_argument("--drape", action="store_true", help="Drape-aware generator (must match training).")
    p.add_argument("--seed-base", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=12)
    a = p.parse_args()
    score_synuc(
        a.ckpt,
        a.index_dir,
        drape=a.drape,
        split=a.split,
        dz_min=a.dz_min,
        dz_max=a.dz_max,
        noise_nt=a.noise_nt,
        out_px=a.out_px,
        seed_base=a.seed_base,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
    )


if __name__ == "__main__":
    main()

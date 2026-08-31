"""Torch `collate_fn` + `worker_init_fn` for magsr datasets.

`pool_collate` stacks each sample's per-product tensors into a single
`(C, H, W)` tensor (one channel per product), then batches into
`(B, C, H, W)`. Feed into `DataLoader(..., collate_fn=pool_collate)`.

`worker_init_fn` drops raster handles inherited from the parent process
when `num_workers > 0` on fork-based platforms (Linux default). Without
it, the lazy KSA backends produce intermittent `TIFFReadEncodedTile()
failed` errors because libtiff state can't be shared safely across forked
processes.

Because HR/LR tensors share the same `(B, C, H, W)` layout, flips,
rotations, and other torchvision transforms can be applied once per tensor
and stay in sync across channels. Apply them in the training loop.
"""

from __future__ import annotations

from typing import Any, Sequence


def pool_collate(
    batch: Sequence[dict[str, Any]],
    *,
    hr_products: Sequence[str] | None = None,
    lr_products: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Collate samples, stacking per-product tensors into channel axes.

    Input: list of `{hr: {prod: (H, W)}, lr: {...},
    dem: (H, W)?, meta: {...}}`.

    Output:
        {
            "hr":      (B, C_hr, H_hr, W_hr),
            "lr":      (B, C_lr, H_lr, W_lr),
            "dem":      (B, 1, H_dem, W_dem),    # only if base emits `dem`
            "meta": [sample_meta, ...],
        }

    Channel order follows `hr_products` / `lr_products` when supplied,
    otherwise the iteration order of each sample's dict (all samples must
    agree on keys and order). `meta` is passed through as a list of dicts
    rather than being tensor-ified, since it carries mixed types.
    """
    import torch

    if not batch:
        raise ValueError("pool_collate received an empty batch")

    def resolve_keys(passed: Sequence[str] | None, sample_dict: dict[str, Any]) -> list[str]:
        return list(passed) if passed is not None else list(sample_dict.keys())

    hr_keys = resolve_keys(hr_products, batch[0]["hr"])
    lr_keys = resolve_keys(lr_products, batch[0]["lr"])

    def stack_products(sample: dict[str, Any], group: str, keys: Sequence[str]) -> "torch.Tensor":
        # Each per-product tensor is (H, W); stack along a new channel dim
        # to produce (C, H, W) for this sample.
        return torch.stack([sample[group][k] for k in keys], dim=0)

    hr = torch.stack([stack_products(s, "hr", hr_keys) for s in batch])
    lr = torch.stack([stack_products(s, "lr", lr_keys) for s in batch])

    out: dict[str, Any] = {
        "hr": hr,
        "lr": lr,
        "meta": [s["meta"] for s in batch],
    }

    if "dem" in batch[0]:
        # DEM is a single raster, not nested under products. Stack to
        # (B, 1, H, W) so it carries the same channel dim as hr/lr.
        out["dem"] = torch.stack([s["dem"] for s in batch]).unsqueeze(1)

    if "lr_aux" in batch[0]:
        # Aux LR products: nested {product: (H_lr, W_lr)} per sample, in
        # config.lr_aux_products order. Stack to (B, C_aux, H_lr, W_lr) keeping that
        # channel order so dataset.lr_aux_to_channels normalizes each correctly.
        aux_keys = list(batch[0]["lr_aux"].keys())
        out["lr_aux"] = torch.stack([torch.stack([s["lr_aux"][k] for k in aux_keys], dim=0) for s in batch])

    if "ms" in batch[0]:
        # Landsat bands: nested {band: (H_dem, W_dem)} per sample, in config.ms_bands
        # order. Stack to (B, n_bands, H_dem, W_dem); pooled to LR in ms_to_channels.
        ms_keys = list(batch[0]["ms"].keys())
        out["ms"] = torch.stack([torch.stack([s["ms"][k] for k in ms_keys], dim=0) for s in batch])

    return out


def worker_init_fn(worker_id: int) -> None:
    """Drop inherited raster handles at the start of each DataLoader worker.

    Usage:
        DataLoader(
            dataset,
            num_workers=4,
            collate_fn=pool_collate,
            worker_init_fn=worker_init_fn,
        )
    """
    from magsr.datasets.io import clear_source_cache

    clear_source_cache()

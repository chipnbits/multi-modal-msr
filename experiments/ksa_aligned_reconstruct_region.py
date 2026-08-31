"""Reconstruct a single polygonal region defined by a vector AOI file.

Sibling to `ksa_aligned_reconstruct.py`. Drops patches outside the AOI polygon and
patches whose LR window is mostly NoData. Writes one HR GeoTIFF and a composite PNG.

Usage:
    uv run python experiments/ksa_aligned_reconstruct_region.py \
        --checkpoint checkpoints/<run>/<run>_best.pt \
        --aoi  "$MAGSR_KSA_ALIGNED_ROOT/_shp/reconstruction_zone/aoi_for_msr.gpkg" \
        --stride-lr-px 11

"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
import torch
from shapely.ops import unary_union

from magsr.cli import add_channel_args
from magsr.datasets import KSAAlignedConfig
from magsr.datasets.ksa_shield_aligned import KSAShieldAlignedDataset
from magsr.models import load_checkpoint
from magsr.reconstruct import (
    Normalizer,
    ReconItem,
    assign_patch_blocks,
    build_aux_lr_channels,
    plan_lr_patches,
    plot_multi_block_grid,
    reconstruct_region_multichannel,
    write_reconstruction_geotiff,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--aoi", type=Path, required=True, help="Vector AOI (.gpkg, .geojson, .shp)")
    p.add_argument("--label", default="zone", help="Label used in output filenames + composite")
    p.add_argument("--lr-product", default="RTP")
    p.add_argument("--hr-truth-product", default="AMF_RTP")
    p.add_argument("--stride-lr-px", type=int, default=None)
    p.add_argument("--buffer-lr-px", type=int, default=0)
    p.add_argument("--min-valid-frac", type=float, default=0.5)
    p.add_argument("--blend-kind", default="auto", choices=("auto", "linear", "cosine", "gaussian", "ones"))
    p.add_argument("--batch-size", type=int, default=8)
    add_channel_args(p)
    p.add_argument(
        "--block-id",
        type=int,
        default=2,
        help="Scalar 1-indexed survey block for multi-domain (num_domains>1) models; "
        "AOI is B2-dominant so 2 is the default. Ignored when --block-by-location is set.",
    )
    p.add_argument(
        "--block-by-location",
        action="store_true",
        help="Assign each patch's survey block by its geographic center via point-in-polygon "
        "against the block shapefiles (nearest block for gap patches). Overrides --block-id.",
    )
    p.add_argument(
        "--block-shp-dir",
        type=Path,
        default=None,
        help="Directory of per-block survey shapefiles (Geophysics_BLOCK_{1,2,3}.shp) used by "
        "--block-by-location. Default: <ksa_aligned_root>/_shp/Geophysics.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("figures/recon_zone"))
    return p.parse_args()


def load_aoi_geometry(path: Path, target_crs: str | object) -> object:
    gdf = gpd.read_file(path).to_crs(target_crs)
    return unary_union(gdf.geometry)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    multichannel = bool(args.use_dem or args.lr_aux)
    cfg = KSAAlignedConfig.from_yaml(
        load_dem=bool(args.use_dem),
        dem_mode=args.dem_mode,
        lr_aux_products=tuple(args.lr_aux),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_lr = cfg.patch_px // cfg.lr_scale
    stride_lr = args.stride_lr_px if args.stride_lr_px is not None else patch_lr // 2
    lr_path = cfg.inference_lr_product_path(args.lr_product)
    truth_path = cfg.hr_product_path(args.hr_truth_product) if args.hr_truth_product else None

    with rasterio.open(lr_path) as src:
        lr_crs = src.crs
    aoi_geom = load_aoi_geometry(args.aoi, lr_crs)

    normalizer = Normalizer.from_json(cfg.normalization_path)

    # Aux LR products read from the full-coverage inference LR rasters (same grid as the mag LR).
    aux_dataset = None
    lr_aux_paths: dict[str, Path] | None = None
    if multichannel:
        aux_dataset = KSAShieldAlignedDataset.for_channels(cfg, product=args.lr_product)
        lr_aux_paths = {p: cfg.inference_lr_product_path(p) for p in cfg.lr_aux_products}

    model = load_checkpoint(args.checkpoint, device=device)

    plan = plan_lr_patches(
        polygon_world=aoi_geom,
        lr_path=lr_path,
        patch_lr_px=patch_lr,
        stride_lr_px=stride_lr,
        buffer_lr_px=args.buffer_lr_px,
        scale=cfg.lr_scale,
        min_valid_frac=args.min_valid_frac,
    )
    aux_channels = (
        build_aux_lr_channels(
            plan,
            dataset=aux_dataset,
            lr_aux_paths=lr_aux_paths,
            dem_path=cfg.dem_path,
        )
        if multichannel
        else None
    )

    # Domain tagging for multi-domain (num_domains>1) models: per-patch by location, or scalar.
    multi_domain = getattr(model, "num_domains", 1) > 1
    block_id = None
    block_ids = None
    if multi_domain:
        if args.block_by_location:
            block_shp_dir = args.block_shp_dir or (cfg.root / "_shp" / "Geophysics")
            block_shp_paths = {b: block_shp_dir / f"Geophysics_BLOCK_{b}.shp" for b in (1, 2, 3)}
            block_ids = assign_patch_blocks(plan, block_shp_paths)
            counts = {b: int((block_ids == b).sum()) for b in (1, 2, 3)}
            print(f"block distribution (per-patch by location): {counts}")
        else:
            block_id = args.block_id
            print(f"block tag (scalar): {block_id}")

    arr = reconstruct_region_multichannel(
        model,
        lr_path,
        plan,
        normalizer=normalizer,
        aux_channels=aux_channels,
        blend_kind=args.blend_kind,
        batch_size=args.batch_size,
        device=device,
        block_id=block_id,
        block_ids=block_ids,
    )
    out_tif = args.out_dir / f"recon_{args.label}.tif"
    write_reconstruction_geotiff(arr, plan, out_tif)
    print(f"{args.label}: {len(plan.positions)} patches → {out_tif}")

    composite = args.out_dir / f"recon_{args.label}_composite.png"
    plot_multi_block_grid(
        [
            ReconItem(
                label=args.label, plan=plan, lr_path=lr_path, recon_path=out_tif, truth_path=truth_path
            )
        ],
        save_path=composite,
        title=f"Patchwise SR — blend={args.blend_kind}, stride_lr={stride_lr}, min_valid_frac={args.min_valid_frac}",
    )
    print(f"composite → {composite}")


if __name__ == "__main__":
    main()

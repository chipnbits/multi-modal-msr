"""
Stage 2: Generate synchronized HR/LR patch pairs from the WA rasters.

Reads the 20m HR and 80m LR GeoTIFFs, applies optional science mask filtering,
and produces normalized .npy patch pairs into train/ and test/ pools.
The train/val split happens later, at dataset construction time.

Defaults come from `configs/datasets.yaml` (`wa` section); CLI flags override
individual fields for ad-hoc runs.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np

from magsr.datasets import WAConfig
from magsr.datasets.western_australia import compute_global_percentiles, generate_wa_patch_pairs

parser = argparse.ArgumentParser(description="Patch the WA aeromagnetic dataset")
parser.add_argument("--config", type=Path, default=None)
parser.add_argument("--hr-patch-px", dest="patch_px", type=int, default=None)
parser.add_argument("--stride-px", type=int, default=None)
parser.add_argument("--min-valid-frac", type=float, default=None)
parser.add_argument(
    "--q-low", dest="q_low", type=int, default=None, help="Lower percentile for clip [0-100]"
)
parser.add_argument(
    "--q-high", dest="q_high", type=int, default=None, help="Upper percentile for clip [0-100]"
)
parser.add_argument("--mask", type=str, default=None, help="Path to science_mask.npy")
parser.add_argument("--mask-meta", type=str, default=None, help="Path to science_mask_meta.json")
parser.add_argument(
    "--lr-aux",
    nargs="*",
    default=None,
    metavar="PROD",
    help="Extra LR products to store, e.g. --lr-aux 1VD (1VD computed from LR mag via FFT).",
)
parser.add_argument(
    "--use-dem",
    action="store_true",
    default=None,
    help="Also patch the snapped DEM (cfg.dem_filename) for the DEM-gradient channel.",
)
parser.add_argument("--patch-dir", type=Path, default=None, help="Override output patch dir.")
args = parser.parse_args()

cfg = WAConfig.from_yaml(args.config)
override_keys = {"patch_px", "stride_px", "min_valid_frac", "q_low", "q_high"}
overrides = {k: v for k, v in vars(args).items() if k in override_keys and v is not None}
if args.lr_aux is not None:
    overrides["lr_aux"] = tuple(args.lr_aux)
if args.use_dem:
    overrides["use_dem"] = True
if args.patch_dir is not None:
    overrides["patch_dir"] = args.patch_dir
if overrides:
    cfg = replace(cfg, **overrides)
print(
    f"Config: lr_aux={cfg.lr_aux} use_dem={cfg.use_dem} dem_mode={cfg.dem_mode} "
    f"patch={cfg.patch_px}/stride={cfg.stride_px} -> {cfg.patch_dir}"
)

hr_path = cfg.data_dir / cfg.hr_filename
lr_path = cfg.data_dir / cfg.lr_filename
geojson_path = cfg.data_dir / cfg.split_filename

split_gdf = gpd.read_file(geojson_path)
train_polygon = split_gdf[split_gdf["set_type"] == "Train"].geometry.iloc[0]

print(f"Computing percentile ({cfg.q_low}-{cfg.q_high}) stats from training region...")
stats = compute_global_percentiles(hr_path, train_polygon, q_low=cfg.q_low, q_high=cfg.q_high)
print(f"Clip range: [{stats['vmin']:.2f}, {stats['vmax']:.2f}] nT")

mask = None
mask_meta = None
if args.mask is not None:
    mask = np.load(args.mask)
    with open(args.mask_meta) as f:
        mask_meta = json.load(f)
    print(f"Science mask loaded: {mask.shape}, valid={mask.mean():.1%}")

print(
    f"Generating patches (hr={cfg.patch_px}px, stride={cfg.stride_px}px, min_valid={cfg.min_valid_frac:.0%})..."
)
manifest = generate_wa_patch_pairs(
    hr_path=hr_path,
    lr_path=lr_path,
    split_gdf=split_gdf,
    output_dir=cfg.patch_dir,
    stats=stats,
    cfg=cfg,
    mask=mask,
    mask_meta=mask_meta,
)

print(f"\nOutput: {cfg.patch_dir}")
print(f"Patches: {manifest['counts']}")

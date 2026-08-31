"""KSA Shield aligned-grid dataset backend.

Reads a pre-snapped grid where
every raster is already co-registered on one UTM37N grid. HR (60 m), LR
(180 m, pre-interpolated), and DEM (30 m) share the same origin, and pixel
ratios are exact integers: `1 LR px = 3 x 3 HR px = 6 x 6 DEM px`.

Block handling`: the 60 m validity raster encodes
each SGS survey block as its own integer ID (0 = no data, 1/2/3 = blocks),
so patches can be enumerated per-block and filtered at build time. A patch's
`source_id` is `ksa_aligned/B{block}`. No UTM zones here because the entire
snapshot lives in EPSG:32637 already.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

from magsr import DATA_DIR, ksa_aligned_root
from magsr.datasets.config import load_yaml_section
from magsr.datasets.io import RasterSource, open_raster
from magsr.datasets.patching import (
    CellGridSpec,
    PatchGridSpec,
    PatchIndex,
    PatchWindow,
    sliding_window_patches,
)
from magsr.normalize import center_scale01, clip_scale01, denorm01


@dataclass(frozen=True)
class CellGridSplitConfig:
    """Cell-grid split scheme parameters (the `cellgrid` yaml subsection).

    Geometry (cell side + buffer, in patches) consumed by
    05_ksa_aligned_build_cell_indices.py; split fractions + seed by
    06_assign_cell_splits.py. CLI flags on those scripts override
    individual fields for ad-hoc variants.
    """

    cell_patches: int = 8
    buffer_patches: int = 1
    min_cell_valid_frac: float = 0.0
    val_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = 42


@dataclass(frozen=True)
class KSAAlignedConfig:
    """Paths + scale constants + patch-build/load params for the pre-snapped KSA snapshot.

    All rasters under `root` share a single EPSG:32637 grid. Scales are in
    pixels-per-pixel between the canonical layers (HR 60 m is the reference).
    The dataset root comes from `MAGSR_KSA_ALIGNED_ROOT`.

    Build/load params come from `configs/datasets.yaml` via `from_yaml()`.
    """

    root: Path = field(default_factory=ksa_aligned_root)
    epsg: str = "EPSG:32637"
    hr_px_m: int = 60
    lr_px_m: int = 180
    dem_px_m: int = 30
    lr_scale: int = 3  # HR pixels per LR pixel
    dem_scale: int = 2  # DEM pixels per HR pixel
    # Patch build
    blocks: tuple[int, ...] | None = None  # None => auto-discover from mask
    patch_px: int = 132
    stride_px: int = 66
    min_valid_frac: float = 0.97
    lr_validity_product: str | None = "RTP"
    # Dataset load
    hr_products: tuple[str, ...] = ("AMF_RTP",)
    lr_products: tuple[str, ...] = ("RTP",)
    load_dem: bool = True
    # DEM channel normalization: per-patch mean removed, recentered to [0, 1]
    dem_relief_m: float = 500.0
    # DEM input representation when `load_dem`:
    #   "relief" -> 1 channel, mean-removed relative relief (the dem_relief_m scaling above)
    #   "grad"   -> 2 channels, pooled-DEM gradient (dz/dx, dz/dy) in m per 180 m pixel,
    dem_mode: str = "relief"
    dem_grad_clip: float = 70.0
    lr_aux_products: tuple[str, ...] = ()
    # Landsat multispectral bands (1..7)
    ms_bands: tuple[int, ...] = ()
    # Derived multispectral band-RATIO features (alteration indices)
    ms_features: tuple[str, ...] = ()
    norm_mode: str = "global"  # "global" or "blockwise" — default for dataset.normalize/denormalize
    # Cell-grid split scheme (04b/05b)
    cellgrid: CellGridSplitConfig = field(default_factory=CellGridSplitConfig)

    def __post_init__(self) -> None:
        if self.norm_mode not in ("global", "blockwise"):
            raise ValueError(f"norm_mode must be 'global' or 'blockwise'; got {self.norm_mode!r}")
        if self.dem_mode not in ("relief", "grad"):
            raise ValueError(f"dem_mode must be 'relief' or 'grad'; got {self.dem_mode!r}")
        bad = [bi for bi in self.ms_bands if bi not in _MS_BAND_ROBUST_RANGE]
        if bad:
            raise ValueError(f"ms_bands {bad} not in 1..7 (known: {sorted(_MS_BAND_ROBUST_RANGE)})")
        badf = [f for f in self.ms_features if f not in _MS_FEATURE_SPEC]
        if badf:
            raise ValueError(f"ms_features {badf} unknown (known: {sorted(_MS_FEATURE_SPEC)})")

    @classmethod
    def default(cls) -> "KSAAlignedConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path: Path | None = None, **overrides: Any) -> "KSAAlignedConfig":
        data = load_yaml_section("ksa_aligned", path=path)
        if "blocks" in data and data["blocks"] is not None:
            data["blocks"] = tuple(data["blocks"])
        for key in ("hr_products", "lr_products"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        if isinstance(data.get("cellgrid"), dict):
            data["cellgrid"] = CellGridSplitConfig(**data["cellgrid"])
        data.update(overrides)
        return cls(**data)

    @property
    def patch_grid_spec(self) -> PatchGridSpec:
        return PatchGridSpec(
            patch_px=self.patch_px,
            stride_px=self.stride_px,
            min_valid_frac=self.min_valid_frac,
        )

    @property
    def cell_grid_spec(self) -> CellGridSpec:
        """The configured cell grid layered on `patch_grid_spec` (04b's sweep geometry)."""
        return CellGridSpec(
            patch=self.patch_grid_spec,
            cell_patches=self.cellgrid.cell_patches,
            buffer_patches=self.cellgrid.buffer_patches,
            min_cell_valid_frac=self.cellgrid.min_cell_valid_frac,
        )

    ###
    # File storage paths for dataset
    ###

    @property
    def hr_dir(self) -> Path:
        return self.root / "02_snap_owned" / "aeromagnetics" / "60m" / "cubicspline"

    @property
    def lr_dir(self) -> Path:
        return self.root / "02_snap_owned" / "aeromagnetics" / "180m" / "cubicspline"

    @property
    def inference_lr_dir(self) -> Path:
        return self.root / "03_inference_lr" / "cubicspline"

    @property
    def dem_path(self) -> Path:
        return self.root / "02_snap_owned" / "elevation" / "snapped_cubicspline_dem30m.tif"

    @property
    def ms_dir(self) -> Path:
        return self.root / "02_snap_owned" / "multispectral" / "cubicspline"

    def ms_band_path(self, band: int) -> Path:
        """Path to one snapped 30 m Landsat band (1..7)."""
        return self.ms_dir / f"snapped_landsat_b{band}.tif"

    @property
    def mask_path(self) -> Path:
        """Block-ID mask. Prefers ``<root>/00_base_grid/`` (a full dataset tree);
        falls back to the committed ``data/ksa_base_grids/`` so a from-scratch build
        works with no extra copy step (the shipped grids are the canonical KSA mask)."""
        in_root = self.root / "00_base_grid" / "magnetic_mask_grid60m.tif"
        if in_root.exists():
            return in_root
        return DATA_DIR / "ksa_base_grids" / "magnetic_mask_grid60m.tif"

    @property
    def patch_index_dir(self) -> Path:
        return DATA_DIR / "processed" / "ksa_aligned" / "patch_indices"

    @property
    def normalization_path(self) -> Path:
        return self.patch_index_dir / "normalization.json"

    def hr_source_product_path(self, product: str) -> Path:
        return self.hr_dir / f"snapped_cubicspline_MAG_{product}.tif"

    def hr_normalized_product_path(self, product: str) -> Path:
        return self.hr_dir / f"snapped_cubicspline_MAG_{product}_blockwise_unbiased.tif"

    def hr_product_path(self, product: str) -> Path:
        """HR raster path — prefers the `_blockwise_unbiased` per-block zero-mean version if present, else source."""
        normalized = self.hr_normalized_product_path(product)
        if normalized.exists():
            return normalized
        return self.hr_source_product_path(product)

    def lr_product_path(self, product: str) -> Path:
        stem = _LR_FILENAME_STEMS[product]
        return self.lr_dir / f"snapped_cubicspline_{stem}.tif"

    def inference_lr_product_path(self, product: str) -> Path:
        """Full-coverage LR raster (extends beyond HR-aligned regions) for inference-time reconstruction."""
        stem = _LR_FILENAME_STEMS[product]
        return self.inference_lr_dir / f"snapped_cubicspline_{stem}.tif"


_LR_FILENAME_STEMS: dict[str, str] = {
    "RTP": "RTP",
    "TMI": "TMI",
    "ANS": "RTP_ANSIG",
    "1VD": "RTP_FVD",
}

# Per-product (vmin, vmax) for normalizing an aux LR channel to [0, 1]
# global robust q0.5/q99.5 percentiles of each
_LR_AUX_ROBUST_RANGE: dict[str, tuple[float, float]] = {
    "1VD": (-0.327, 0.447),
    "ANS": (0.003, 0.685),
    "TMI": (-422.6, 334.7),
}

# Landsat band -> short name and global robust (q0.5, q99.5) reflectance range, used to
# normalize each pooled multispectral channel to [0, 1] independently.
_MS_BAND_NAMES: dict[int, str] = {
    1: "coastal",
    2: "blue",
    3: "green",
    4: "red",
    5: "nir",
    6: "swir1",
    7: "swir2",
}
_MS_BAND_ROBUST_RANGE: dict[int, tuple[float, float]] = {
    1: (0.068, 0.179),
    2: (0.075, 0.200),
    3: (0.090, 0.263),
    4: (0.099, 0.342),
    5: (0.114, 0.414),
    6: (0.119, 0.504),
    7: (0.109, 0.460),
}

# Derived band-ratio (alteration index) features: name -> (numerator_band, denominator_band,
# vmin, vmax). Ranges are the pooled ratio's global robust q0.5/q99.5.
_MS_FEATURE_SPEC: dict[str, tuple[int, int, float, float]] = {
    "ferrous": (6, 5, 0.927, 1.464),  # SWIR1/NIR — ferrous/alteration (best single channel)
    "clay": (6, 7, 0.963, 1.249),  # SWIR1/SWIR2 — clay/ferrous-alteration
    "ironoxide": (4, 2, 1.291, 2.380),  # red/blue — ferric iron-oxide
}


# ============================================================
# Mask + block helpers
# ============================================================
_SOURCE_ID_RE = re.compile(r"ksa_aligned/B(\d+)$")


def _read_block_mask(config: KSAAlignedConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the 60 m block-ID mask as a raw uint8 array.

    The mask encodes block IDs directly (0 = no data, 1/2/3 = survey blocks),
    so we need integer values, not the boolean `compute_mask` returns.
    """
    with rasterio.open(config.mask_path) as ds:
        arr = ds.read(1)
        meta = {
            "transform": ds.transform,
            "crs": ds.crs,
            "nodata": ds.nodata,
            "width": ds.width,
            "height": ds.height,
        }
    return arr, meta


def _present_blocks(arr: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(arr) if int(v) > 0)


def _read_lr_validity_mask(
    config: KSAAlignedConfig,
    product: str,
    hr_meta: dict[str, Any],
) -> np.ndarray:
    """Return the LR raster's bool validity mask upsampled to the 60 m HR grid.

    The aligned snapshot has exact integer pixel ratios (`lr_scale = 3`), so
    we replicate each LR pixel's validity into a 3x3 HR block via `np.repeat`
    and then crop/pad to the HR extent to absorb any off-by-one at the edge.
    """
    lr_path = config.lr_product_path(product)
    with rasterio.open(lr_path) as ds:
        lr = ds.read(1)
        lr_nodata = ds.nodata

    lr_valid = np.isfinite(lr)
    if lr_nodata is not None:
        lr_valid &= lr != lr_nodata

    scale = config.lr_scale
    upsampled = np.repeat(np.repeat(lr_valid, scale, axis=0), scale, axis=1)

    hr_h, hr_w = hr_meta["height"], hr_meta["width"]
    up_h, up_w = upsampled.shape
    if up_h >= hr_h and up_w >= hr_w:
        return upsampled[:hr_h, :hr_w]
    padded = np.zeros((hr_h, hr_w), dtype=bool)
    padded[:up_h, :up_w] = upsampled
    return padded


def _validate_spec(spec: PatchGridSpec, lr_scale: int) -> None:
    """Enforce divisibility so HR patch origins map to integer LR/DEM offsets."""
    if spec.patch_px % lr_scale != 0:
        lo = (spec.patch_px // lr_scale) * lr_scale
        raise ValueError(
            f"patch_px={spec.patch_px} must be a multiple of {lr_scale} "
            f"(LR scale). Nearest valid: {lo} or {lo + lr_scale}."
        )
    if spec.stride_px % lr_scale != 0:
        lo = (spec.stride_px // lr_scale) * lr_scale
        raise ValueError(
            f"stride_px={spec.stride_px} must be a multiple of {lr_scale} "
            f"(LR scale). Nearest valid: {lo} or {lo + lr_scale}."
        )


def iter_aligned_blocks(config: KSAAlignedConfig | None = None) -> list[int]:
    """Discover which SGS survey block IDs are present in the mask raster."""
    cfg = config or KSAAlignedConfig.default()
    arr, _ = _read_block_mask(cfg)
    present = _present_blocks(arr)
    if not present:
        raise FileNotFoundError(f"No nonzero block IDs in mask {cfg.mask_path}; expected 1/2/3.")
    return present


def build_aligned_block_patches(
    block: int,
    *,
    spec: PatchGridSpec,
    config: KSAAlignedConfig,
    lr_validity_product: str | None = ...,  # sentinel: use config default
) -> tuple[list[PatchWindow], np.ndarray, dict[str, Any]]:
    """Enumerate patches for one SGS block on the aligned grid.

    When `lr_validity_product` is set (default: `config.lr_validity_product`),
    the block mask is AND-ed with that LR product's validity before patches are
    enumerated, so every emitted patch has both HR and LR magnetic data. The
    returned mask reflects the combined validity.
    """
    _validate_spec(spec, config.lr_scale)
    lr_prod = config.lr_validity_product if lr_validity_product is ... else lr_validity_product
    arr, meta = _read_block_mask(config)
    block_mask = arr == block
    if not block_mask.any():
        raise ValueError(f"Block {block} not present in mask. Available: {_present_blocks(arr)}")
    if lr_prod:
        lr_valid = _read_lr_validity_mask(config, lr_prod, meta)
        effective = block_mask & lr_valid
    else:
        effective = block_mask
    patches = sliding_window_patches(effective, meta, spec, source_id=f"ksa_aligned/B{block}")
    return patches, effective, meta


# ============================================================
# Torch-style dataset
# ============================================================
class KSAShieldAlignedDataset:
    """Torch-compatible dataset serving HR/LR/DEM patch triples from one pre-snapped KSA snapshot.

    Returns samples shaped `{'hr': {...}, 'lr': {...}, 'dem': tensor, 'meta': {...}}`.
    Tensors are float32 `(H, W)` — no channel dim. `meta` includes UTM patch
    bounds, the block ID parsed from `source_id`, and a WGS84 patch-center
    `(lat, lon)` via a cached pyproj transformer.
    """

    def __init__(self, index: PatchIndex, *, config: KSAAlignedConfig):
        self.index = index
        self.config = config
        self.hr_products = list(config.hr_products)
        self.lr_products = list(config.lr_products)
        self.load_dem = config.load_dem

        self._hr: dict[str, RasterSource] = {
            p: open_raster(config.hr_product_path(p)) for p in self.hr_products
        }
        self._lr: dict[str, RasterSource] = {
            p: open_raster(config.lr_product_path(p)) for p in self.lr_products
        }
        self._dem: RasterSource | None = open_raster(config.dem_path) if config.load_dem else None
        self.lr_aux_products = list(config.lr_aux_products)
        missing = [p for p in self.lr_aux_products if p not in _LR_AUX_ROBUST_RANGE]
        if missing:
            raise ValueError(
                f"lr_aux_products {missing} have no entry in _LR_AUX_ROBUST_RANGE "
                f"(known: {sorted(_LR_AUX_ROBUST_RANGE)}). Add their global robust (vmin, vmax)."
            )
        self._lr_aux: dict[str, RasterSource] = {
            p: open_raster(config.lr_product_path(p)) for p in self.lr_aux_products
        }
        self.ms_bands = list(config.ms_bands)
        self.ms_features = list(config.ms_features)
        # Open the UNION of raw bands needed by direct bands + ratio features (sorted).
        feat_bands = {b for f in self.ms_features for b in _MS_FEATURE_SPEC[f][:2]}
        self._ms_order = sorted(set(self.ms_bands) | feat_bands)
        self._ms: dict[int, RasterSource] = {
            bi: open_raster(config.ms_band_path(bi)) for bi in self._ms_order
        }

        self.norm_stats: dict | None = (
            json.loads(config.normalization_path.read_text())
            if config.normalization_path.exists()
            else None
        )

        from pyproj import Transformer

        self._to_wgs84 = Transformer.from_crs(config.epsg, "EPSG:4326", always_xy=True)

    @classmethod
    def for_channels(cls, config: KSAAlignedConfig, *, product: str = "RTP") -> "KSAShieldAlignedDataset":
        """Patchless instance used purely for its channel-normalization methods.

        Reconstruction reuses `dem_features` / `lr_aux_to_channels` so inference channels are
        byte-identical to training; no patches are served (`len(ds) == 0`).
        """
        empty = PatchIndex(spec=config.patch_grid_spec, product=product, patches=[])
        return cls(empty, config=config)

    @staticmethod
    def _block_from_patch(t: PatchWindow) -> int:
        m = _SOURCE_ID_RE.match(t.source_id)
        if not m:
            raise ValueError(f"Unexpected source_id for KSA aligned patch: {t.source_id!r}")
        return int(m.group(1))

    def _bounds(
        self,
        blocks: list[int] | None,
        mode: str,
        *,
        batch_size: int,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        if self.norm_stats is None:
            raise RuntimeError(
                f"norm_stats is None — {self.config.normalization_path} not found. "
                "Run scripts/build_ksa_dataset/04_compute_ksa_aligned_normalization.py."
            )
        if mode == "global":
            g = self.norm_stats["global"]
            vmin = torch.full((batch_size, 1, 1, 1), float(g["vmin"]), device=device, dtype=dtype)
            vmax = torch.full((batch_size, 1, 1, 1), float(g["vmax"]), device=device, dtype=dtype)
            return vmin, vmax
        if mode == "blockwise":
            if blocks is None or len(blocks) != batch_size:
                raise ValueError(
                    f"mode='blockwise' needs a blocks list of length {batch_size}, got {blocks!r}"
                )
            bs = self.norm_stats["blocks"]
            vmin = torch.tensor([bs[str(b)]["vmin"] for b in blocks], device=device, dtype=dtype).view(
                -1, 1, 1, 1
            )
            vmax = torch.tensor([bs[str(b)]["vmax"] for b in blocks], device=device, dtype=dtype).view(
                -1, 1, 1, 1
            )
            return vmin, vmax
        raise ValueError(f"mode must be 'global' or 'blockwise', got {mode!r}")

    def normalize(
        self,
        tensor: "torch.Tensor",
        blocks: list[int] | None = None,
        mode: str | None = None,
        *,
        nan_fill: float | None = None,
    ) -> "torch.Tensor":
        """Clip+scale a `(B, C, H, W)` tensor to `[0, 1]` using `self.norm_stats`.

        `mode` defaults to `self.config.norm_mode`. Pass explicitly to override.

        Pass a `nan_fill` for model *inputs*: some patches carry NaN pixels
        `magsr.reconstruct.build.MAG_INPUT_NAN_FILL` indicates the fill value
        """
        m = mode if mode is not None else self.config.norm_mode
        vmin, vmax = self._bounds(
            blocks, m, batch_size=tensor.shape[0], device=tensor.device, dtype=tensor.dtype
        )
        out = clip_scale01(tensor, vmin, vmax)
        return out if nan_fill is None else out.nan_to_num(nan_fill)

    def denormalize(
        self,
        tensor: "torch.Tensor",
        blocks: list[int] | None = None,
        mode: str | None = None,
    ) -> "torch.Tensor":
        """Inverse of `normalize` — `[0, 1]` back to nT. `mode` defaults to `self.config.norm_mode`."""
        m = mode if mode is not None else self.config.norm_mode
        vmin, vmax = self._bounds(
            blocks, m, batch_size=tensor.shape[0], device=tensor.device, dtype=tensor.dtype
        )
        return denorm01(tensor, vmin, vmax)

    def _pool_dem(self, dem: "torch.Tensor", factor: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Avg-pool the raw 30 m DEM by `factor` over valid pixels.

        Returns `(pooled, valid)`: `pooled` is raw metres (0 where the cell had no valid
        30 m pixels), `valid` a bool mask. `factor=dem_scale*lr_scale` lands on the 180 m
        LR grid; `factor=dem_scale` lands on the 60 m HR grid (genuine sub-LR detail).
        """
        import torch
        import torch.nn.functional as F

        mask = torch.isfinite(dem)
        summed = F.avg_pool2d(torch.where(mask, dem, torch.zeros_like(dem)), factor)
        count = F.avg_pool2d(mask.to(dem.dtype), factor)
        return summed / count.clamp(min=1e-6), count > 0

    def _pool_dem_to_lr(self, dem: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Avg-pool the raw 30 m DEM onto the 180 m LR grid"""
        return self._pool_dem(dem, self.config.dem_scale * self.config.lr_scale)

    @staticmethod
    def _grad_xy(pooled: "torch.Tensor", valid: "torch.Tensor", c: float) -> "torch.Tensor":
        """Central-difference (dz/dx, dz/dy) of `pooled`, mapped to [0, 1] via ±`c` (0.5 = flat),
        neutral 0.5 on invalid cells. Returns `(B, 2, H, W)`; channel order (dz/dx, dz/dy)."""
        import torch
        import torch.nn.functional as F

        zp = F.pad(pooled, (1, 1, 1, 1), mode="replicate")
        gx = (zp[..., 1:-1, 2:] - zp[..., 1:-1, :-2]) * 0.5  # d/dx (cols, E-W)
        gy = (zp[..., 2:, 1:-1] - zp[..., :-2, 1:-1]) * 0.5  # d/dy (rows, N-S)
        gx = center_scale01(gx, c)
        gy = center_scale01(gy, c)
        neutral = torch.full_like(gx, 0.5)
        return torch.cat([torch.where(valid, gx, neutral), torch.where(valid, gy, neutral)], dim=1)

    def dem_features(self, dem: "torch.Tensor") -> "torch.Tensor":
        """DEM model-input channel(s), per `config.dem_mode`: "relief" (1ch) or "grad" (2ch)."""
        if self.config.dem_mode == "grad":
            return self.dem_grad_to_lr(dem)
        elif self.config.dem_mode == "relief":
            return self.dem_to_lr(dem)
        else:
            raise ValueError(f"Unknown dem_mode {self.config.dem_mode!r}")

    def dem_to_lr(self, dem: "torch.Tensor") -> "torch.Tensor":
        """Pool the raw 30 m DEM onto the 180 m LR grid as a mean-centred relief channel.

        The per-patch mean is subtracted (removing absolute elevation) and the result
        mapped to `[0, 1]` with `0.5 = patch mean` and `±dem_relief_m` metres
        Returns `(B, 1, H_lr, W_lr)`, empty 180 m cells get 0.5 (the patch mean).
        """
        import torch

        pooled, valid = self._pool_dem_to_lr(dem)
        valid_f = valid.to(pooled.dtype)
        patch_mean = (pooled * valid_f).sum(dim=(-2, -1), keepdim=True) / valid_f.sum(
            dim=(-2, -1), keepdim=True
        ).clamp(min=1.0)
        rel = center_scale01(pooled - patch_mean, self.config.dem_relief_m)
        return torch.where(valid, rel, torch.full_like(rel, 0.5))

    def dem_grad_to_lr(self, dem: "torch.Tensor") -> "torch.Tensor":
        """Pooled-DEM gradient as 2 channels (dz/dx, dz/dy) — slope magnitude + direction.

        Pools the 30 m DEM to the 180 m LR grid, then takes the central-difference
        spatial gradient (m per 180 m pixel) along x (E-W) and y (N-S). maps to `[0, 1]`

        Returns `(B, 2, H_lr, W_lr)`; Channel order: (dz/dx, dz/dy).
        """
        pooled, valid = self._pool_dem_to_lr(dem)
        return self._grad_xy(pooled, valid, self.config.dem_grad_clip)

    def dem_grad_to_hr(self, dem: "torch.Tensor") -> "torch.Tensor":
        """HR-native DEM gradient (2ch dz/dx, dz/dy) on the 60 m HR grid — `(B, 2, H_hr, W_hr)`."""
        pooled, valid = self._pool_dem(dem, self.config.dem_scale)
        return self._grad_xy(pooled, valid, self.config.dem_grad_clip / self.config.lr_scale)

    def lr_aux_to_channels(self, aux: "torch.Tensor") -> "torch.Tensor":
        """Normalize stacked raw LR aux products to `[0, 1]` model input channels.

        `aux` is `(B, C_aux, H_lr, W_lr)` raw values with NaN for missing pixels, channel
        order = `self.lr_aux_products` (as stacked by `pool_collate`). `_LR_AUX_ROBUST_RANGE`
        holds per channel (global robust q0.5/q99.5 of that LR raster) (vmin, vmax)
        """
        import torch

        chans = []
        for i, prod in enumerate(self.lr_aux_products):
            vmin, vmax = _LR_AUX_ROBUST_RANGE[prod]
            chans.append(clip_scale01(aux[:, i : i + 1], vmin, vmax))
        return torch.cat(chans, dim=1).nan_to_num(0.5)

    def ms_to_channels(self, ms: "torch.Tensor") -> "torch.Tensor":
        """Pool stacked raw 30 m Landsat bands to the 180 m LR grid -> normalized [0, 1] channels."""
        import torch
        import torch.nn.functional as F

        f = self.config.dem_scale * self.config.lr_scale
        mask = torch.isfinite(ms)
        summed = F.avg_pool2d(torch.where(mask, ms, torch.zeros_like(ms)), f)
        count = F.avg_pool2d(mask.to(ms.dtype), f)
        pooled = summed / count.clamp(min=1e-6)  # (B, n, H_lr, W_lr) reflectance, order = _ms_order
        pos = {b: i for i, b in enumerate(self._ms_order)}
        chans = []
        for bi in self.ms_bands:  # direct bands
            vmin, vmax = _MS_BAND_ROBUST_RANGE[bi]
            chans.append(clip_scale01(pooled[:, pos[bi] : pos[bi] + 1], vmin, vmax))
        for name in self.ms_features:  # ratio features
            nb_, db_, vmin, vmax = _MS_FEATURE_SPEC[name]
            r = pooled[:, pos[nb_] : pos[nb_] + 1] / (pooled[:, pos[db_] : pos[db_] + 1] + 1e-3)
            chans.append(clip_scale01(r, vmin, vmax))
        return torch.cat(chans, dim=1).nan_to_num(0.5)

    def assemble_lr_input(self, lr, blocks, *, dem=None, lr_aux=None, ms=None, nan_fill=0.0):
        """Normalize the raw LR mag tensor and cat the extra input channels in training order
        (mag, DEM, LR-aux, MS). A channel group is included iff its tensor is passed (caller-guarded).
        HR is never touched here."""
        x = self.normalize(lr, blocks=blocks, nan_fill=nan_fill)
        if dem is not None:
            x = torch.cat([x, self.dem_features(dem)], dim=1)
        if lr_aux is not None:
            x = torch.cat([x, self.lr_aux_to_channels(lr_aux)], dim=1)
        if ms is not None:
            x = torch.cat([x, self.ms_to_channels(ms)], dim=1)
        return x

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        import torch

        t = self.index.patches[i]
        block = self._block_from_patch(t)
        hr_px = self.index.spec.patch_px
        lr_px = hr_px // self.config.lr_scale
        dem_px = hr_px * self.config.dem_scale

        lr_row = t.row_px // self.config.lr_scale
        lr_col = t.col_px // self.config.lr_scale
        dem_row = t.row_px * self.config.dem_scale
        dem_col = t.col_px * self.config.dem_scale

        hr: dict[str, torch.Tensor] = {}
        for prod in self.hr_products:
            arr = self._hr[prod].read_window(t.row_px, t.col_px, hr_px, hr_px)
            hr[prod] = torch.from_numpy(arr)

        lr: dict[str, torch.Tensor] = {}
        for prod in self.lr_products:
            arr = self._lr[prod].read_window(lr_row, lr_col, lr_px, lr_px)
            lr[prod] = torch.from_numpy(arr)

        sample: dict[str, Any] = {"hr": hr, "lr": lr}

        if self._dem is not None:
            arr = self._dem.read_window(dem_row, dem_col, dem_px, dem_px)
            sample["dem"] = torch.from_numpy(arr)

        if self._lr_aux:
            # Aux products share the 180 m LR grid, so the same LR window as the
            # magnetic channel. Insertion order follows config.lr_aux_products.
            sample["lr_aux"] = {
                p: torch.from_numpy(src.read_window(lr_row, lr_col, lr_px, lr_px))
                for p, src in self._lr_aux.items()
            }

        if self._ms:
            # Landsat bands share the 30 m DEM grid, so the same window as DEM.
            sample["ms"] = {
                bi: torch.from_numpy(src.read_window(dem_row, dem_col, dem_px, dem_px))
                for bi, src in self._ms.items()
            }

        cx = 0.5 * (t.left + t.right)
        cy = 0.5 * (t.top + t.bottom)
        lon, lat = self._to_wgs84.transform(cx, cy)

        sample["meta"] = {
            "source_id": t.source_id,
            "block": block,
            "row_px": t.row_px,
            "col_px": t.col_px,
            "left": t.left,
            "bottom": t.bottom,
            "right": t.right,
            "top": t.top,
            "lat": float(lat),
            "lon": float(lon),
            "valid_frac": t.valid_frac,
            "name": t.name,
        }
        return sample


# ============================================================
# Convenience factory: train/val/test splits
# ============================================================
def build_ksa_aligned_datasets(
    cfg: KSAAlignedConfig | None = None,
    *,
    index_dir: Path,
    **overrides: Any,
) -> dict[str, KSAShieldAlignedDataset]:
    """Load deterministic train/val/test splits from pre-baked PatchIndex JSONs.

    Expects `scripts/build_ksa_dataset/06_assign_cell_splits.py` to have written
    `train.json`, `val.json`, `test.json` under `index_dir`. Each file is a single
    `PatchIndex` whose patches span all blocks for that split. No random splitting
    happens at load time — the partition is whatever the build script froze.

    Args:
        cfg: `KSAAlignedConfig` to use. Defaults to
            `KSAAlignedConfig.from_yaml(**overrides)` when `None`; when a config
            is passed, `overrides` are applied on top.
        index_dir: Directory holding the split JSONs, e.g.
            `.../patch_indices_cellgrid8_fold3`. Required — there is no
            default split.
    """
    if cfg is None:
        cfg = KSAAlignedConfig.from_yaml(**overrides)
    elif overrides:
        cfg = replace(cfg, **overrides)

    splits: dict[str, KSAShieldAlignedDataset] = {}
    missing: list[Path] = []
    for split in ("train", "val", "test"):
        path = index_dir / f"{split}.json"
        if not path.exists():
            missing.append(path)
            continue
        splits[split] = KSAShieldAlignedDataset(PatchIndex.load(path), config=cfg)
    if missing:
        joined = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Missing KSA aligned split JSON(s):\n  "
            f"{joined}\n"
            "Run scripts/build_ksa_dataset/06_assign_cell_splits.py to generate them."
        )
    return splits

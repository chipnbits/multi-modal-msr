"""Run the patch sweep: read LR once, infer + blend, denormalize once."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry

from magsr.normalize import Normalizer
from magsr.reconstruct.blending import BlendKind, make_blend_weight

""" Patch planning for reconstruction """

# Define a type alia for (xmin, ymin, xmax, ymax) identifying a bounding box
WorldBounds = tuple[float, float, float, float]

# NaN fill for the magnetic model-input channel at reconstruction time. Kept equal to the
# training loop, which fills LR-input NaNs with 0.0 before the model
MAG_INPUT_NAN_FILL = 0.0


def _read_masked(
    ds: "rasterio.DatasetReader",
    *,
    window: Window,
    out_shape: tuple[int, int] | None = None,
    boundless: bool = False,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Read band 1 over `window` as float32 with the raster's nodata mapped to NaN.

    Centralizes the `read -> (value == nodata) -> NaN` pattern repeated across the
    reconstruction reads. `out_shape` / `boundless` (+ `fill_value`) forward to `ds.read`.
    """
    kwargs: dict[str, Any] = {"window": window}
    if out_shape is not None:
        kwargs["out_shape"] = out_shape
    if boundless:
        kwargs["boundless"] = True
        kwargs["fill_value"] = fill_value
    arr = ds.read(1, **kwargs).astype(np.float32, copy=False)
    nodata = ds.nodata
    if nodata is not None and not np.isnan(nodata):
        arr = np.where(arr == nodata, np.float32(np.nan), arr)
    return arr


def _resolve_device(model: torch.nn.Module, device: "torch.device | None") -> "torch.device":
    """Device of `model`'s first parameter, or CPU for parameter-free models (e.g. bicubic)."""
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@dataclass(frozen=True)
class PatchPlan:
    """
    A data container to store a plan for patchifying a region of interest from a LR raster.

    Args:
        lr_window (Window): Window across the whole region of interest, including buffer
        lr_window_transform (Affine): Transform from pixel coords to world coords
        patch_lr_px (int): pixel size of the patches for reconstruction
        stride_lr_px (int): pixel stride between adjacent patches
        scale (int): super-resolution scale factor for HR patch size ratio
        positions (list[tuple[int, int]]): list of (row, col) top-left coords of patches
        crs (CRS): coordinate reference system of the LR raster and world coords
    """

    lr_window: Window
    lr_window_transform: Affine
    patch_lr_px: int
    stride_lr_px: int
    scale: int
    positions: list[tuple[int, int]]
    crs: CRS

    @property
    def patch_hr_px(self) -> int:
        return self.patch_lr_px * self.scale

    @property
    def stride_hr_px(self) -> int:
        return self.stride_lr_px * self.scale

    @property
    def lr_size(self) -> tuple[int, int]:
        return int(self.lr_window.height), int(self.lr_window.width)

    @property
    def hr_size(self) -> tuple[int, int]:
        h, w = self.lr_size
        return h * self.scale, w * self.scale

    @property
    def hr_window_transform(self) -> Affine:
        t = self.lr_window_transform
        return Affine(t.a / self.scale, t.b, t.c, t.d, t.e / self.scale, t.f)

    @property
    def world_bbox(self) -> WorldBounds:
        """LR-window world bounds as `(left, bottom, right, top)` — for `from_bounds` reads."""
        t = self.lr_window_transform
        left = t.c
        right = t.c + self.lr_window.width * t.a
        top = t.f
        bottom = t.f + self.lr_window.height * t.e
        return left, bottom, right, top

    @property
    def world_extent(self) -> list[float]:
        """LR-window extent as `[left, right, bottom, top]` — for `imshow(..., extent=...)`."""
        left, bottom, right, top = self.world_bbox
        return [left, right, bottom, top]


def _polygon_bounds(polygon: WorldBounds | Any) -> WorldBounds:
    """Get (minx, miny, maxx, maxy) bounds from a polygon object or a tuple."""
    if isinstance(polygon, tuple) and len(polygon) == 4:
        return tuple(float(v) for v in polygon)
    bounds = getattr(polygon, "bounds", None)
    if bounds is None or len(bounds) != 4:
        raise TypeError(
            f"polygon_world must be (minx, miny, maxx, maxy) or expose .bounds; got {polygon!r}"
        )
    return tuple(float(v) for v in bounds)


def _patch_positions(extent: int, patch: int, stride: int) -> list[int]:
    """Top-left positions for `patch`-sized windows covering `[0, extent]`."""
    if extent < patch:
        raise ValueError(f"extent {extent} smaller than patch {patch}")
    if stride <= 0:
        raise ValueError(f"stride {stride} must be positive")
    positions = list(range(0, extent - patch + 1, stride))
    last = extent - patch
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def plan_lr_patches(
    *,
    polygon_world: WorldBounds | Any,
    lr_path: Path | str,
    patch_lr_px: int,
    stride_lr_px: int,
    buffer_lr_px: int | None = None,
    scale: int = 3,
    min_valid_frac: float = 0.0,
) -> PatchPlan:
    """
    Create a PatchPlan to apply super-resolution reconstruction over a region of interest

    The plan takes a set of closed bounds and a raster and determines an appropriate sequence
    of patches to cover over the region with buffer at boundary. The patches are aligned to
    pixels on a grid.

    Args:
        polygon_world (WorldBounds | Any): (xmin, ymin, xmax, ymax) bounds, an object with .bounds,
            or a shapely geometry. When a shapely geometry is passed, patches whose bbox does not
            intersect the geometry are dropped (in addition to the bbox-based window planning).
        lr_path (Path | str): path to the LR raster to derive the grid and metadata from
        patch_lr_px (int): pixel size of the patches for reconstruction
        stride_lr_px (int): pixel stride between adjacent patches
        buffer_lr_px (int | None): optional buffer size in pixels to add around the region; if None,
            defaults to patch_lr_px. Pass 0 explicitly when the AOI already includes a buffer.
        scale (int): super-resolution scale factor for HR patch size ratio
        min_valid_frac (float): drop patches whose LR window has fewer non-NaN pixels than this
            fraction. 0.0 (default) keeps every position.
    """
    buffer_lr_px = buffer_lr_px if buffer_lr_px is not None else patch_lr_px
    minx, miny, maxx, maxy = _polygon_bounds(polygon_world)
    aoi_geom = polygon_world if isinstance(polygon_world, BaseGeometry) else None

    with rasterio.open(lr_path) as src:
        raw: Window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        padded = (
            Window(
                raw.col_off - buffer_lr_px,
                raw.row_off - buffer_lr_px,
                raw.width + 2 * buffer_lr_px,
                raw.height + 2 * buffer_lr_px,
            )
            .round_offsets()
            .round_lengths()
        )
        final_window = padded.intersection(Window(0, 0, src.width, src.height))
        window_transform = src.window_transform(final_window)
        lr_crs = src.crs

        rows_pos = _patch_positions(int(final_window.height), patch_lr_px, stride_lr_px)
        cols_pos = _patch_positions(int(final_window.width), patch_lr_px, stride_lr_px)
        positions = [(r, c) for r in rows_pos for c in cols_pos]

        if aoi_geom is not None:
            kept: list[tuple[int, int]] = []
            for r, c in positions:
                left, top = window_transform * (c, r)
                right, bottom = window_transform * (c + patch_lr_px, r + patch_lr_px)
                patch_box = shapely_box(
                    min(left, right), min(top, bottom), max(left, right), max(top, bottom)
                )
                if aoi_geom.intersects(patch_box):
                    kept.append((r, c))
            positions = kept

        if min_valid_frac > 0.0 and positions:
            lr_arr = _read_masked(src, window=final_window)
            valid = np.isfinite(lr_arr)
            positions = [
                (r, c)
                for r, c in positions
                if valid[r : r + patch_lr_px, c : c + patch_lr_px].mean() >= min_valid_frac
            ]

    return PatchPlan(
        lr_window=final_window,
        lr_window_transform=window_transform,
        patch_lr_px=patch_lr_px,
        stride_lr_px=stride_lr_px,
        scale=scale,
        positions=positions,
        crs=lr_crs,
    )


""" Reconstruction run """


def infer_patch(
    model: torch.nn.Module,
    lr_patch: np.ndarray | torch.Tensor,
    *,
    nan_fill: float = 0.5,
    device: torch.device | None = None,
    block_id: int | None = None,
    block_ids: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Run the model on `(h, w)`, `(B, h, w)`, or `(B, 1, h, w)`; returns SR matching the leading shape.

    For multi-domain models (`num_domains > 1`), pass a domain tag either as
    `block_id` (scalar 1-indexed block number; every patch in the batch tagged the
    same) OR `block_ids` (array/list of length B, one 1-indexed block per patch).
    `block_ids` takes precedence over `block_id` when both are given.
    """
    device = _resolve_device(model, device)

    if isinstance(lr_patch, torch.Tensor):
        x = lr_patch.to(device=device, dtype=torch.float32).clone()
    else:
        x = torch.from_numpy(np.asarray(lr_patch, dtype=np.float32)).to(device)

    nan_mask = torch.isnan(x)
    if nan_mask.any():
        x[nan_mask] = nan_fill

    squeeze_batch = False
    if x.ndim == 2:
        x = x[None, None]
        squeeze_batch = True
    elif x.ndim == 3:
        x = x[:, None]
    elif x.ndim != 4:
        raise ValueError(f"lr_patch must be 2/3/4-D, got {tuple(x.shape)}")

    with torch.no_grad():
        if getattr(model, "num_domains", 1) > 1:
            if block_ids is not None:
                dom = torch.as_tensor(block_ids, dtype=torch.long, device=device)
                if dom.shape[0] != x.shape[0]:
                    raise ValueError(f"block_ids length {dom.shape[0]} != batch size {x.shape[0]}")
            elif block_id is not None:
                dom = torch.full((x.shape[0],), int(block_id), dtype=torch.long, device=device)
            else:
                raise ValueError("block_id or block_ids required for multi-domain model (num_domains > 1)")
            sr = model(x, dom).clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
        else:
            sr = model(x).clamp(0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)

    sr = sr[:, 0]
    if squeeze_batch:
        sr = sr[0]
    return sr


def _read_lr_norm(lr_path: Path | str, plan: PatchPlan, normalizer: Normalizer) -> np.ndarray:
    """Read the buffered LR window (nodata -> NaN) and clip+scale it to [0, 1] (NaN preserved)."""
    with rasterio.open(lr_path) as lr_src:
        lr_arr = _read_masked(lr_src, window=plan.lr_window)
    return normalizer.normalize(lr_arr)


def _sweep_and_blend(
    model: torch.nn.Module,
    stack: np.ndarray,
    plan: PatchPlan,
    *,
    blend_kind: BlendKind,
    batch_size: int,
    device: torch.device | None,
    nan_fill: float,
    block_id: int | None,
    block_ids: np.ndarray | None,
) -> np.ndarray:
    """Sweep a normalized `(C, H_lr, W_lr)` `stack` through the model patchwise and blend.

    Shared core of `reconstruct_region` (C=1) and `reconstruct_region_multichannel` (C>1).
    Returns the blended SR in normalized [0, 1] with NaN outside covered pixels (not yet
    denormalized). `stack` still carries NaN in the magnetic channel; `infer_patch` fills it.
    """
    hr_h, hr_w = plan.hr_size
    value_sum = np.zeros((hr_h, hr_w), dtype=np.float32)
    weight_sum = np.zeros((hr_h, hr_w), dtype=np.float32)
    weight = make_blend_weight(plan.patch_hr_px, plan.stride_hr_px, kind=blend_kind)

    patch_lr = plan.patch_lr_px
    patch_hr = plan.patch_hr_px
    scale = plan.scale

    for start in range(0, len(plan.positions), batch_size):
        chunk = plan.positions[start : start + batch_size]
        batch = np.stack(
            [stack[:, r : r + patch_lr, c : c + patch_lr] for r, c in chunk],
            axis=0,
        )  # (B, C, h, w)
        chunk_blocks = block_ids[start : start + batch_size] if block_ids is not None else None
        sr_batch = infer_patch(
            model,
            batch,
            nan_fill=nan_fill,
            device=device,
            block_id=block_id,
            block_ids=chunk_blocks,
        )
        for (r_lr, c_lr), sr in zip(chunk, sr_batch):
            r_hr, c_hr = r_lr * scale, c_lr * scale
            value_sum[r_hr : r_hr + patch_hr, c_hr : c_hr + patch_hr] += sr * weight
            weight_sum[r_hr : r_hr + patch_hr, c_hr : c_hr + patch_hr] += weight

    valid = weight_sum > 0
    out_norm = np.full_like(value_sum, np.float32(np.nan))
    out_norm[valid] = value_sum[valid] / weight_sum[valid]
    return out_norm


def reconstruct_region(
    model: torch.nn.Module,
    lr_path: Path | str,
    plan: PatchPlan,
    *,
    normalizer: Normalizer,
    blend_kind: BlendKind = "auto",
    batch_size: int = 8,
    device: torch.device | None = None,
    nan_fill: float = MAG_INPUT_NAN_FILL,
    block_id: int | None = None,
) -> np.ndarray:
    """Read the buffered LR window, sweep + blend SR patches, return denormalised HR float32 (nT).

    `block_id` (1-indexed block number) selects the domain for multi-domain models.
    """
    device = _resolve_device(model, device)
    lr_norm = _read_lr_norm(lr_path, plan, normalizer)  # (H_lr, W_lr), carries NaN
    out_norm = _sweep_and_blend(
        model,
        lr_norm[None],  # (1, H_lr, W_lr)
        plan,
        blend_kind=blend_kind,
        batch_size=batch_size,
        device=device,
        nan_fill=nan_fill,
        block_id=block_id,
        block_ids=None,
    )
    return normalizer.denormalize(out_norm)


def build_aux_lr_channels(
    plan: PatchPlan,
    *,
    dataset: Any,
    lr_aux_paths: dict[str, Path | str] | None = None,
    dem_path: Path | str | None = None,
) -> np.ndarray | None:
    """Build the extra normalized model-input channels on the LR grid for a multichannel model.

    Reuses the dataset's OWN normalization methods (`dem_features`, `lr_aux_to_channels`) so the
    inference channels are byte-identical to training. Channel order matches the training
    `torch.cat` sequence in `ksa_aligned_rdn_train.py`: DEM channels first (when `use_dem`),
    then the LR aux channels in `dataset.lr_aux_products` order. The magnetic channel is NOT
    included here — `reconstruct_region_multichannel` prepends it.

    Returns `(C_aux, H_lr, W_lr)` float32 in [0, 1], or `None` when the model uses no extra
    channels. Reads over exactly `plan.lr_window` (the same buffered window mag is read from).
    """
    import torch

    if getattr(dataset, "ms_bands", None) or getattr(dataset, "ms_features", None):
        raise NotImplementedError(
            "build_aux_lr_channels does not build multispectral (ms) channels; a model trained "
            "with ms_bands/ms_features cannot be reconstructed through this path."
        )

    lr_h, lr_w = plan.lr_size
    left, bottom, right, top = plan.world_bbox
    chans: list[np.ndarray] = []

    # DEM channels first (matches training: cat([mag, dem_features, lr_aux, ...])).
    if getattr(dataset, "load_dem", False) and dem_path is not None:
        dem_factor = dataset.config.dem_scale * dataset.config.lr_scale  # 30 m -> 180 m
        dem_h, dem_w = lr_h * dem_factor, lr_w * dem_factor
        with rasterio.open(dem_path) as dem_src:
            dem_win = from_bounds(left, bottom, right, top, transform=dem_src.transform)
            dem_arr = _read_masked(
                dem_src, window=dem_win, out_shape=(dem_h, dem_w), boundless=True, fill_value=np.nan
            )
        # (1, 1, H_dem, W_dem) — matches the collated batch["dem"] layout dem_features expects.
        dem_t = torch.from_numpy(dem_arr)[None, None]
        dem_feats = dataset.dem_features(dem_t)  # (1, C_dem, H_lr, W_lr) in [0, 1]
        chans.append(dem_feats[0].cpu().numpy().astype(np.float32, copy=False))

    # LR aux channels (e.g. 1VD), in dataset.lr_aux_products order.
    aux_products = list(getattr(dataset, "lr_aux_products", []))
    if aux_products:
        if lr_aux_paths is None:
            raise ValueError("dataset has lr_aux_products but lr_aux_paths is None")
        raw = []
        for prod in aux_products:
            with rasterio.open(lr_aux_paths[prod]) as src:
                arr = _read_masked(src, window=plan.lr_window)
            raw.append(arr)
        aux_t = torch.from_numpy(np.stack(raw, axis=0))[None]  # (1, C_aux, H_lr, W_lr)
        aux_chans = dataset.lr_aux_to_channels(aux_t)  # (1, C_aux, H_lr, W_lr) in [0, 1]
        chans.append(aux_chans[0].cpu().numpy().astype(np.float32, copy=False))

    if not chans:
        return None
    return np.concatenate(chans, axis=0)


def assign_patch_blocks(
    plan: PatchPlan,
    block_shp_paths: dict[int, Path | str],
) -> np.ndarray:
    """Assign a 1-indexed survey block to every patch in `plan` by geographic location.

    For each LR patch, the patch CENTER world coordinate is computed from the plan's
    LR-window transform. The block is chosen by point-in-polygon against the reprojected
    block polygons (to the plan CRS). Centers inside no block are assigned the NEAREST
    block (min distance to the 3 polygons). Returns `(N_patches,)` int array of block ids
    matching the order of `plan.positions`.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    t = plan.lr_window_transform
    half = plan.patch_lr_px / 2.0
    centers = [Point(*(t * (c + half, r + half))) for r, c in plan.positions]

    # Reproject each block polygon to the plan CRS; keep a single (unioned) geometry per block.
    block_ids = sorted(block_shp_paths)
    geoms: dict[int, BaseGeometry] = {}
    for bid in block_ids:
        gdf = gpd.read_file(block_shp_paths[bid]).to_crs(plan.crs)
        geoms[bid] = gdf.geometry.union_all()

    out = np.empty(len(centers), dtype=np.int64)
    for i, pt in enumerate(centers):
        inside = [bid for bid in block_ids if geoms[bid].contains(pt)]
        if inside:
            out[i] = inside[0]
        else:
            out[i] = min(block_ids, key=lambda bid: geoms[bid].distance(pt))
    return out


def reconstruct_region_multichannel(
    model: torch.nn.Module,
    lr_path: Path | str,
    plan: PatchPlan,
    *,
    normalizer: Normalizer,
    aux_channels: np.ndarray | None,
    blend_kind: BlendKind = "auto",
    batch_size: int = 8,
    device: torch.device | None = None,
    nan_fill: float = MAG_INPUT_NAN_FILL,
    block_id: int | None = None,
    block_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Multichannel variant of `reconstruct_region`.

    Reads the mag LR window, normalizes it, stacks `aux_channels` (pre-built, normalized,
    on the SAME LR grid as `plan.lr_window`) below it, sweeps `(B, C, h, w)` patches through
    the model, blends, and denormalizes the single-channel SR output with the mag `normalizer`.

    `aux_channels` is `(C_aux, H_lr, W_lr)` in [0, 1] (or `None`, in which case this is
    identical to `reconstruct_region`). Channel order of the stacked input is
    `[mag, *aux_channels]`, matching how training cats `[mag, dem_features, lr_aux_channels]`.

    Domain tagging (multi-domain models): pass a scalar `block_id` for the whole plan, or
    `block_ids` (length `len(plan.positions)`, one 1-indexed block per patch) for per-patch
    tagging. `block_ids` takes precedence.
    """
    if block_ids is not None and len(block_ids) != len(plan.positions):
        raise ValueError(f"block_ids length {len(block_ids)} != number of patches {len(plan.positions)}")
    if aux_channels is None:
        return reconstruct_region(
            model,
            lr_path,
            plan,
            normalizer=normalizer,
            blend_kind=blend_kind,
            batch_size=batch_size,
            device=device,
            nan_fill=nan_fill,
            block_id=block_id,
        )

    device = _resolve_device(model, device)
    lr_norm = _read_lr_norm(lr_path, plan, normalizer)  # (H_lr, W_lr), still carries NaN

    lr_h, lr_w = plan.lr_size
    if aux_channels.shape[1:] != (lr_h, lr_w):
        raise ValueError(f"aux_channels grid {aux_channels.shape[1:]} != LR window grid {(lr_h, lr_w)}")
    # Mag on top, aux below — order matches training cat([mag, *aux]).
    stack = np.concatenate([lr_norm[None], aux_channels], axis=0)  # (C, H_lr, W_lr)

    out_norm = _sweep_and_blend(
        model,
        stack,
        plan,
        blend_kind=blend_kind,
        batch_size=batch_size,
        device=device,
        nan_fill=nan_fill,
        block_id=block_id,
        block_ids=block_ids,
    )
    return normalizer.denormalize(out_norm)


""" IO Helpers for reconstruciton """


def load_rect_json(path: Path | str) -> WorldBounds:
    """
    Read a rect JSON written → `(x0, y0, x1, y1)`.

    This can be used to read for example the rectangular polygon bounds for
    the holdout splits from KSA Aligned.
    """
    data = json.loads(Path(path).read_text())
    r = data["rect"]
    return float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])


def write_reconstruction_geotiff(arr: np.ndarray, plan: PatchPlan, out_path: Path | str) -> None:
    profile = {
        "driver": "GTiff",
        "height": plan.hr_size[0],
        "width": plan.hr_size[1],
        "count": 1,
        "dtype": "float32",
        "crs": plan.crs,
        "transform": plan.hr_window_transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32, copy=False), 1)

"""Patch primitives + patch-construction operations for slicing patches from rasters.

Types (`PatchGridSpec`, `PatchWindow`, `CellGridSpec`, `PatchIndex`) and the
operations that build them from a raster (`compute_mask`,
`sliding_window_patches`, `save_mask_geotiff`, `bucket_patches_by_cell`,
`assign_cell_splits`) live together.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import box as shapely_box

from magsr.datasets.io import RasterSource


@dataclass(frozen=True)
class PatchGridSpec:
    """Sliding-window sweep parameters for enumerating patches over a raster or cell.

    128x128 at 64 stride = half-overlapping patches, etc.

    Attributes:
        patch_px (int): Patch side length, in pixels (patches are square).
        stride_px (int): Step between consecutive patch origins, in pixels;
            stride_px < patch_px gives overlapping patches.
        min_valid_frac (float): Minimum fraction of valid (non-nodata) pixels
            a patch must contain to be kept, in [0, 1].
    """

    patch_px: int
    stride_px: int
    min_valid_frac: float = 0.95

    def stride_m(self, pixel_size_m: float) -> float:
        """Stride in world units (m), given the raster's pixel size."""
        return self.stride_px * pixel_size_m

    def patch_m(self, pixel_size_m: float) -> float:
        """Patch side length in world units (m), given the raster's pixel size."""
        return self.patch_px * pixel_size_m


@dataclass(frozen=True)
class PatchWindow:
    """One HR patch location in pixel + world coordinates.

    Two coordinate spaces, told apart by suffix: `row_px`/`col_px` are pixel
    space; bare `left`/`bottom`/`right`/`top` are world coordinates in the
    CRS given by `epsg` (meters for UTM rasters), following the
    (left, bottom, right, top) bounds convention of rasterio/shapely.

    Attributes:
        source_id (str): Free-form identifier of the source raster / sub-region
            (e.g. "wa/Train", "ksa_aligned/B2"); lets one flat list mix patches
            from several (dataset, sub-region) combinations.
        row_px (int): Pixel row offset of the patch's top-left corner from
            raster origin — always a multiple of the sweep stride, not a
            sequential patch index.
        col_px (int): Pixel column offset of the patch's top-left corner from
            raster origin.
        left (float): Western world x-coordinate (smaller x).
        bottom (float): Southern world y-coordinate (numerically smaller y).
        right (float): Eastern world x-coordinate (larger x).
        top (float): Northern world y-coordinate (numerically larger y).
        valid_frac (float): Fraction of valid (non-nodata) pixels in the patch, in [0, 1].
        epsg (str | None): EPSG code of the world coordinate reference system, or None if unknown.
    """

    source_id: str
    row_px: int
    col_px: int
    left: float
    bottom: float
    right: float
    top: float
    valid_frac: float
    epsg: str | None = None

    @classmethod
    def from_pixel(
        cls,
        *,
        source_id: str,
        row_px: int,
        col_px: int,
        transform: Affine,
        patch_px: int,
        valid_frac: float,
        epsg: str | None = None,
    ) -> "PatchWindow":
        """Build a PatchWindow from a pixel offset, georeferenced via `transform`.

        `transform` is the raster's `affine.Affine` (rasterio's
        `dataset.transform`, carried here in `meta["transform"]`), mapping
        pixel space to world space. Of its six coefficients (a, b, c, d, e, f):
        `a` is the x pixel size (positive), `e` the y pixel size (negative for
        north-up rasters), and (`c`, `f`) the world coordinates of the
        raster's top-left corner; `b`/`d` (rotation/shear) are assumed zero.
        Bottom thus ends up numerically smaller than top, as expected in world
        coordinates. `patch_px` is the window side length in pixels — cells
        pass a larger value than base patches.
        """
        left = transform.c + col_px * transform.a
        top = transform.f + row_px * transform.e
        right = left + patch_px * transform.a
        bottom = top + patch_px * transform.e
        return cls(
            source_id=source_id,
            row_px=row_px,
            col_px=col_px,
            left=float(left),
            bottom=float(bottom),
            right=float(right),
            top=float(top),
            valid_frac=float(valid_frac),
            epsg=epsg,
        )

    @property
    def name(self) -> str:
        """Unique, filesystem-safe identifier: source_id slug + pixel origin."""
        slug = self.source_id.replace("/", "_")
        return f"{slug}_r{self.row_px:05d}_c{self.col_px:05d}"

    @property
    def width(self) -> float:
        """East-west extent in world units (m for UTM CRSs)."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """North-south extent in world units (m for UTM CRSs)."""
        return self.top - self.bottom


@dataclass(frozen=True)
class CellGridSpec:
    """Mid-scale dicing parameters layered on a base PatchGridSpec.

    Cell origins sit on a regular pitch grid aligned to pixel (0, 0):

        cell_px  = cell_patches * patch.patch_px             # cell interior
        pitch_px = cell_px + buffer_patches * patch.patch_px

    Trailing partial cells at the raster edge are kept by sweeping with
    `allow_partial=True` — a full-size cell window rarely fits flush
    against the raster bottom/right, and dropping the partials loses every
    base patch in that band (the bottom of B3 spans ~10 patch rows there).

    Attributes:
        patch (PatchGridSpec): Base patch sweep enumerated inside each cell.
        cell_patches (int): Cell side length, in base-patch lengths
            (Hedgementation uses 12x12 cells).
        buffer_patches (int): Width of the strip of dropped ground *between*
            adjacent cells, in base-patch lengths — what keeps patches of
            different splits from ever sharing pixels.
        min_cell_valid_frac (float): Minimum valid-pixel fraction for a cell
            to be kept, in [0, 1]; 0.0 keeps every lattice position.
    """

    patch: PatchGridSpec
    cell_patches: int = 12
    buffer_patches: int = 1
    min_cell_valid_frac: float = 0.0

    @property
    def cell_px(self) -> int:
        """Cell interior side length, in pixels."""
        return self.cell_patches * self.patch.patch_px

    @property
    def pitch_px(self) -> int:
        """Distance between adjacent cell origins (interior + buffer), in pixels."""
        return self.cell_px + self.buffer_patches * self.patch.patch_px

    def as_cell_sweep(self) -> PatchGridSpec:
        """The cell grid expressed as a plain patch sweep (patch = cell
        interior, stride = pitch), so `sliding_window_patches` enumerates
        cells with no new traversal code. A cell is just a bigger
        PatchWindow."""
        return PatchGridSpec(
            patch_px=self.cell_px,
            stride_px=self.pitch_px,
            min_valid_frac=self.min_cell_valid_frac,
        )

    def cell_of(self, row_px: int, col_px: int) -> tuple[int, int] | None:
        """Grid index (cell_row, cell_col) of the cell fully containing the
        base patch whose pixel origin is (row_px, col_px), else None.

        The patch fits iff its in-cell offset (origin mod pitch) plus the
        patch length stays within the cell interior on both axes; otherwise
        it straddles a buffer strip. A cell window's own index is the same
        floor-division applied to its origin (a cell's offset is always 0).
        """
        if (row_px % self.pitch_px) + self.patch.patch_px > self.cell_px:
            return None
        if (col_px % self.pitch_px) + self.patch.patch_px > self.cell_px:
            return None
        return (row_px // self.pitch_px, col_px // self.pitch_px)


@dataclass
class PatchIndex:
    """Persistent patch coordinate list shared by all dataset backends.

    Attributes:
        spec (PatchGridSpec): Sweep that produced the patches, so an index
            is self-describing.
        product (str): Raster product the patches were enumerated on
            (e.g. "AMF_RTP").
        patches (list[PatchWindow]): The patch windows themselves.
        extra (dict[str, Any]): Dataset-specific metadata (e.g. `{"block": 2}`
            for KSA, `{"dataset": "wa"}` for WA). Opaque to the index itself
            but round-trips through save/load.
    """

    spec: PatchGridSpec
    product: str
    patches: list[PatchWindow] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self) -> Iterator[PatchWindow]:
        return iter(self.patches)

    def source_ids(self) -> list[str]:
        """Sorted unique `source_id` values across all patches."""
        return sorted({t.source_id for t in self.patches})

    def filter(self, predicate: Callable[[PatchWindow], bool]) -> "PatchIndex":
        """New index keeping only patches where `predicate(patch)` is True."""
        return PatchIndex(
            spec=self.spec,
            product=self.product,
            patches=[t for t in self.patches if predicate(t)],
            extra=dict(self.extra),
        )

    def split(
        self,
        fractions: dict[str, float],
        *,
        seed: int = 42,
    ) -> dict[str, "PatchIndex"]:
        """Random split into named subsets by `fractions` (must sum to 1.0)."""
        total = sum(fractions.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"fractions must sum to 1.0, got {total}")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.patches))
        out: dict[str, PatchIndex] = {}
        start = 0
        for name, frac in fractions.items():
            n = int(round(frac * len(self.patches)))
            picks = order[start : start + n].tolist()
            out[name] = PatchIndex(
                spec=self.spec,
                product=self.product,
                patches=[self.patches[i] for i in picks],
                extra=dict(self.extra),
            )
            start += n
        return out

    def save(self, path: Path | str) -> None:
        """Write the index (spec + patches + extra) as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "spec": asdict(self.spec),
            "product": self.product,
            "extra": self.extra,
            "patches": [asdict(t) for t in self.patches],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "PatchIndex":
        """Read an index written by `save`."""
        with open(path) as f:
            payload = json.load(f)
        spec = PatchGridSpec(**payload["spec"])
        patches = [PatchWindow(**t) for t in payload["patches"]]
        return cls(
            spec=spec,
            product=payload["product"],
            patches=patches,
            extra=payload.get("extra", {}),
        )


# ============================================================
# Patch construction: mask + sliding-window enumeration
# ============================================================


def compute_mask(
    source: RasterSource,
    *,
    block_size: int = 512,
    nodata_override: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a full-resolution boolean validity mask + georef meta.

    Mask cells are True where the pixel is neither nodata nor NaN/inf.
    Block-wise reads keep peak memory bounded for VRTs that blend many
    source rasters.

    Returns:
        mask (np.ndarray): Bool array of shape (height, width).
        meta (dict): Georeferencing carried alongside the mask everywhere:
            "transform" (affine.Affine), "crs" (rasterio CRS),
            "nodata" (float | None), "width"/"height" (int).
    """
    ds = source.dataset
    width, height = ds.width, ds.height
    src_nodata = nodata_override if nodata_override is not None else ds.nodata

    mask = np.zeros((height, width), dtype=bool)
    for row_off in range(0, height, block_size):
        h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            w = min(block_size, width - col_off)
            data = ds.read(1, window=Window(col_off, row_off, w, h))
            valid = np.isfinite(data)
            if src_nodata is not None:
                valid &= data != src_nodata
            mask[row_off : row_off + h, col_off : col_off + w] = valid

    meta = {
        "transform": ds.transform,
        "crs": ds.crs,
        "nodata": src_nodata,
        "width": width,
        "height": height,
    }
    return mask, meta


def save_mask_geotiff(
    mask: np.ndarray,
    meta: dict[str, Any],
    out_path: Path | str,
) -> None:
    """Persist a bool mask as a uint8 GeoTIFF openable in QGIS."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=meta["height"],
        width=meta["width"],
        count=1,
        dtype="uint8",
        crs=meta["crs"],
        transform=meta["transform"],
        nodata=0,
        compress="deflate",
    ) as dst:
        dst.write(mask.astype(np.uint8), 1)


def sliding_window_patches(
    mask: np.ndarray,
    meta: dict[str, Any],
    spec: PatchGridSpec,
    *,
    source_id: str,
    polygon=None,
    allow_partial: bool = False,
) -> list[PatchWindow]:
    """Slide a `spec.patch_px` square over the mask with `spec.stride_px` step.

    Algorithm:
      1. Start at pixel (row_px=0, col_px=0) — top-left of the raster.
      2. Advance by `stride_px` horizontally, wrapping to the next row.
      3. For each position, slice the mask and compute `valid_frac`.
      4. Drop patches whose `valid_frac` is below `spec.min_valid_frac`.
      5. Optionally drop patches whose world-coord bbox doesn't intersect
         `polygon` (shapely geometry). WA uses this for train/test splits.

    Args:
        mask: Bool validity array from `compute_mask` (full raster).
        meta: Georef dict from `compute_mask`; world coordinates come from
            `meta["transform"]` via `PatchWindow.from_pixel`.
        spec: Sweep geometry; pass `CellGridSpec.as_cell_sweep()` to
            enumerate cells instead of base patches.
        source_id: Stamped on every emitted window.
        polygon: Optional shapely geometry filter.
        allow_partial: Also emit trailing windows that overhang the raster's
            bottom/right edge. `valid_frac` is computed over the clipped
            slice; the window's world bbox keeps its nominal (full) size.
            Base patches must stay full-size (False); cell sweeps pass True
            so ground near the raster edge isn't lost — a window taller
            than the leftover raster band would otherwise drop everything
            beneath the last full lattice row (e.g. the bottom of B3).

    Returns:
        Kept windows in row-major sweep order.
    """
    transform = meta["transform"]
    crs = meta.get("crs")
    epsg = (
        f"EPSG:{crs.to_epsg()}"
        if crs is not None and getattr(crs, "to_epsg", None) and crs.to_epsg()
        else (str(crs) if crs is not None else None)
    )

    height, width = mask.shape
    patch_px = spec.patch_px
    stride_px = spec.stride_px
    min_valid_frac = spec.min_valid_frac

    row_end = height if allow_partial else height - patch_px + 1
    col_end = width if allow_partial else width - patch_px + 1
    patches: list[PatchWindow] = []
    for r in range(0, row_end, stride_px):
        for c in range(0, col_end, stride_px):
            patch = mask[r : r + patch_px, c : c + patch_px]  # numpy clips at edges
            vf = float(patch.mean())
            if vf < min_valid_frac:
                continue

            tw = PatchWindow.from_pixel(
                source_id=source_id,
                row_px=r,
                col_px=c,
                transform=transform,
                patch_px=patch_px,
                valid_frac=vf,
                epsg=epsg,
            )

            if polygon is not None:
                box = shapely_box(tw.left, tw.bottom, tw.right, tw.top)
                if not polygon.intersects(box):
                    continue

            patches.append(tw)
    return patches


# ============================================================
# Cell grid: Hedgementation-style mid-scale split units
# ============================================================
#
# raster -> cells (square groups of `cell_patches` x `cell_patches` base patches,
# separated by a buffer strip) -> base patches (the usual PatchGridSpec
# sweep, unchanged). Train/val/test membership is decided per cell.


def bucket_patches_by_cell(
    patches: list[PatchWindow],
    cells: list[PatchWindow],
    spec: CellGridSpec,
) -> dict[str, list[PatchWindow]]:
    """Group base patches by their owning cell, keyed by the cell's `.name`.

    `patches` comes from a `sliding_window_patches` sweep with `spec.patch`,
    `cells` from a sweep with `spec.as_cell_sweep()`, both over the same
    mask (same raster + source_id, since grid indices are only meaningful
    within one raster). Pure pixel arithmetic — no raster reads.

    Patches that straddle a buffer strip, or whose cell was dropped by
    `min_cell_valid_frac`, are excluded. Cells keeping zero patches are
    omitted from the result.

    Returns:
        {cell.name: patches inside that cell's interior}.
    """
    by_index = {(c.row_px // spec.pitch_px, c.col_px // spec.pitch_px): c for c in cells}
    out: dict[str, list[PatchWindow]] = {}
    for t in patches:
        key = spec.cell_of(t.row_px, t.col_px)
        if key is None or key not in by_index:
            continue
        out.setdefault(by_index[key].name, []).append(t)
    return out


def assign_cell_splits(
    cells: list[PatchWindow],
    fractions: dict[str, float],
    *,
    seed: int = 42,
    weights: list[int] | None = None,
) -> dict[str, str]:
    """Randomly assign whole cells to named splits: {cell.name: split}.

    Cell-level analogue of `PatchIndex.split`: validate that fractions sum
    to 1.0, permute the cells with a seeded rng, carve contiguous runs by
    fraction. Calling this once per block/region and merging the dicts
    gives an even spread of every split across regions with no extra
    stratification machinery.

    `weights` (aligned with `cells`) makes the fractions targets in total
    weight rather than cell count — pass per-cell patch counts so a split's
    share of *patches* tracks its fraction even though edge cells hold far
    fewer patches than interior ones. Splits are filled smallest-fraction
    first, each taking permuted cells (at least one) while doing so brings
    its total weight closer to target, so any error stays within half a
    cell; the largest split (train) absorbs all remaining cells.
    """
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {total}")
    w = np.ones(len(cells)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(cells):
        raise ValueError(f"weights length {len(w)} != cells length {len(cells)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cells))
    by_size = sorted(fractions, key=lambda k: fractions[k])
    out: dict[str, str] = {}
    pos = 0
    for name in by_size[:-1]:
        target = fractions[name] * float(w.sum())
        got = 0.0
        while pos < len(order) and (
            (got == 0.0 and target > 0) or abs(got + w[order[pos]] - target) < abs(got - target)
        ):
            i = order[pos]
            out[cells[i].name] = name
            got += w[i]
            pos += 1
    for i in order[pos:]:
        out[cells[i].name] = by_size[-1]
    return out

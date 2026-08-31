"""Plot helpers for masks and patch grids.

Shared between the KSA patch-index pipeline and any WA equivalents.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from rasterio.transform import Affine

from magsr.datasets.patching import PatchGridSpec, PatchWindow, sliding_window_patches

# |value| above this is treated as a sentinel/outlier when loading rasters for display.
SENTINEL_ABS_THRESHOLD = 1e5

# White = nodata, light gray = valid — keeps bright patch overlays readable.
MASK_CMAP = ListedColormap(["white", "#c8c8c8"])

# Boundary-retention figure: NaN/no-data = white, valid HR pixels = blue-gray, so
# the coloured retain/reject rectangles overlaid on top read clearly. The valid
# tone is saturated enough that the data/NaN edge still shows under translucent fills.
BOUNDARY_VALID_RGB = "#a6bccb"
BOUNDARY_MASK_CMAP = ListedColormap(["white", BOUNDARY_VALID_RGB])

# Retention categories for `plot_boundary_patch_retention`, keyed by how a base
# patch fares against the min-valid-fraction rule. `label` may carry a `{thr}`
# placeholder (filled with the threshold percent at draw time).
BOUNDARY_STYLE = {
    "full": dict(
        face="#2ca02c",
        edge="#1b5e20",
        alpha=0.16,
        ls="solid",
        label="retained — fully valid",
    ),
    "kept_nan": dict(
        face="#f5a623",
        edge="#a86400",
        alpha=0.42,
        ls="solid",
        label="retained — contains NaN (neutral-filled)",
    ),
    "rejected": dict(
        face="#d62728",
        edge="#7f0000",
        alpha=0.34,
        ls=(0, (3, 2)),
        label="rejected — < {thr} valid",
    ),
}

# Distinct high-contrast palette for grid_cycle coloring.
PALETTE = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#ff7f00",
    "#984ea3",
    "#a65628",
    "#f781bf",
    "#ffff33",
    "#999999",
    "#66c2a5",
]


def load_magnetic_strided(
    path: Path | str, stride: int
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Decimated HR raster + (left, right, bottom, top) extent. NaN-aware, scrubs sentinel outliers."""
    with rasterio.open(path) as src:
        out_h, out_w = src.height // stride, src.width // stride
        arr = src.read(1, out_shape=(out_h, out_w), out_dtype=np.float32)
        nd = src.nodata
        b = src.bounds
    if nd is not None and not np.isnan(nd):
        arr[arr == nd] = np.nan
    arr[np.abs(arr) > SENTINEL_ABS_THRESHOLD] = np.nan
    return arr, (b.left, b.right, b.bottom, b.top)


def mask_extent(meta: dict) -> list[float]:
    """Return [left, right, bottom, top] for use with imshow(extent=...)."""
    t = meta["transform"]
    left = t.c
    top = t.f
    right = left + meta["width"] * t.a
    bottom = top + meta["height"] * t.e  # t.e is negative for north-up
    return [left, right, bottom, top]


def _downsample_mask(
    mask: np.ndarray,
    meta: dict,
    max_axis: int,
) -> tuple[np.ndarray, dict]:
    """Stride-decimate a mask so its longest side is <= `max_axis`.

    Returns a view (or the original if no downsampling is needed) and a
    meta dict whose affine has been scaled to match the stride, so world
    coords — and patch overlays that reference absolute world coords — still
    land in the correct location on the plotted image.
    """
    h, w = mask.shape
    stride = max(1, int(np.ceil(max(h, w) / max_axis)))
    if stride == 1:
        return mask, meta
    ds_mask = mask[::stride, ::stride]
    t = meta["transform"]
    new_meta = dict(meta)
    new_meta["transform"] = Affine(t.a * stride, t.b, t.c, t.d, t.e * stride, t.f)
    new_meta["width"] = ds_mask.shape[1]
    new_meta["height"] = ds_mask.shape[0]
    return ds_mask, new_meta


def _valid_bbox_world(
    mask: np.ndarray,
    meta: dict,
    pad_frac: float = 0.02,
) -> tuple[float, float, float, float] | None:
    """World-coord (left, right, bottom, top) of the True region in `mask`.

    Returns `None` if the mask is entirely False. `pad_frac` adds a small
    margin around the bbox so patch rectangles at the edge aren't flush
    against the axis frame.
    """
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.argmax(rows))
    r1 = len(rows) - int(np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = len(cols) - int(np.argmax(cols[::-1]))
    t = meta["transform"]
    left = t.c + c0 * t.a
    right = t.c + c1 * t.a
    top = t.f + r0 * t.e
    bottom = t.f + r1 * t.e  # t.e negative -> bottom < top
    pad_x = (right - left) * pad_frac
    pad_y = (top - bottom) * pad_frac
    return left - pad_x, right + pad_x, bottom - pad_y, top + pad_y


def plot_mask(
    mask: np.ndarray,
    meta: dict,
    title: str,
    out_path: Path | str,
    *,
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 120,
    max_display_axis: int = 4000,
    fit_to_data: bool = True,
) -> None:
    """Single-panel mask PNG in world coordinates.

    Masks larger than `max_display_axis` pixels on the longest side are
    stride-decimated for rendering only — the world-coord extent is
    unchanged. Keeps matplotlib from OOMing on 400+ MB full-AOI masks
    like the pre-snapped KSA grid.

    `fit_to_data` (default True) zooms the axes to the world-coord bbox
    of the True region, so a small block embedded in a large canonical
    grid renders focused instead of as a tiny island in the corner.
    """
    mask, meta = _downsample_mask(mask, meta, max_display_axis)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(
        mask,
        extent=mask_extent(meta),
        origin="upper",
        cmap=MASK_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    if fit_to_data:
        bbox = _valid_bbox_world(mask, meta)
        if bbox is not None:
            left, right, bottom, top = bbox
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    gc.collect()


def plot_patches_over_mask(
    mask: np.ndarray,
    meta: dict,
    patches: Iterable[PatchWindow],
    spec: PatchGridSpec,
    title: str,
    out_path: Path | str,
    *,
    color_by: str | None = "grid_cycle",
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 150,
    max_display_axis: int = 4000,
    fit_to_data: bool = True,
) -> None:
    """Mask background with patch rectangles overlaid.

    `color_by`:
      - None         — all patches in `PALETTE[0]`, alpha 0.3
      - 'grid_cycle' — cycle PALETTE by (row//stride + col//stride) so
                       neighbors are distinct; helps spot stride seams
      - 'parity'     — 4-color checkerboard on (row_parity, col_parity);
                       useful for visualizing 2x2 overlap patterns

    Masks larger than `max_display_axis` on the longest side are
    stride-decimated for rendering only. Patch overlays still reference
    absolute world coords, so they land on the correct location of the
    (now smaller) image.

    `fit_to_data` (default True) zooms the axes to the world-coord bbox
    of the True region. With a sub-region embedded in a large canonical
    grid (e.g. one KSA block on the full pre-snapped AOI), this gives a
    focused per-block view instead of a tiny island in a vast canvas.
    """
    patches = list(patches)
    mask, meta = _downsample_mask(mask, meta, max_display_axis)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(
        mask,
        extent=mask_extent(meta),
        origin="upper",
        cmap=MASK_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    rects: list[Rectangle] = []
    colors: list[str] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    for t in patches:
        rects.append(Rectangle((t.left, t.bottom), t.width, t.height))
        centers_x.append(0.5 * (t.left + t.right))
        centers_y.append(0.5 * (t.bottom + t.top))
        if color_by == "grid_cycle":
            g = (t.row_px // spec.stride_px) + (t.col_px // spec.stride_px)
            colors.append(PALETTE[g % len(PALETTE)])
        elif color_by == "parity":
            gr = (t.row_px // spec.stride_px) % 2
            gc_ = (t.col_px // spec.stride_px) % 2
            colors.append(PALETTE[gr * 2 + gc_])
        else:
            colors.append(PALETTE[0])

    coll = PatchCollection(
        rects,
        facecolors="none",
        edgecolors=colors,
        alpha=0.7,
        linewidths=0.4,
    )
    coll.set_rasterized(True)
    ax.add_collection(coll)

    ax.scatter(
        centers_x,
        centers_y,
        c=colors,
        s=4,
        marker="o",
        linewidths=0,
        alpha=0.9,
        rasterized=True,
    )

    if fit_to_data:
        bbox = _valid_bbox_world(mask, meta)
        if bbox is not None:
            left, right, bottom, top = bbox
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"{title}\n"
        f"{len(patches)} patches — patch={spec.patch_px}, "
        f"stride={spec.stride_px}, min_valid={spec.min_valid_frac:.0%}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    gc.collect()


# ============================================================
# Boundary patch-retention illustration
# ============================================================
#
# The KSA raster is irregular: boundary patches hold valid pixels but also NaN.
# A patch is kept iff its valid fraction >= min_valid_frac (0.97); the NaN it
# still carries is neutral-filled at train time and masked out of metrics. This
# figure zooms into a boundary window and colours each candidate patch by that
# retain/reject decision, so the rule is legible against the NaN mask.


def _summed_area(mask: np.ndarray) -> np.ndarray:
    """Integral image of a bool mask (shape (H+1, W+1)) for O(1) window sums."""
    sat = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(mask, axis=0, dtype=np.int64), axis=1, out=sat[1:, 1:])
    return sat


def pick_boundary_crop(
    mask: np.ndarray,
    spec: PatchGridSpec,
    *,
    crop_h_px: int,
    crop_w_px: int,
    stride_px: int,
    min_valid_frac: float,
) -> tuple[int, int, float, dict[str, int]]:
    """Scan for a crop window straddling the data/NaN edge with a mix of patch fates.

    Slides a `crop_h_px` x `crop_w_px` window over the mask (on a coarse grid)
    and, using an integral image, counts how many enclosed base patches would be
    fully valid, retained-with-NaN, or rejected. Returns the best
    `(row0, col0, score, counts)` — one that shows all three categories against a
    partly-NaN background so the retention rule is visible. Falls back to the most
    category-diverse window if no crop cleanly contains all three.
    """
    h, w = mask.shape
    ph = spec.patch_px
    crop_h_px = min(crop_h_px, h)
    crop_w_px = min(crop_w_px, w)
    sat = _summed_area(mask)

    def win_valid(r: int, c: int, hh: int, ww: int) -> int:
        return int(sat[r + hh, c + ww] - sat[r, c + ww] - sat[r + hh, c] + sat[r, c])

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, 0, 0.0, {"full": 0, "kept_nan": 0, "rejected": 0}
    r_lo, r_hi = int(rows[0]), min(int(rows[-1]), h - crop_h_px)
    c_lo, c_hi = int(cols[0]), min(int(cols[-1]), w - crop_w_px)
    step = max(ph, min(crop_h_px, crop_w_px) // 3)

    # Track the strictly-good crop (all three fates, sensible fill) separately
    # from a relaxed fallback (most edge patches), so we always return something.
    best: tuple | None = None
    fallback: tuple | None = None
    for r0 in range(r_lo, max(r_lo, r_hi) + 1, step):
        for c0 in range(c_lo, max(c_lo, c_hi) + 1, step):
            counts = {"full": 0, "kept_nan": 0, "rejected": 0}
            for r in range(r0, r0 + crop_h_px - ph + 1, stride_px):
                for c in range(c0, c0 + crop_w_px - ph + 1, stride_px):
                    vf = win_valid(r, c, ph, ph) / (ph * ph)
                    if vf <= 0:
                        continue
                    if vf >= 1.0 - 1e-9:
                        counts["full"] += 1
                    elif vf >= min_valid_frac:
                        counts["kept_nan"] += 1
                    else:
                        counts["rejected"] += 1
            edge = counts["kept_nan"] + counts["rejected"]
            fb_score = edge + counts["full"]
            if fallback is None or fb_score > fallback[0]:
                fallback = (fb_score, r0, c0, counts)
            overall = win_valid(r0, c0, crop_h_px, crop_w_px) / (crop_h_px * crop_w_px)
            if not (0.30 <= overall <= 0.80):
                continue
            balance = min(counts["full"], counts["kept_nan"], counts["rejected"])
            if balance == 0:
                continue
            # Favour a window where all three fates are well represented (balance
            # dominates) before rewarding more edge patches — a lopsided all-red
            # sliver is a poorer illustration than a clean valid/boundary/NaN mix.
            score = 10 * balance + edge
            if best is None or score > best[0]:
                best = (score, r0, c0, counts)

    chosen = best if best is not None else fallback
    _, r0, c0, counts = chosen
    return r0, c0, float(chosen[0]), counts


def plot_boundary_patch_retention(
    mask: np.ndarray,
    meta: dict,
    spec: PatchGridSpec,
    title: str,
    out_path: Path | str,
    *,
    crop: tuple[int, int] | None = None,
    crop_patches: int | tuple[int, int] = 8,
    stride_px: int | None = None,
    min_valid_frac: float | None = None,
    annotate_frac: bool = True,
    compact: bool = False,
    simple_legend: bool = False,
    figsize: tuple[float, float] | None = None,
    dpi: int = 200,
) -> tuple[int, int, dict[str, int]]:
    """Zoom into a boundary window and colour each patch by its retain/reject fate.

    Background is the validity mask (white = NaN/no-data, blue-gray = valid HR
    pixels). Every candidate patch of the `spec` sweep that falls in the window is
    outlined; those overlapping valid ground are filled by category — fully valid,
    retained-with-NaN (>= `min_valid_frac`), or rejected (< it). Retained-with-NaN
    and rejected patches are annotated with their valid fraction so the threshold
    is legible.

    `crop_patches` is the window size in base patches: an int for a square window
    or `(cols, rows)` for a rectangle (use a wide `(cols, rows)` for a compact
    strip). `crop` pins the window top-left as `(row0, col0)` in mask pixels;
    `None` auto-selects a boundary window via `pick_boundary_crop`. The plotting
    `stride_px` defaults to `spec.patch_px` (a clean non-overlapping grid, clearest
    for the rule) independent of the training sampler's denser stride.

    `compact` drops the map axes/labels and shrinks the title, legend, and
    annotations to a tight wide strip that drops straight into a two-column paper.

    Returns the `(row0, col0, counts)` actually drawn.
    """
    ph = spec.patch_px
    stride_px = ph if stride_px is None else stride_px
    thr = spec.min_valid_frac if min_valid_frac is None else min_valid_frac
    cols, rows_p = (crop_patches, crop_patches) if isinstance(crop_patches, int) else crop_patches
    h, w = mask.shape
    crop_h_px = min(rows_p * ph, h)
    crop_w_px = min(cols * ph, w)
    if figsize is None:
        figsize = (7.2, 2.7) if compact else (9.5, 9.5)

    if crop is None:
        row0, col0, _, _ = pick_boundary_crop(
            mask,
            spec,
            crop_h_px=crop_h_px,
            crop_w_px=crop_w_px,
            stride_px=stride_px,
            min_valid_frac=thr,
        )
    else:
        row0, col0 = int(crop[0]), int(crop[1])
    row0 = max(0, min(int(row0), h - crop_h_px))
    col0 = max(0, min(int(col0), w - crop_w_px))

    sub = mask[row0 : row0 + crop_h_px, col0 : col0 + crop_w_px]
    t = meta["transform"]
    sub_t = Affine(t.a, t.b, t.c + col0 * t.a, t.d, t.e, t.f + row0 * t.e)
    sub_meta = {**meta, "transform": sub_t, "width": sub.shape[1], "height": sub.shape[0]}
    extent = mask_extent(sub_meta)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(
        sub,
        extent=extent,
        origin="upper",
        cmap=BOUNDARY_MASK_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Crisp data/NaN edge so the irregular boundary reads through the fills.
    # contour needs increasing coords, so feed a vertically-flipped array + y-axis.
    xs = sub_t.c + (np.arange(sub.shape[1]) + 0.5) * sub_t.a
    ys = sub_t.f + (np.arange(sub.shape[0]) + 0.5) * sub_t.e
    ax.contour(
        xs,
        ys[::-1],
        sub[::-1].astype(float),
        levels=[0.5],
        colors="#2b2b2b",
        linewidths=1.0 if compact else 1.3,
    )

    sweep = PatchGridSpec(patch_px=ph, stride_px=stride_px, min_valid_frac=0.0)
    cand = sliding_window_patches(sub, sub_meta, sweep, source_id="boundary")

    edge_lw = 0.9 if compact else 1.1
    anno_fs = 5.0 if compact else 6.5
    counts = {"full": 0, "kept_nan": 0, "rejected": 0}
    for pw in cand:
        # Faint full lattice so the dropped (all-NaN) cells still read as tiling.
        ax.add_patch(
            Rectangle(
                (pw.left, pw.bottom),
                pw.width,
                pw.height,
                facecolor="none",
                edgecolor="#9aa0a6",
                linewidth=0.3,
                alpha=0.5,
            )
        )
        vf = pw.valid_frac
        if vf <= 0:
            continue
        cat = "full" if vf >= 1.0 - 1e-9 else ("kept_nan" if vf >= thr else "rejected")
        counts[cat] += 1
        style = BOUNDARY_STYLE[cat]
        ax.add_patch(
            Rectangle(
                (pw.left, pw.bottom),
                pw.width,
                pw.height,
                facecolor=style["face"],
                edgecolor=style["edge"],
                linewidth=edge_lw,
                linestyle=style["ls"],
                alpha=style["alpha"],
            )
        )
        if annotate_frac and cat != "full":
            # Floor to one decimal near the top so a 99.997%-valid kept patch
            # reads "99.9%", never a misleading "100%" on a sub-100% patch.
            pct = f"{np.floor(vf * 1000) / 10:.1f}%" if vf >= 0.99 else f"{round(vf * 100)}%"
            ax.text(
                0.5 * (pw.left + pw.right),
                0.5 * (pw.bottom + pw.top),
                pct,
                ha="center",
                va="center",
                fontsize=anno_fs,
                color=style["edge"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.72),
            )

    from matplotlib.patches import Patch  # local: only this figure builds a legend

    # `simple_legend` shows just the three patch fates with short labels; the full
    # legend also carries the mask swatches and the numeric threshold.
    short = {"full": "retained", "kept_nan": "partial NaN", "rejected": "rejected"}
    legend = (
        []
        if simple_legend
        else [
            Patch(facecolor=BOUNDARY_VALID_RGB, edgecolor="#6b7b88", label="valid HR pixels"),
            Patch(facecolor="white", edgecolor="#2b2b2b", label="NaN / no-data"),
        ]
    )
    for cat in ("full", "kept_nan", "rejected"):
        style = BOUNDARY_STYLE[cat]
        label = short[cat] if simple_legend else style["label"].format(thr=f"{thr:.0%}")
        legend.append(
            Patch(
                facecolor=style["face"],
                edgecolor=style["edge"],
                alpha=max(style["alpha"], 0.4),
                linestyle=style["ls"],
                label=label,
            )
        )
    # Below the axes so it never covers boundary patches; one row when compact.
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02 if compact else -0.08),
        ncol=len(legend) if compact else 3,
        fontsize=6.5 if compact else 8.5,
        framealpha=0.95,
        borderpad=0.4 if compact else 0.6,
        columnspacing=1.0 if compact else 1.4,
        handlelength=1.4,
        handletextpad=0.5,
    )

    ax.set_aspect("equal")
    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    else:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title(
            f"{title}\n{ph}px patches, min-valid = {thr:.0%}  "
            f"(retained {counts['full'] + counts['kept_nan']}, rejected {counts['rejected']})",
            fontsize=12,
            fontweight="bold",
        )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    gc.collect()
    return row0, col0, counts

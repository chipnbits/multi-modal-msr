"""LR / reconstruction / HR-truth viewers for one or many patchwise-inference runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from magsr.reconstruct.build import PatchPlan
from magsr.viz import SENTINEL_ABS_THRESHOLD  # single-sourced (|value| > 1e5 → nodata scrub)

WorldBBox = tuple[float, float, float, float]


def _read_finite(
    path: Path | str,
    *,
    window: Window | None = None,
    bbox: WorldBBox | None = None,
) -> np.ndarray:
    """Read band 1, NaN-cleaned. Pass `bbox=(left, bottom, right, top)` to align by world coords
    (correct when the source raster doesn't share a pixel grid with the plan's LR raster)."""
    with rasterio.open(path) as src:
        if bbox is not None:
            if window is not None:
                raise ValueError("pass either `window` or `bbox`, not both")
            window = (
                from_bounds(*bbox, transform=src.transform)
                .round_offsets()
                .round_lengths()
                .intersection(Window(0, 0, src.width, src.height))
            )
        arr = src.read(1, window=window).astype(np.float32, copy=False)
        nd = src.nodata
    if nd is not None and not np.isnan(nd):
        arr = np.where(arr == nd, np.nan, arr)
    arr[np.abs(arr) > SENTINEL_ABS_THRESHOLD] = np.nan
    return arr


def _percentile_range(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return -1.0, 1.0
    return float(np.percentile(finite, 2)), float(np.percentile(finite, 98))


def plot_reconstruction(
    *,
    out_path: Path | str,
    lr_path: Path | str,
    plan: PatchPlan,
    hr_truth_path: Path | str | None = None,
    save_path: Path | str | None = None,
    show_patch_centres: bool = False,
) -> plt.Figure:
    """Side-by-side LR (over the buffered window) / reconstruction / HR truth (if given)."""
    lr_arr = _read_finite(lr_path, window=plan.lr_window)
    recon_view = _read_finite(out_path)

    truth_view = None
    if hr_truth_path is not None:
        truth_view = _read_finite(hr_truth_path, bbox=plan.world_bbox)

    vmin, vmax = _percentile_range(recon_view)

    n = 3 if truth_view is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    t = plan.lr_window_transform
    extent = [
        t.c,
        t.c + plan.lr_window.width * t.a,
        t.f + plan.lr_window.height * t.e,
        t.f,
    ]

    axes[0].imshow(lr_arr, extent=extent, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"LR (buffered window, {lr_arr.shape[1]}x{lr_arr.shape[0]} px)")
    axes[1].imshow(recon_view, extent=extent, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Reconstruction ({recon_view.shape[1]}x{recon_view.shape[0]} HR px)")
    if truth_view is not None:
        axes[2].imshow(truth_view, extent=extent, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
        axes[2].set_title("HR truth")

    if show_patch_centres:
        ht = plan.hr_window_transform
        for r_lr, c_lr in plan.positions:
            cx = ht.c + (c_lr * plan.scale + plan.patch_hr_px / 2) * ht.a
            cy = ht.f + (r_lr * plan.scale + plan.patch_hr_px / 2) * ht.e
            axes[1].plot(cx, cy, ".", color="red", markersize=2)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


@dataclass(frozen=True)
class ReconItem:
    """One block's plan + paths to the LR raster, the written reconstruction TIF, and (optional) HR truth."""

    label: str
    plan: PatchPlan
    lr_path: Path
    recon_path: Path
    truth_path: Path | None = None


def plot_multi_block_grid(
    items: list[ReconItem],
    *,
    save_path: Path | str | None = None,
    cmap: str = "gray",
    percentile: tuple[float, float] = (2, 98),
    title: str | None = None,
    cbar_label: str = "Magnetic intensity (nT)",
) -> plt.Figure:
    """N rows (one per block) × 2-or-3 cols (LR / Reconstruction / HR truth) with one shared colorbar."""
    rows = []
    for item in items:
        bbox = item.plan.world_bbox
        lr_arr = _read_finite(item.lr_path, window=item.plan.lr_window)
        recon_arr = _read_finite(item.recon_path)
        # Clip LR display to the recon's effective footprint so AOI plots aren't confusing
        # (LR otherwise shows the full buffered window, recon shows only patched area).
        H_lr, W_lr = lr_arr.shape
        H_hr, W_hr = recon_arr.shape
        if H_hr % H_lr == 0 and W_hr % W_lr == 0 and H_hr // H_lr == W_hr // W_lr:
            scale = H_hr // H_lr
            valid_lr = np.isfinite(recon_arr).reshape(H_lr, scale, W_lr, scale).any(axis=(1, 3))
            lr_arr = np.where(valid_lr, lr_arr, np.nan)
        rows.append(
            {
                "label": item.label,
                "plan": item.plan,
                "lr": lr_arr,
                "recon": recon_arr,
                "truth": (
                    _read_finite(item.truth_path, bbox=bbox) if item.truth_path is not None else None
                ),
            }
        )

    pooled = []
    for r in rows:
        for arr in (r["lr"], r["recon"], r["truth"]):
            if arr is None:
                continue
            v = arr[np.isfinite(arr)]
            if v.size:
                pooled.append(v)
    if not pooled:
        vmin, vmax = -1.0, 1.0
    else:
        flat = np.concatenate(pooled)
        vmin = float(np.percentile(flat, percentile[0]))
        vmax = float(np.percentile(flat, percentile[1]))

    has_truth = any(r["truth"] is not None for r in rows)
    cols = 3 if has_truth else 2
    n = len(rows)
    fig, axes = plt.subplots(n, cols, figsize=(5.5 * cols, 5.5 * n), squeeze=False)

    last_im = None
    col_titles = ["LR (180 m)", "Reconstruction (60 m)", "HR truth (60 m)"]
    for i, r in enumerate(rows):
        left_m, right_m, bottom_m, top_m = r["plan"].world_extent
        extent_km = [0.0, (right_m - left_m) / 1000.0, 0.0, (top_m - bottom_m) / 1000.0]
        for j, (key, ctitle) in enumerate(zip(("lr", "recon", "truth"), col_titles)):
            if j >= cols:
                break
            arr = r[key]
            ax = axes[i, j]
            ax.set_facecolor("0.5")  # mid-grey behind NaN/no-data
            if arr is None:
                ax.axis("off")
            else:
                last_im = ax.imshow(arr, extent=extent_km, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{r['label']} — {ctitle}" if i == 0 else r["label"] + f" — {ctitle}")
            ax.set_aspect("equal")
            ax.set_xlabel("X (km)")
            ax.set_ylabel("Y (km)")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97 if title else 1.0])
    if last_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        fig.colorbar(last_im, cax=cbar_ax, label=cbar_label)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

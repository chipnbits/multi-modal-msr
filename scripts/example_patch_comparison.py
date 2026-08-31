"""
Visualise the super-resolution task for each dataset.

Produces up to two sets of PNGs under
``figures/example_patch_comparison/``, each split into batches of
``BATCH_SIZE`` examples (default ``16 = 4 × 4``):

  * ``wa_{1..4}.png`` — Western Australia Goldfields, 4× super-resolution.
    Columns: LR | LR ↑ | HR | Residual.

  * ``aligned_{1..4}.png`` — Pre-snapped KSA Shield grid, 3× SR + DEM.
    Columns: LR | LR ↑ | HR | Residual | DEM.

If a dataset's prerequisites aren't present on this machine (missing
snapshot, un-built patch indices, etc.) that dataset is skipped with a
printed message rather than failing the whole run.
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.ndimage import zoom

from magsr import ROOT_FOLDER
from magsr.datasets import build_ksa_aligned_datasets, build_wa_datasets

N_EXAMPLES = 16
BATCH_SIZE = 4
SEED = 44
OUT_DIR = ROOT_FOLDER / "figures" / "example_patch_comparison"


# ============================================================
# Small math / array helpers
# ============================================================
def _safe_range(*arrs: np.ndarray) -> tuple[float, float]:
    """Shared (vmin, vmax) across several arrays, ignoring NaN/inf."""
    lows: list[float] = []
    highs: list[float] = []
    with warnings.catch_warnings():
        # `np.nanmin` warns on all-NaN arrays; we handle non-finite
        # results explicitly below.
        warnings.simplefilter("ignore", RuntimeWarning)
        for a in arrs:
            lo = float(np.nanmin(a))
            hi = float(np.nanmax(a))
            if np.isfinite(lo) and np.isfinite(hi):
                lows.append(lo)
                highs.append(hi)
    if not lows:
        return 0.0, 1.0
    return min(lows), max(highs)


def _sample_indices(n_total: int, n_samples: int, seed: int) -> list[int]:
    n = min(n_samples, n_total)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n, replace=False).tolist()


def _patch_label(sample: dict[str, Any], idx: int) -> str:
    meta = sample.get("meta", {})
    name = meta.get("name") or meta.get("source_id") or f"idx{idx}"
    return str(name)


def _nan_fill(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with the per-array nanmean so bicubic zoom does not
    blow up a few edge-NaN pixels into a wide NaN blot via its 4-tap
    kernel. Falls back to 0.0 if the whole array is NaN."""
    if not np.isnan(arr).any():
        return arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fill = float(np.nanmean(arr))
    if not np.isfinite(fill):
        fill = 0.0
    return np.where(np.isnan(arr), fill, arr)


# ============================================================
# Figure building
# ============================================================
def _compute_row_data(
    sample: dict[str, Any],
    *,
    hr_key: str,
    lr_key: str,
    scale: int,
    include_dem: bool,
) -> dict[str, np.ndarray]:
    """Pull raw arrays for one row (lr / lr_up / hr / residual / optional dem)."""
    hr = sample["hr"][hr_key].numpy()
    lr = sample["lr"][lr_key].numpy()

    if scale > 1:
        lr_up = zoom(_nan_fill(lr), scale, order=3)
    else:
        # KSA shield already warps LR to HR shape on read; bicubic would
        # be a visual no-op.
        lr_up = lr

    data = {"lr": lr, "lr_up": lr_up, "hr": hr, "residual": hr - lr_up}
    if include_dem:
        data["dem"] = sample["dem"].numpy()
    return data


def _build_row_panels(
    data: dict[str, np.ndarray],
    *,
    scale: int,
    mag_norm: Normalize,
    include_dem: bool,
) -> list[tuple[str, np.ndarray, Normalize, str]]:
    """Return a list of (column_key, data, norm, cmap) per row.

    The grey `mag_norm` is shared across the whole figure so LR / LR↑ /
    HR panels are directly comparable. Residual and DEM keep per-row
    norms because their magnitudes vary too much across rows to share.
    """
    residual = data["residual"]
    res_abs = float(np.nanmax(np.abs(residual))) if np.isfinite(residual).any() else 1.0
    if res_abs == 0 or not np.isfinite(res_abs):
        res_abs = 1.0
    res_norm = TwoSlopeNorm(vmin=-res_abs, vcenter=0, vmax=res_abs)

    panels: list[tuple[str, np.ndarray, Normalize, str]] = [
        ("lr", data["lr"], mag_norm, "gray"),
    ]
    if scale > 1:
        panels.append(("lr_up", data["lr_up"], mag_norm, "gray"))
    panels += [
        ("hr", data["hr"], mag_norm, "gray"),
        ("residual", residual, res_norm, "RdBu_r"),
    ]

    if include_dem:
        dem = data["dem"]
        dem_lo, dem_hi = _safe_range(dem)
        dem_norm = Normalize(vmin=dem_lo, vmax=dem_hi)
        panels.append(("dem", dem, dem_norm, "terrain"))

    return panels


def _column_title(col_key: str, data: np.ndarray, scale: int) -> str:
    h, w = data.shape
    labels = {
        "lr": f"LR input\n({h}×{w})",
        "lr_up": f"LR bicubic ↑{scale}\n({h}×{w})",
        "hr": f"HR ground truth\n({h}×{w})",
        "residual": "Residual\n(HR − LR↑)",
        "dem": f"DEM\n({h}×{w})",
    }
    return labels[col_key]


def comparison_figure(
    ds,
    *,
    hr_key: str,
    lr_key: str,
    scale: int,
    include_dem: bool,
    suptitle: str,
    out_prefix: Path,
    sample_indices: list[int] | None = None,
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Render comparison figures for a single dataset, one PNG per batch.

    Indices are chunked into groups of ``batch_size``; each chunk is
    saved as ``{out_prefix}_{i}.png`` starting at ``i=1`` (so
    ``out_prefix = OUT_DIR / "shield"`` yields ``shield_1.png``,
    ``shield_2.png``, …). When `sample_indices` is provided the figure
    renders exactly those patches (used to render the same geographic
    anchors across the two KSA backends); otherwise it draws a random
    seeded sample.
    """
    if sample_indices is not None:
        indices = list(sample_indices)
    else:
        indices = _sample_indices(len(ds), n_examples, seed)
    if not indices:
        print(f"[{out_prefix.name}] dataset split is empty — skipping")
        return

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for batch_num, start in enumerate(range(0, len(indices), batch_size), start=1):
        batch_indices = indices[start : start + batch_size]
        samples = [ds[i] for i in batch_indices]
        row_data = [
            _compute_row_data(s, hr_key=hr_key, lr_key=lr_key, scale=scale, include_dem=include_dem)
            for s in samples
        ]
        grey_arrays: list[np.ndarray] = []
        for d in row_data:
            grey_arrays.extend([d["lr"], d["lr_up"], d["hr"]])
        mag_vmin, mag_vmax = _safe_range(*grey_arrays)
        mag_norm = Normalize(vmin=mag_vmin, vmax=mag_vmax)

        rows = [
            _build_row_panels(d, scale=scale, mag_norm=mag_norm, include_dem=include_dem) for d in row_data
        ]
        n_rows = len(rows)
        n_cols = len(rows[0])

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows + 0.5), dpi=150)
        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for row_idx, (sample_idx, sample, panels) in enumerate(zip(batch_indices, samples, rows)):
            for col_idx, (col_key, data, norm, cmap) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                im = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(
                        _column_title(col_key, data, scale),
                        fontsize=10,
                        fontweight="bold",
                    )
                if col_idx == 0:
                    meta = sample.get("meta", {})
                    lat = meta.get("lat")
                    lon = meta.get("lon")
                    if lat is not None and lon is not None:
                        ylabel = f"{_patch_label(sample, sample_idx)}\n({lat:.3f}°, {lon:.3f}°)"
                    else:
                        ylabel = _patch_label(sample, sample_idx)
                    ax.set_ylabel(ylabel, fontsize=8)
                if col_key in ("residual", "dem"):
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)

        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.00)
        plt.tight_layout()
        out_path = out_prefix.with_name(f"{out_prefix.name}_{batch_num}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)


# ============================================================
# Dataset-specific entry points
# ============================================================
def _try(name: str, fn) -> None:
    """Run a dataset-specific figure builder, reporting failures as skips."""
    try:
        fn()
    except FileNotFoundError as e:
        print(f"[{name}] skipped: {e}")
    except Exception as e:  # noqa: BLE001 — keep other datasets running
        print(f"[{name}] skipped due to unexpected error: {type(e).__name__}: {e}")


def _wa_figure() -> None:
    splits = build_wa_datasets()
    comparison_figure(
        splits["test"],
        hr_key="MAG",
        lr_key="MAG",
        scale=4,
        include_dem=False,
        suptitle=("Western Australia Goldfields — 4× super-resolution " "(20 m HR / 80 m LR)"),
        out_prefix=OUT_DIR / "wa",
    )


def _ksa_aligned_figure() -> None:
    aligned_splits = build_ksa_aligned_datasets(
        index_dir=ROOT_FOLDER / "data/processed/ksa_aligned/patch_indices_cellgrid8_fold3"
    )
    aligned_test = aligned_splits["test"]
    if len(aligned_test) == 0:
        raise FileNotFoundError(
            "KSA aligned test split is empty; run scripts/ksa_aligned_build_patch_indices.py."
        )

    comparison_figure(
        aligned_test,
        hr_key="AMF_RTP",
        lr_key="RTP",
        scale=3,
        include_dem=True,
        suptitle=("KSA Shield aligned (pre-snapped grid) — 3× super-resolution + 30 m DEM"),
        out_prefix=OUT_DIR / "aligned",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _try("wa", _wa_figure)
    _try("aligned", _ksa_aligned_figure)


if __name__ == "__main__":
    main()

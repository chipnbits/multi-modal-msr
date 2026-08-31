"""Schematic of the FVD (first-vertical-derivative) physics-loss operator.

Walks a single HR magnetic patch through every step of
``magsr.fourier.first_vertical_derivative`` so the FFT operator reads as a picture:

    raw f  --(subtract plane a·x+b·y+c)-->  detrended  --(x Tukey window)-->  tapered
           --(zero-pad + implied periodic tiling)-->  3x3 tile
           --(FFT)-->  |F|  --(x |k| upward-derivative kernel)-->  |F|·|k|  --(IFFT)-->  ∂f/∂z

The panels reuse the *actual* operator helpers (`bilinear_fit`, `tukey2d`, `pad_centered`,
`make_wavenumbers_rfft`, …) so the schematic is exactly the code path, not a redraw. The
fitted plane and the Tukey window are drawn as small insets on the arrows that apply them,
and the |k| kernel as an inset on the multiply that forms the derivative. Spectra are shown
as centred (fftshift) full-FFT log-magnitude for legibility — the operator itself uses the
equivalent rfft2 half-spectrum.

    uv run python experiments/wa_dataset/schematic_fvd_operator.py --dataset ksa --idx 1037

Output defaults to SVG so the panels import into Inkscape / draw.io as separate objects
(each imshow is one embedded raster; every arrow, frame, title and inset stays vector).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch

from magsr import ROOT_FOLDER
from magsr.datasets import build_ksa_aligned_datasets, build_wa_datasets
from magsr.fourier._fft_utils import (
    apply_plane,
    bilinear_fit,
    crop_centered,
    make_wavenumbers,
    pad_centered,
    tukey2d,
)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def load_patch_nt(dataset: str, idx: int) -> torch.Tensor:
    """Return a single fully-valid HR patch as an ``(H, W)`` float tensor in nanoTesla."""
    if dataset == "ksa":
        # any fold works — this is illustration only; use the canonical fold-3 test split.
        ds = build_ksa_aligned_datasets(
            index_dir=ROOT_FOLDER / "data/processed/ksa_aligned/patch_indices_cellgrid8_fold3"
        )["test"]
        hr = ds[idx]["hr"]["AMF_RTP"].float()  # already in nT
    elif dataset == "wa":
        ds = build_wa_datasets()["test"]
        hr = ds[idx]["hr"]["MAG"].float() * (ds.vmax - ds.vmin) + ds.vmin  # [0,1] -> nT
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    if not torch.isfinite(hr).all():
        raise SystemExit(f"{dataset} patch {idx} has NaNs; pick a fully-valid patch")
    return hr


def compute_pipeline(field: torch.Tensor, dx: float, dy: float, taper_frac: float):
    """Run every step of `first_vertical_derivative`, returning the intermediate arrays.

    Mirrors `magsr.fourier.fvd.first_vertical_derivative` step for step; additionally keeps
    the centred full-FFT spectra (for display) alongside the operator's rfft path.
    """
    h, w = field.shape
    pad_to = _next_pow2(int(1.5 * max(h, w)))

    plane_coef = bilinear_fit(field)  # (a, b, c)
    plane = apply_plane(plane_coef, h, w)
    detr = field - plane  # detrend (not restored)
    win = tukey2d(h, w, taper_frac)
    taper = detr * win  # Tukey blend
    f_pad = pad_centered(taper, pad_to)  # zero-pad to power of two

    # Display spectra use the centred full FFT; the operator uses the equivalent rfft2.
    kx, ky, k = make_wavenumbers(pad_to, pad_to, dx, dy)  # full-grid |k|
    F = torch.fft.fft2(f_pad)
    Fk = F * k  # |k| = upward-derivative kernel
    out = crop_centered(
        torch.fft.irfft2(torch.fft.rfft2(f_pad) * k[:, : pad_to // 2 + 1], s=(pad_to, pad_to)), h, w
    )

    shift = torch.fft.fftshift
    return {
        "field": field.numpy(),
        "plane": plane.numpy(),
        "detr": detr.numpy(),
        "win": win.numpy(),
        "taper": taper.numpy(),
        "f_pad": f_pad.numpy(),
        "spec": shift(F.abs()).numpy(),
        "kernel": shift(k).numpy(),
        "spec_k": shift(Fk.abs()).numpy(),
        "out": out.numpy(),
        "coef": tuple(float(t) for t in plane_coef),
        "pad_to": pad_to,
    }


# --------------------------------------------------------------------------- #
# Drawing helpers (figure-fraction coordinates for full layout control)
# --------------------------------------------------------------------------- #
def add_panel(fig, rect, img, *, cmap, vmin, vmax, title, frame="0.35"):
    ax = fig.add_axes(rect)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(frame)
        s.set_linewidth(1.0)
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    return ax


def add_inset(fig, rect, img, *, cmap, vmin, vmax, label):
    ax = fig.add_axes(rect)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.5")
        s.set_linewidth(0.6)
    ax.set_title(label, fontsize=8, pad=2, color="0.25")
    return ax


def arrow(fig, x0, y0, x1, y1, label=None, *, lx=None, ly=None):
    fig.patches.append(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=1.6,
            color="0.2",
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        fig.text(
            lx if lx is not None else (x0 + x1) / 2,
            ly if ly is not None else (y0 + y1) / 2 + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="0.15",
        )


def draw_tiled(fig, rect, f_pad, *, cmap, vmin, vmax):
    """3x3 replication of the padded tile — the periodicity the DFT assumes."""
    tiled = np.tile(f_pad, (3, 3))
    ax = fig.add_axes(rect)
    n = f_pad.shape[0]
    ax.imshow(tiled, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    # grid lines between the nine copies; highlight the central (true) tile.
    for i in (1, 2):
        ax.axhline(i * n - 0.5, color="0.4", lw=0.7)
        ax.axvline(i * n - 0.5, color="0.4", lw=0.7)
    ax.add_patch(plt.Rectangle((n - 0.5, n - 0.5), n, n, fill=False, edgecolor="crimson", lw=1.6))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.35")
        s.set_linewidth(1.0)
    ax.set_title("periodic tiling → ∞  (padded)", fontsize=10, pad=4)
    return ax


def main(args: argparse.Namespace) -> None:
    field = load_patch_nt(args.dataset, args.idx)

    P = compute_pipeline(field, args.dx, args.dy, args.taper_frac)

    # colour scales — robust percentiles so structure isn't washed out by edge extremes.
    raw_lo, raw_hi = np.percentile(P["field"], [1, 99])
    dmax = float(np.percentile(np.abs(P["detr"]), 99))  # symmetric for detrend/taper/tiles
    omax = float(np.percentile(np.abs(P["out"]), 99))
    smax = np.log10(P["spec"].max() + 1e-9)
    smin = smax - 4  # 4 decades of spectral dynamic range
    logspec = np.log10(P["spec"] + 1e-9)
    logspec_k = np.log10(P["spec_k"] + 1e-9)
    DIV = "RdBu_r"

    fig = plt.figure(figsize=(17, 8.6))
    a, b, c = P["coef"]
    fig.suptitle(
        f"FVD physics-loss operator — {args.dataset.upper()} test patch #{args.idx}"
        f"     (first vertical derivative,  H(k)=|k|)",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    # ---- Row 1: spatial-domain preprocessing -----------------------------------
    y1, pw, ph = 0.55, 0.155, 0.30
    x_raw, x_detr, x_tap = 0.015, 0.235, 0.455
    add_panel(
        fig,
        [x_raw, y1, pw, ph],
        P["field"],
        cmap="gray",
        vmin=raw_lo,
        vmax=raw_hi,
        title="raw field  f  (nT)",
    )
    add_panel(
        fig, [x_detr, y1, pw, ph], P["detr"], cmap=DIV, vmin=-dmax, vmax=dmax, title="detrended  f − plane"
    )
    add_panel(
        fig, [x_tap, y1, pw, ph], P["taper"], cmap=DIV, vmin=-dmax, vmax=dmax, title="tapered  ×  w(x,y)"
    )
    # 3x3 periodic tile (wider panel closing the row)
    draw_tiled(fig, [0.675, y1, ph * 8.6 / 17, ph], P["f_pad"], cmap=DIV, vmin=-dmax, vmax=dmax)

    # arrows + operator insets on row 1. Insets sit centred in each gap, just above
    # the arrow (below the panel titles at y1+ph).
    wi = 0.05
    hi = wi * 17 / 8.6
    ay1 = y1 + ph / 2
    iy1 = ay1 + 0.005

    def gap_center(x_end, x_next):
        return (x_end + x_next) / 2 - wi / 2

    arrow(fig, x_raw + pw, ay1, x_detr, ay1, "subtract\nplane", ly=ay1 - 0.055)
    add_inset(
        fig,
        [gap_center(x_raw + pw, x_detr), iy1, wi, hi],
        P["plane"],
        cmap="gray",
        vmin=raw_lo,
        vmax=raw_hi,
        label=f"plane a·x+b·y+c\na={a:.2g}, b={b:.2g}",
    )
    arrow(fig, x_detr + pw, ay1, x_tap, ay1, "Tukey\nblend", ly=ay1 - 0.055)
    add_inset(
        fig,
        [gap_center(x_detr + pw, x_tap), iy1, wi, hi],
        P["win"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        label="Tukey  w(x,y)",
    )
    arrow(fig, x_tap + pw, ay1, 0.675, ay1, "zero-pad", ly=ay1 - 0.035)

    # ---- Row 2: frequency domain ----------------------------------------------
    y2 = 0.075
    x_spec, x_speck, x_out = 0.10, 0.40, 0.70
    add_panel(
        fig,
        [x_spec, y2, pw, ph],
        logspec,
        cmap="magma",
        vmin=smin,
        vmax=smax,
        title="FFT magnitude  log|F|",
    )
    add_panel(
        fig,
        [x_speck, y2, pw, ph],
        logspec_k,
        cmap="magma",
        vmin=smin,
        vmax=smax,
        title="|F| · |k|  (derivative spectrum)",
    )
    add_panel(
        fig,
        [x_out, y2, pw, ph],
        P["out"],
        cmap=DIV,
        vmin=-omax,
        vmax=omax,
        title="∂f/∂z  =  IFFT{ |F|·|k| }",
    )

    # transition arrow from the tiled panel down into the FFT
    ay2 = y2 + ph / 2
    arrow(fig, 0.72, y1 - 0.01, x_spec + pw / 2, y2 + ph + 0.005, "FFT", lx=0.47, ly=0.49)
    arrow(fig, x_spec + pw, ay2, x_speck, ay2, "×|k|\nkernel", ly=ay2 - 0.06)
    add_inset(
        fig,
        [gap_center(x_spec + pw, x_speck), ay2 + 0.005, wi, hi],
        P["kernel"],
        cmap="viridis",
        vmin=0,
        vmax=float(P["kernel"].max()),
        label="|k| upward\nderivative",
    )
    arrow(fig, x_speck + pw, ay2, x_out, ay2, "IFFT")

    out = args.out or (
        ROOT_FOLDER / "figures" / "fvd_schematic" / f"fvd_operator_schematic_{args.dataset}_{args.idx}.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # SVG keeps every arrow/frame/label vector; imshow panels embed as rasters at `dpi`.
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"plane coef: a={a:.4g}  b={b:.4g}  c={c:.4g}   pad_to={P['pad_to']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ksa", "wa"], default="ksa", help="Which test split to sample.")
    ap.add_argument("--idx", type=int, default=1037, help="Test-patch index (must be fully valid).")
    ap.add_argument(
        "--dx", type=float, default=1.0, help="pixel size in x (per-pixel wavenumber; scale only)."
    )
    ap.add_argument("--dy", type=float, default=1.0, help="pixel size in y.")
    ap.add_argument(
        "--taper-frac", type=float, default=0.15, help="Tukey taper fraction (operator default)."
    )
    ap.add_argument("--dpi", type=int, default=200, help="Raster dpi for the embedded imshow panels.")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (extension sets format; default figures/fvd_schematic/*.svg).",
    )
    main(ap.parse_args())

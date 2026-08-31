"""Narrow-column 2x2 schematic of the FVD (first-vertical-derivative) operator.

A single-column-friendly redraw of ``schematic_fvd_operator.py``: instead of the
wide 8-panel filmstrip, it shows the operator as a four-step clockwise loop with
the spatial domain on the left column and the frequency domain on the right:

    (1) detrend+taper+pad tile   --FFT-->   (2) |F|
                 ^                                  |
               IFFT                              x |k|
                 |                                  v
    (4) ∂f/∂z            <--IFFT--        (3) |F|·|k|

The two spectral panels (2) and (3) share ONE log-magnitude colour scale, so the
|k| high-pass boost reads directly: DC is nulled (dark centre) and the periphery
is amplified relative to (2). Panel (1) is the padded tile, so the taper feathering
into the zero guard ring is visible.

Reuses the *exact* operator compute (`compute_pipeline`) and dataset loader
(`load_patch_nt`) from the sibling ``schematic_fvd_operator`` module, so this is a
pure re-layout of the same code path — no operator logic is duplicated.

    uv run python experiments/plots/schematic_fvd_operator_2x2.py --dataset ksa --idx 1037

SVG by default so each panel/arrow/label imports into Inkscape as a separate object.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Sibling module (same directory) — reuse the operator compute + panel/arrow helpers.
from schematic_fvd_operator import add_panel, arrow, compute_pipeline, load_patch_nt

from magsr import ROOT_FOLDER

DIV = "RdBu_r"  # diverging map for the (signed) spatial panels
SPEC = "magma"  # sequential map for the log-magnitude spectra


def main(args: argparse.Namespace) -> None:
    field = load_patch_nt(args.dataset, args.idx)
    P = compute_pipeline(field, args.dx, args.dy, args.taper_frac)

    # Spatial colour scales — symmetric about 0 so zero renders neutral (white).
    dmax = float(np.percentile(np.abs(P["detr"]), 99))
    omax = float(np.percentile(np.abs(P["out"]), 99))

    # ONE shared log-magnitude scale across BOTH spectra (the point of the figure):
    # cover whichever panel has more energy, then show 4 decades of dynamic range.
    logspec = np.log10(P["spec"] + 1e-9)
    logspec_k = np.log10(P["spec_k"] + 1e-9)
    smax = float(max(logspec.max(), logspec_k.max()))
    smin = smax - 4.0

    TITLE_FS, ARROW_FS = 16, 16  # large enough to survive downscale to \columnwidth

    fig = plt.figure(figsize=(5.2, 5.4))

    # 2x2 grid in figure-fraction coords. Left column = spatial, right = frequency.
    pw = ph = 0.38
    xL, xR = 0.065, 0.545
    yT, yB = 0.515, 0.065

    def titled(rect, img, cmap, vmin, vmax, title):
        """add_panel imshow + a larger, explicitly-sized title."""
        ax = add_panel(fig, rect, img, cmap=cmap, vmin=vmin, vmax=vmax, title=None)
        ax.set_title(title, fontsize=TITLE_FS, pad=6)
        return ax

    # Operations live on the arrows (FFT, x|k|, IFFT); panels are titled by
    # content. Panel (3) is left untitled — it is manifestly |F| after x|k|,
    # and a title there would collide with the incoming arrow label.
    # (1) top-left: clean detrended + tapered tile (native size, no pad shown).
    titled([xL, yT, pw, ph], P["taper"], DIV, -dmax, dmax, "detrend + taper")
    # (2) top-right: FFT magnitude.
    titled([xR, yT, pw, ph], logspec, SPEC, smin, smax, r"$\log|F|$")
    # (3) bottom-right: filtered spectrum, SAME colour scale as (2) — the shared
    # colourbar (right) makes the |k| boost a like-for-like comparison.
    add_panel(fig, [xR, yB, pw, ph], logspec_k, cmap=SPEC, vmin=smin, vmax=smax, title=None)
    # (4) bottom-left: spatial output = IFFT of the filtered spectrum.
    titled([xL, yB, pw, ph], P["out"], DIV, -omax, omax, r"$\partial f/\partial z$")

    # One colourbar shared by BOTH spectra — visually proves the common scale.
    sm = ScalarMappable(norm=Normalize(vmin=smin, vmax=smax), cmap=SPEC)
    cax = fig.add_axes([xR + pw + 0.03, yB, 0.026, (yT + ph) - yB])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$\log|F|$  (shared)", fontsize=13)
    cb.ax.tick_params(labelsize=11)

    # Sequential arrows, clockwise: (1)->(2)->(3)->(4). Draw the arrow only and
    # place larger labels by hand for full control over size/position.
    m = 0.006  # small margin so arrowheads don't touch the panel frames
    yT_mid, yB_mid = yT + ph / 2, yB + ph / 2
    xR_mid = xR + pw / 2
    xmid_cols = (xL + pw + xR) / 2

    arrow(fig, xL + pw + m, yT_mid, xR - m, yT_mid, None)  # (1)->(2) FFT
    fig.text(xmid_cols, yT_mid + 0.028, "FFT", ha="center", va="bottom", fontsize=ARROW_FS, color="0.15")
    arrow(fig, xR_mid, yT - m, xR_mid, yB + ph + m, None)  # (2)->(3) x|k|
    fig.text(
        xR_mid + 0.022,
        (yT + yB + ph) / 2,
        r"$\times\,|k|$",
        ha="left",
        va="center",
        fontsize=ARROW_FS,
        color="0.15",
    )
    arrow(fig, xR - m, yB_mid, xL + pw + m, yB_mid, None)  # (3)->(4) IFFT
    fig.text(xmid_cols, yB_mid - 0.03, "IFFT", ha="center", va="top", fontsize=ARROW_FS, color="0.15")

    out = args.out or (
        ROOT_FOLDER / "figures" / "fvd_schematic" / f"fvd_operator_2x2_{args.dataset}_{args.idx}.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"pad_to={P['pad_to']}  shared log-|F| range=[{smin:.2f}, {smax:.2f}]")
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
        help="Output path (extension sets format; default figures/fvd_schematic/*_2x2_*.svg).",
    )
    main(ap.parse_args())

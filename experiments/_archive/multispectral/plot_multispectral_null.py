"""The multispectral negative result, as screen-then-test (no R^2, deliberately).

Multispectral has no physics route to helping -- Landsat reflectance does not enter the
upward-continuation operator, unlike the DEM (which sets the drape height). So the ONLY
way a spectral channel could help is by carrying information coupled to the magnetic
target. That makes a correlation ranking a *valid screen* for the best candidate (it is
NOT a proof of usefulness -- the DEM control showed correlation ranks channels backwards
when a physics route exists; but spectral data has none).

Left  -- rank every candidate channel/product by two couplings to the magnetic field:
         |Spearman| with the raw field, and structural-edge alignment (does the channel's
         gradient magnitude line up with the analytic signal, i.e. magnetic source edges).
         The ferrous ratio (SWIR1/NIR) is the single strongest candidate by BOTH -- and
         even it is weak (< 0.12).
Right -- we fed that best candidate to the network. It does not beat the mag-only
         baseline (73.7 vs 73.2 nT, fold-3 test) -- an extra channel to overfit, no signal.

Sources: results/multispectral/band_ranking_fullraster.csv (the two coupling columns);
the ferrous network numbers are the fold-3 stride-66 test grid, same split as Table XI.

Run:  uv run python experiments/plots/plot_multispectral_null.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magsr import ROOT_FOLDER

CTRL = "#1F4E79"
NULL = "#B5493A"
FIELD = "#8CB4CE"
GREY = "#9A9A9A"

RANK_CSV = ROOT_FOLDER / "results/multispectral/band_ranking_fullraster.csv"
# Pretty names for the raw channel ids in the CSV.
PRETTY = {
    "Ferrous_b6/b5": "Ferrous  SWIR1/NIR",
    "IronOxide_b4/b2": "Iron oxide  red/blue",
    "Clay_b6/b7": "Clay  SWIR1/SWIR2",
    "NDVI_b5b4": "NDVI",
    "b1_coastal": "coastal",
    "b2_blue": "blue",
    "b3_green": "green",
    "b4_red": "red",
    "b5_nir": "NIR",
    "b6_swir1": "SWIR1",
    "b7_swir2": "SWIR2",
    "PC1": "PC1 (brightness)",
    "PC2": "PC2",
    "PC3": "PC3",
}
# Same nb13 / k0 / s12 recipe; only the input channels differ. Fold-3 test.
NET = [("mag only\n(control)", 73.24), ("mag + ferrous ratio\n(strongest candidate)", 73.67)]


def load_ranking() -> list[tuple[str, float, float]]:
    """(label, |edge coupling|, |Spearman with field|), sorted by edge coupling."""
    out = []
    for r in csv.DictReader(open(RANK_CSV)):
        ch = r["channel"]
        out.append((PRETTY.get(ch, ch), abs(float(r["struct_edge"])), abs(float(r["spearman_hr"]))))
    return sorted(out, key=lambda t: t[1])  # ascending -> ferrous ends on top of barh


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT_FOLDER / "figures" / "talk" / "multispectral_null.png")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_ranking()
    labs = [r[0] for r in rows]
    edge = [r[1] for r in rows]
    spear = [r[2] for r in rows]
    y = np.arange(len(rows))
    is_ferrous = [l.startswith("Ferrous") for l in labs]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 6.6), width_ratios=[2.1, 1])

    # ---- left: rank candidates by two independent couplings; ferrous tops both.
    # Bars follow the two series colours everywhere (legend stays consistent); ferrous is
    # distinguished only by sitting on top of the sort, plus a faint neutral band.
    fi = is_ferrous.index(True)
    ax0.axhspan(y[fi] - 0.5, y[fi] + 0.5, color="#000000", alpha=0.05, zorder=0)
    h = 0.40
    ax0.barh(y + h / 2, edge, h, color=CTRL, edgecolor="white", label="gradient alignment")
    ax0.barh(y - h / 2, spear, h, color=FIELD, edgecolor="white", label="Spearman with magnetic anomaly")
    ax0.set_yticks(y, labs, fontsize=10)
    ax0.set_xlim(0, 0.125)
    ax0.set_xlabel("$|$correlation$|$ with the magnetic field\n(0 = none, dimensionless)", fontsize=11)
    ax0.legend(fontsize=10.5, frameon=False, loc="lower right")
    ax0.grid(axis="x", alpha=0.25, lw=0.6)
    ax0.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax0.spines[s].set_visible(False)

    # ---- right: feed the best candidate to the network -> no help
    x = np.arange(len(NET))
    nv = [v for _, v in NET]
    ax1.bar(x, nv, 0.55, color=[CTRL, FIELD], edgecolor="white", zorder=2)
    for xi, v in zip(x, nv):
        ax1.text(
            xi,
            v * 0.985,
            f"{v:.1f} nT",
            ha="center",
            va="top",
            color="white",
            fontsize=14,
            fontweight="bold",
            zorder=3,
        )
    ax1.set_xticks(x, [l for l, _ in NET], fontsize=10.5)
    ax1.set_ylim(0, max(nv) * 1.12)
    ax1.set_ylabel("Test RMSE (nT)  —  KSA fold 3", fontsize=11)
    ax1.grid(axis="y", alpha=0.25, lw=0.6)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.set_title(f"No gain  ($+${nv[1] - nv[0]:.1f} nT)", fontsize=13, fontweight="bold", loc="left")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  strongest candidate: {labs[fi]}  edge {edge[fi]:.3f}  |Spearman| {spear[fi]:.3f}")


if __name__ == "__main__":
    main()

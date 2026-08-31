"""Talk figures for the idealized forward-operator (synthetic UC) study.

Two figures, both read from the master eval CSV (results/ksa_ablations_eval.csv,
written by experiments/ksa_ablations/evaluate.py):

  error_budget.png   The same model, fold and patches on three LR sources: the real
                     survey, a drape-aware operator-consistent LR, and a flat
                     (terrain-blind) one. Error falls ~6x once the LR is consistent
                     with a known operator, so downward continuation is not the
                     bottleneck on KSA -- survey inconsistency is.

  modality_grid.png  The completed 2x6 grid: RMSE reduction from each auxiliary
                     channel, under the flat operator (no terrain coupling to
                     explain) and the drape operator (coupling present). The DEM
                     pays off ~10x more when the survey actually drapes, which is
                     the controlled evidence that its value is drape geometry and
                     not generic geological correlation.

Run:  uv run python experiments/plots/plot_synuc_study.py
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magsr import ROOT_FOLDER

FLAT = "#8CB4CE"  # terrain-blind operator: no coupling to recover
DRAPE = "#1F4E79"  # drape-aware operator: true terrain coupling injected
REAL = "#B5493A"  # real survey pairs
GREY = "#9A9A9A"

# Channel stacks in ablation order, with the labels used on the slides.
CHANNELS = [
    ("1vd", "+1VD"),
    ("demgrad", "+DEM-grad"),
    ("1vd_demgrad", "+1VD\n+DEM-grad"),
    ("demrelief", "+DEM-relief"),
    ("relief_1vd", "+DEM-relief\n+1VD"),
]


def load(path: Path) -> dict[tuple[str, str], dict]:
    """(operator, channels) -> {"rmse": nT test RMSE} from the master eval CSV.

    Synthetic rows come from the synuc grid (variant == "synuc"); real-survey rows from
    the real-LR eval of the matching fold-3 nb13 runs, with the checkpoint selected by
    argmin validation RMSE (the same rule make_tables.py uses).
    """
    D: dict[tuple[str, str], dict] = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["scope"] == "Net":
            D[(r["run"], r["variant"])][(r["split"], r["metric"])] = float(r["mean"])

    def real(run: str) -> dict:
        cands = {v: d for (rn, v), d in D.items() if rn == run and v != "last" and ("val", "rmse") in d}
        best = min(cands, key=lambda v: cands[v][("val", "rmse")])
        return {"rmse": cands[best][("test", "rmse")]}

    s12 = "rdnpp_x3_ksa_f3_nb13_k0_dwd2e-3_s12"
    rows = {
        ("real", "bicubic"): {"rmse": D[("bicubic_ksa_f3", "bicubic")][("test", "rmse")]},
        ("real", "mag"): real(s12),
        ("real", "1vd_demgrad_fvd5"): real(f"{s12}_1vd_demgrad_fvd5"),
    }
    for op, pfx in (("flat", "synuc"), ("drape", "synucd")):
        for stack in ("mag", *(k for k, _ in CHANNELS)):
            key = (f"rdnpp_x3_ksa_f3_nb13_k0_p264_{pfx}_{stack}", "synuc")
            if key in D:
                rows[(op, stack)] = {"rmse": D[key][("test", "rmse")]}
    return rows


def error_budget(rows: dict, out: Path) -> None:
    """Real vs operator-consistent LR: the gap that motivates the whole study."""
    bars = [
        ("Bicubic (real LR)", rows[("real", "bicubic")]["rmse"], GREY),
        ("Real LR, mag-only", rows[("real", "mag")]["rmse"], REAL),
        ("Real LR, best multi-modal", rows[("real", "1vd_demgrad_fvd5")]["rmse"], REAL),
        ("Synthetic drape LR, mag-only", rows[("drape", "mag")]["rmse"], DRAPE),
        ("Synthetic drape LR, best", rows[("drape", "relief_1vd")]["rmse"], DRAPE),
        ("Synthetic flat LR, mag-only", rows[("flat", "mag")]["rmse"], FLAT),
        ("Synthetic flat LR, best", rows[("flat", "relief_1vd")]["rmse"], FLAT),
    ]
    labels, vals, colors = zip(*bars)
    y = np.arange(len(bars))[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.barh(y, vals, color=colors, height=0.68, edgecolor="white", linewidth=0.8)
    for yi, v in zip(y, vals):
        ax.text(v + 1.2, yi, f"{v:.1f}", va="center", ha="left", fontsize=10.5, fontweight="bold")

    ax.set_yticks(y, labels, fontsize=10.5)
    ax.set_xlabel("Per-patch test RMSE (nT)  —  KSA fold 3, identical model and patches")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # Split the real-LR rows from the operator-consistent ones: the only thing that
    # changes across the divider is where the LR came from.
    ax.axhline(y[3] + 0.5, color="#333333", lw=1.0, ls=(0, (4, 3)))
    ax.text(
        103,
        y[1],
        "REAL\nSURVEY LR",
        fontsize=9,
        fontweight="bold",
        color=REAL,
        ha="center",
        va="center",
        rotation=90,
        clip_on=False,
        linespacing=1.4,
    )
    ax.text(
        103,
        y[5] + 0.5,
        "OPERATOR-\nCONSISTENT LR",
        fontsize=9,
        fontweight="bold",
        color=DRAPE,
        ha="center",
        va="center",
        rotation=90,
        clip_on=False,
        linespacing=1.4,
    )

    real = rows[("real", "mag")]["rmse"]
    syn = rows[("drape", "mag")]["rmse"]
    ax.text(
        22,
        y[5] + 0.35,
        f"Same model, same fold, same patches.\n"
        f"Only the LR source changes  →  {real / syn:.1f}x lower error.\n"
        f"The forward operator is not the bottleneck; survey inconsistency $\\epsilon$ is.",
        fontsize=10.5,
        color="#222222",
        va="center",
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.6", fc="#F2F5F8", ec="#C6D2DC", lw=0.9),
    )
    ax.set_title(
        "Downward continuation is not what limits MSR on KSA",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def modality_grid(rows: dict, out: Path) -> None:
    """The 2x6 controlled ablation: the DEM only pays when the survey drapes."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.8), width_ratios=[1.15, 1])

    # -- left: absolute RMSE, every input stack, both operators
    keys = ["mag"] + [k for k, _ in CHANNELS]
    names = ["mag only"] + [lab.replace("\n", " ") for _, lab in CHANNELS]
    x = np.arange(len(keys))
    w = 0.38
    fv = [rows[("flat", k)]["rmse"] for k in keys]
    dv = [rows[("drape", k)]["rmse"] for k in keys]
    ax0.bar(x - w / 2, fv, w, label="Flat operator", color=FLAT, edgecolor="white")
    ax0.bar(x + w / 2, dv, w, label="Draped operator", color=DRAPE, edgecolor="white")
    for xi, v in zip(x - w / 2, fv):
        ax0.text(xi, v + 0.12, f"{v:.2f}", ha="center", fontsize=8.5)
    for xi, v in zip(x + w / 2, dv):
        ax0.text(xi, v + 0.12, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax0.set_xticks(x, names, fontsize=9, rotation=18, ha="right")
    ax0.set_ylabel("Test RMSE (nT)")
    ax0.set_ylim(0, 14.2)
    ax0.legend(fontsize=9.5, frameon=False, loc="upper right")
    ax0.grid(axis="y", alpha=0.25, lw=0.6)
    ax0.set_axisbelow(True)
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)
    ax0.set_title("Every input stack, both operators", fontsize=11.5, fontweight="bold", loc="left")

    # -- right: the gain each channel buys, relative to its own mag-only baseline
    fb, db = rows[("flat", "mag")]["rmse"], rows[("drape", "mag")]["rmse"]
    xg = np.arange(len(CHANNELS))
    fg = [fb - rows[("flat", k)]["rmse"] for k, _ in CHANNELS]
    dg = [db - rows[("drape", k)]["rmse"] for k, _ in CHANNELS]
    ax1.bar(xg - w / 2, fg, w, label="Flat operator", color=FLAT, edgecolor="white")
    ax1.bar(xg + w / 2, dg, w, label="Draped operator", color=DRAPE, edgecolor="white")
    for xi, v in zip(xg - w / 2, fg):
        ax1.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.5)
    for xi, v in zip(xg + w / 2, dg):
        ax1.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(xg, [lab for _, lab in CHANNELS], fontsize=9)
    ax1.set_ylabel("RMSE reduction vs mag-only (nT)")
    ax1.set_ylim(0, 2.15)
    ax1.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax1.grid(axis="y", alpha=0.25, lw=0.6)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.set_title("What each channel actually buys", fontsize=11.5, fontweight="bold", loc="left")

    # 1VD is a deterministic filter of the mag input: it carries no terrain bits, and
    # indeed buys nothing under the operator where terrain is the thing to explain.
    ax1.annotate(
        "1VD alone: null under drape\n(a filter of the mag input —\ncarries no terrain information)",
        xy=(xg[0] + w / 2, dg[0]),
        xytext=(xg[0] + 0.05, 1.42),
        fontsize=8.8,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0),
    )
    ratio = dg[3] / fg[3]
    ax1.annotate(
        f"DEM-relief: {ratio:.0f}x more useful\nwhen the survey drapes",
        xy=(xg[3] + w / 2, dg[3] * 0.80),
        xytext=(xg[2] + 0.42, 1.86),
        fontsize=9.2,
        fontweight="bold",
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0),
    )

    fig.suptitle(
        "The DEM earns its place by explaining survey height, not geology",
        fontsize=13.5,
        fontweight="bold",
        x=0.008,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=ROOT_FOLDER / "results" / "ksa_ablations_eval.csv")
    p.add_argument("--outdir", type=Path, default=ROOT_FOLDER / "figures" / "synuc")
    args = p.parse_args()

    rows = load(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    error_budget(rows, args.outdir / "error_budget.png")
    modality_grid(rows, args.outdir / "modality_grid.png")
    print(f"wrote {args.outdir}/error_budget.png")
    print(f"wrote {args.outdir}/modality_grid.png")


if __name__ == "__main__":
    main()

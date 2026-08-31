"""Slide-ready results figures — replacements for the pasted LaTeX tables in the deck.

All read the ablation record on disk, so they cannot drift from it.

  cv_summary.png     The headline claim, made paired: the combined model over the full
                     5-fold cell-grid CV, and the fold-by-fold comparison against the
                     mag-only control that shows it wins on every fold (a table of means
                     cannot show that; a slope chart can).

  arch_transfer.png  The generalization claim: combo beats the mag-only baseline for both
                     backbones (RDN++, U-Net) on both surveys (KSA x3, WA x4), so the
                     finding is not an artifact of one architecture or one dataset.

  modality_ablation.png  Report Table XI as bars: what each input channel and the FVD loss
                     buys on fold 3, against the two dashed references (bicubic, RDN++
                     baseline). Physics-derived channels help, raw DEM / ANS do not.

The first two read results/ablation_summary.csv (one row per run, best variant); the third
reads the long-form per-variant evals, because the table pins a specific checkpoint per row.

Run:  uv run python experiments/plots/plot_talk_results.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from magsr import ROOT_FOLDER

COMBO = "#1F4E79"
CONTROL = "#B5493A"
BASE = "#8CB4CE"
GREY = "#9A9A9A"
# Qualitative palette for the per-fold slope lines (seaborn "deep", first five).
FOLD_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# The exact runs behind report Tables V, XII and XIII.
CV_STEMS = {
    "Combined model\n(nb13, 1VD+DEM-grad, FVD)": (
        "rdnpp_x3_nb13_k0_dwd2e-3_s12_1vd_demgrad_fvd5_cellgrid8",
        COMBO,
    ),
    "Mag-only control\n(same architecture)": ("rdnpp_x3_nb13_k0_dwd2e-3_s12_magonly_cellgrid8", CONTROL),
    "Paper baseline\n(RDN++ nb23)": ("rdnpp_x3_nb23_noadapt_cellgrid8", BASE),
    "Bicubic": ("bicubic_ksa", GREY),
}
ARCH = {
    "KSA Shield  ($\\times$3, fold 3)": {
        "RDN++": {
            "mag-only": "rdnpp_x3_nb23_noadapt_cellgrid8_f3",
            "multi-modal": "rdnpp_x3_nb13_k0_dwd2e-3_s12_1vd_demgrad_fvd5_cellgrid8_f3",
        },
        "U-Net": {
            "mag-only": "unet_x3_ksa_f3_s12_classcond_112",
            "multi-modal": "unet_x3_ksa_f3_s12_classcond_combo_112",
        },
        "bicubic": "bicubic_ksa_f3",
    },
    "WA Goldfields  ($\\times$4)": {
        "RDN++": {"mag-only": "rdnpp_x4_wa_baseline", "multi-modal": "rdnpp_x4_wa_1vd_demgrad_fvd5"},
        "U-Net": {
            "mag-only": "unet_x4_wa_5M_baseline_onecycle_pct30",
            "multi-modal": "unet_x4_wa_5M_combo_onecycle_pct30",
        },
        "bicubic": "bicubic_wa",
    },
}


# Report Table XI, row for row: (label, run, checkpoint variant). The variant matters —
# the table pins the best checkpoint per run, so we cannot read the summary CSV here.
S12 = "rdnpp_x3_ksa_f3_nb13_k0_dwd2e-3_s12"
MODAL = [
    ("1VD + DEM-grad\n+ FVD loss", f"{S12}_1vd_demgrad_fvd5", "soup_ssim_rmse"),
    ("FVD loss\n$\\lambda=50$", f"{S12}_fvd50", "best_rmse"),
    ("FVD loss\n$\\lambda=5$", f"{S12}_fvd5", "best_rmse"),
    ("1VD\n+ DEM-grad", f"{S12}_1vd_demgrad", "soup_ssim_rmse"),
    ("1VD", f"{S12}_1vd", "best"),
    ("DEM-grad", f"{S12}_demgrad", "best"),
    ("1VD + DEM-relief\n+ FVD loss", f"{S12}_1vd_demrelief_fvd5", "best_rmse"),
    ("ANS", f"{S12}_ans", "best_rmse"),
    ("1VD + ANS\n+ DEM", f"{S12}_1vd_ans_dem", "best"),
    ("1VD + ANS", f"{S12}_1vd_ans", "soup_ssim_rmse"),
    ("mag only\n(control)", S12, "best_rmse"),
    ("DEM\n(raw)", f"{S12}_dem", "best"),
]
# The dashed references. The baseline MUST be the cellgrid8 run, not `rdnpp_x3_ksa_f3_nb23`:
# the latter predates the x3 double-upsample stage (16,800,065 vs 16,836,993 params) and used
# wd=0 with val_every=5, so it is a different network on a coarser selection grid. It scores
# 72.8; the current baseline scores 73.5 and is the one in report Table IV and the 5-fold CV.
MODAL_REFS = {
    "baseline": (
        "RDN++ baseline (nb23)",
        "rdnpp_x3_nb23_noadapt_cellgrid8_f3",
        "soup_ssim_rmse",
        "#5A6B7B",
    ),
    "bicubic": ("bicubic", "bicubic_ksa_f3", "bicubic", GREY),
}

# Fold-3 gain attribution: bicubic -> published baseline -> our arch mag-only -> our arch + the
# modalities and physics loss. The baseline/control near-tie is the load-bearing result: the
# architecture and sampling changes buy efficiency, not accuracy, so the gain is the inputs+loss.
# Param counts read from the checkpoint configs (`num_params`).
ATTRIB = [
    ("Bicubic", "bicubic_ksa_f3", "bicubic", None, GREY),
    (
        "RDN++ baseline\n(published method)",
        "rdnpp_x3_nb23_noadapt_cellgrid8_f3",
        "soup_ssim_rmse",
        16_836_993,
        BASE,
    ),
    ("Our architecture\nmag-only (control)", S12, "best_rmse", 9_593_799, CONTROL),
    (
        "Our architecture\n+ modalities + physics loss",
        f"{S12}_1vd_demgrad_fvd5",
        "soup_ssim_rmse",
        9_595_527,
        COMBO,
    ),
]


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def metrics_by_run(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        r["run"]: {
            "rmse": float(r["test_rmse"]),
            "ssim": float(r["test_ssim"]),
            "msssim": float(r["test_msssim"]),
        }
        for r in rows
    }


def row_of(rows: list[dict], run: str) -> dict:
    return next(r for r in rows if r["run"] == run)


def baseline_wa_ksa(rows: list[dict], out: Path) -> None:
    """Report Table IV, as the contrast it actually is.

    The identical architecture, loss and recipe removes 61% of the bicubic error on WA
    and 10% on KSA. That gap is not a modelling failure -- WA's LR is *synthesized* from
    its own HR, so WA is image SR; KSA's LR is an independent survey, so KSA is true
    downward continuation and carries the inconsistency. Everything after this slide is
    a response to the right-hand panel.
    """
    panels = [
        (
            "WA Goldfields  ($\\times$4)",
            "bicubic_wa",
            "rdnpp_x4_wa_baseline",
            "LR synthesized from HR\n$\\Rightarrow$ image super-resolution",
        ),
        (
            "KSA Shield  ($\\times$3, fold 3)",
            "bicubic_ksa_f3",
            "rdnpp_x3_nb23_noadapt_cellgrid8_f3",
            "LR is an independent survey\n$\\Rightarrow$ true downward continuation",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    for ax, (title, bic_run, rdn_run, why) in zip(axes, panels):
        bic, rdn = row_of(rows, bic_run), row_of(rows, rdn_run)
        vals = [float(bic["test_rmse"]), float(rdn["test_rmse"])]
        ssim = [float(bic["test_ssim"]), float(rdn["test_ssim"])]
        drop = 100.0 * (vals[0] - vals[1]) / vals[0]

        x = [0, 1]
        ax.bar(x, vals, 0.55, color=[GREY, COMBO], edgecolor="white", zorder=2)
        # values inside the bars, so the space above them is free for the drop arrow
        for xi, v in zip(x, vals):
            ax.text(
                xi,
                v * 0.94,
                f"{v:.2f} nT",
                ha="center",
                va="top",
                fontsize=12.5,
                fontweight="bold",
                color="white",
                zorder=3,
            )

        # the headline: how much of the interpolation error the model actually removes
        ax.annotate(
            "",
            xy=(1, vals[1]),
            xytext=(1, vals[0]),
            arrowprops=dict(arrowstyle="-|>", color=CONTROL, lw=2.4, shrinkA=0, shrinkB=2),
        )
        ax.text(
            1.16,
            (vals[0] + vals[1]) / 2,
            f"$-${drop:.0f}%",
            fontsize=23,
            fontweight="bold",
            color=CONTROL,
            va="center",
        )

        ax.set_xticks(
            x, [f"bicubic\nSSIM {ssim[0]:.3f}", f"RDN++ baseline\nSSIM {ssim[1]:.3f}"], fontsize=10.5
        )
        ax.set_xlim(-0.55, 1.85)
        ax.set_ylim(0, vals[0] * 1.10)
        ax.set_ylabel("Test RMSE (nT)")
        # title + the reason the two panels are not comparable, stacked above the axes
        ax.text(0, 1.15, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold")
        ax.text(
            0,
            1.01,
            why,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#666666",
            style="italic",
            va="bottom",
            linespacing=1.45,
        )
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(
        "Same architecture, same loss, same recipe — and the error it removes collapses",
        fontsize=13.5,
        fontweight="bold",
        x=0.008,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}  (WA and KSA bicubic-error reduction)")


def folds_of(rows: list[dict], stem: str, metric: str = "test_rmse") -> dict[int, float]:
    out = {}
    for r in rows:
        if r["run"] == f"{stem}_f{r['fold']}":
            out[int(r["fold"])] = float(r[metric])
    return out


def cv_summary(rows: list[dict], out: Path) -> None:
    series = {lab: (folds_of(rows, stem), c) for lab, (stem, c) in CV_STEMS.items()}
    # 10% larger type throughout: bump the rc base so the default-sized axis ticks and axis
    # labels scale, and multiply every explicit fontsize below by 1.1.
    with plt.rc_context({"font.size": 11.0}):
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12.0, 5.0), width_ratios=[1.10, 0.7, 0.7])

        # -- left: 5-fold mean with the min-max spread across folds
        labs = list(series)
        means = [float(np.mean(list(series[l][0].values()))) for l in labs]
        los = [min(series[l][0].values()) for l in labs]
        his = [max(series[l][0].values()) for l in labs]
        cols = [series[l][1] for l in labs]
        y = np.arange(len(labs))[::-1]

        ax0.barh(y, means, color=cols, height=0.62, edgecolor="white", zorder=2)
        ax0.errorbar(
            means,
            y,
            xerr=[np.array(means) - los, np.array(his) - means],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.3,
            capsize=5,
            zorder=3,
        )
        for yi, m in zip(y, means):
            ax0.text(m + 2.0, yi + 0.30, f"{m:.1f}", va="center", fontsize=12.1, fontweight="bold")
        ax0.set_yticks(y, labs, fontsize=11)
        # Zoomed x-axis: the model means live in 66-70 nT, invisible on a 0-based scale.
        # Top is 84 (not 80) so bicubic's fold-max whisker at 81.7 and its label still fit.
        ax0.set_xlim(40, 84)
        ax0.set_xlabel("5-fold mean test RMSE (nT), bars span fold min-max")
        ax0.grid(axis="x", alpha=0.25, lw=0.6)
        ax0.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax0.spines[s].set_visible(False)
        ax0.set_title(
            "Full 5-fold cell-grid cross-validation", fontsize=12.65, fontweight="bold", loc="left"
        )

        # -- middle + right: the paired view, one panel per metric. The fold spread above is
        # wide enough to swallow the effect, so show it fold-by-fold instead: same cells,
        # same split, one change. SSIM sits parallel to RMSE so both metrics move together.
        combo_lab, ctrl_lab = labs[0], labs[1]
        # Colour encodes fold identity (replaces the per-line "fold N" text tags); the shared
        # legend below decodes it, and each endpoint value keeps its fold's colour so the
        # number ties back to its line even where lines cross.
        fids = sorted(folds_of(rows, CV_STEMS[combo_lab][0]))
        fold_colors = {f: FOLD_PALETTE[i % len(FOLD_PALETTE)] for i, f in enumerate(fids)}
        MEAN_C = "#2A2A2A"

        def paired(ax, metric: str, fmt: str, ylabel: str, lower_better: bool) -> int:
            combo = folds_of(rows, CV_STEMS[combo_lab][0], metric)
            ctrl = folds_of(rows, CV_STEMS[ctrl_lab][0], metric)
            for f in fids:
                c = fold_colors[f]
                ax.plot(
                    [0, 1], [ctrl[f], combo[f]], "-o", color=c, lw=1.8, ms=6, mfc="white", mec=c, zorder=2
                )
                ax.text(
                    1.06,
                    combo[f],
                    f"{combo[f]:{fmt}}",
                    ha="left",
                    va="center",
                    fontsize=9.9,
                    color=c,
                    fontweight="bold",
                )
            ax.plot(
                [0, 1],
                [np.mean([ctrl[f] for f in fids]), np.mean([combo[f] for f in fids])],
                "-",
                color=MEAN_C,
                lw=3.4,
                zorder=3,
            )
            ax.set_xticks([0, 1], ["control", "combined model"], fontsize=11.55)
            ax.set_xlim(-0.15, 1.5)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            return sum((combo[f] < ctrl[f]) if lower_better else (combo[f] > ctrl[f]) for f in fids)

        n_win = paired(ax1, "test_rmse", ".1f", "Test RMSE (nT)", lower_better=True)
        paired(ax2, "test_ssim", ".3f", "Test SSIM", lower_better=False)

        # One shared legend for both slope panels, along the bottom under them.
        handles = [
            Line2D(
                [0],
                [0],
                color=fold_colors[f],
                marker="o",
                mfc="white",
                mec=fold_colors[f],
                lw=1.8,
                ms=6,
                label=f"fold {f}",
            )
            for f in fids
        ]
        handles.append(Line2D([0], [0], color=MEAN_C, lw=3.4, label="5-fold mean"))
        fig.legend(
            handles=handles,
            loc="lower center",
            ncols=len(handles),
            frameon=False,
            fontsize=10.5,
            bbox_to_anchor=(0.70, 0.005),
        )

        combo_r = series[combo_lab][0]
        ctrl_r = series[ctrl_lab][0]
        fids = sorted(combo_r)
        gain = np.mean([ctrl_r[f] for f in fids]) - np.mean([combo_r[f] for f in fids])
        fig.suptitle(
            f"Multi-modal inputs + the physics loss buy {gain:.1f} nT — on every fold",
            fontsize=14.85,
            fontweight="bold",
            x=0.008,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.93))
        fig.savefig(out, dpi=200)
        plt.close(fig)
    print(f"wrote {out}   (5-fold gain {gain:.2f} nT, {n_win}/{len(fids)} folds)")


# The three metrics, one column each: (key, column title, value fmt, higher-is-better).
# RMSE and the two structural metrics live on different scales and point opposite ways, so
# they cannot share a y-axis — hence one small-multiple per metric rather than fatter bars.
ARCH_METRICS = [
    ("rmse", "RMSE (nT)  $\\downarrow$", ".2f", False),
    ("ssim", "SSIM  $\\uparrow$", ".3f", True),
]


def arch_transfer(rows: list[dict], out: Path) -> None:
    m = metrics_by_run(rows)
    datasets = list(ARCH.items())
    backbones = ["RDN++", "U-Net"]
    x = np.arange(len(backbones))
    w = 0.36
    fig, axes = plt.subplots(len(datasets), len(ARCH_METRICS), figsize=(9.6, 7.4))
    for i, (ds, spec) in enumerate(datasets):
        for j, (mk, title, fmt, higher) in enumerate(ARCH_METRICS):
            ax = axes[i, j]
            mag = [m[spec[b]["mag-only"]][mk] for b in backbones]
            mm = [m[spec[b]["multi-modal"]][mk] for b in backbones]
            bic = m[spec["bicubic"]][mk]
            ax.bar(x - w / 2, mag, w, label="mag-only", color=BASE, edgecolor="white", zorder=2)
            ax.bar(x + w / 2, mm, w, label="combined", color=COMBO, edgecolor="white", zorder=2)
            for xi, v in zip(x - w / 2, mag):
                ax.text(xi, v, f"{v:{fmt}}", ha="center", va="bottom", fontsize=8.5)
            for xi, v in zip(x + w / 2, mm):
                ax.text(xi, v, f"{v:{fmt}}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

            ax.axhline(bic, color=GREY, lw=1.3, ls=(0, (5, 3)), zorder=1)
            vals = mag + mm + [bic]
            if higher:
                # Structural metrics cluster near 1: a 0-based axis hides the effect, so zoom
                # to the data. The printed bar values keep the truncation honest. bicubic is
                # the floor, so its line runs low, behind the bars -- hence the gutter label.
                span = (max(vals) - min(vals)) or 1e-3
                ax.set_ylim(min(vals) - span * 0.30, max(vals) + span * 0.55)
            else:
                ax.set_ylim(0, bic * 1.16)  # zero-based: the honest scale for the headline nT
            # bicubic value on the line, in a right-hand gutter clear of every bar
            ax.set_xlim(-0.6, 2.0)
            ax.text(1.98, bic, f"bicubic {bic:{fmt}}", fontsize=8, color="#666666", va="bottom", ha="right")

            ax.set_xticks(x, backbones, fontsize=9.5)
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=11.5, fontweight="bold")
        axes[i, 0].set_ylabel(ds, fontsize=11, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=10.5,
        frameon=False,
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Combined inputs generalize — across both backbones and both surveys",
        fontsize=13.5,
        fontweight="bold",
        x=0.008,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def by_variant(paths: list[Path]) -> dict[tuple[str, str, str], float]:
    """Long-form eval rows -> {(run, variant, metric): test Net mean}."""
    out = {}
    for p in paths:
        for r in load(p):
            if r["scope"] == "Net" and r["split"] == "test":
                out[(r["run"], r["variant"], r["metric"])] = float(r["mean"])
    return out


def gain_attribution(ev: dict[tuple[str, str, str], float], out: Path) -> None:
    """Where the improvement actually comes from — four bars, fold 3.

    Answers the question every committee asks: is the gain just a better architecture? No.
    The control has our architecture with one channel and scores the same as the published
    baseline at 43% of the parameters. The gain arrives only with the modalities and the loss.
    """
    vals = [ev[(run, var, "rmse")] for _, run, var, _, _ in ATTRIB]
    labs = [lab for lab, *_ in ATTRIB]
    pars = [p for *_, p, _ in ATTRIB]
    cols = [c for *_, c in ATTRIB]
    x = np.arange(len(ATTRIB))

    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    ax.bar(x, vals, 0.62, color=cols, edgecolor="white", zorder=2)
    # value + param count both inside the bar, leaving the band above the bars free for
    # the two annotations that carry the argument
    for xi, v, p in zip(x, vals, pars):
        ax.text(
            xi,
            v * 0.955,
            f"{v:.1f} nT",
            ha="center",
            va="top",
            color="white",
            fontsize=13,
            fontweight="bold",
            zorder=3,
        )
        if p:
            ax.text(xi, 2.4, f"{p / 1e6:.1f} M params", ha="center", fontsize=9.5, color="white", zorder=3)

    # (1) baseline -> control: the architecture change is free. Same score, far smaller.
    shrink = 100 * (1 - pars[2] / pars[1])
    y1 = vals[0] * 0.955
    ax.annotate("", xy=(2, y1), xytext=(1, y1), arrowprops=dict(arrowstyle="<->", color="#5A6B7B", lw=1.6))
    ax.text(
        1.5,
        y1 + 1.2,
        f"same score, {shrink:.0f}% fewer parameters\n" "the architecture buys efficiency, not accuracy",
        ha="center",
        fontsize=10,
        color="#5A6B7B",
        fontweight="bold",
        linespacing=1.4,
    )

    # (2) control -> champion: this is the whole gain, and it is the inputs + the loss.
    gain = vals[2] - vals[3]
    ax.annotate(
        "",
        xy=(3, vals[3]),
        xytext=(3, vals[2]),
        arrowprops=dict(arrowstyle="-|>", color=COMBO, lw=2.6, shrinkA=0, shrinkB=2),
    )
    ymid = (vals[2] + vals[3]) / 2
    ax.text(
        3.36,
        ymid + 1.0,
        f"$-${gain:.1f} nT",
        fontsize=21,
        fontweight="bold",
        color=COMBO,
        va="center",
        ha="left",
    )
    ax.text(
        3.36,
        ymid - 3.4,
        "the gain is the conditioning\ninputs and the physics loss",
        fontsize=9.5,
        color=COMBO,
        va="center",
        ha="left",
        linespacing=1.4,
    )

    ax.set_xticks(x, labs, fontsize=10.5)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(0, vals[0] * 1.16)
    ax.set_ylabel("Test RMSE (nT)  —  KSA fold 3")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        "Is the gain just a better architecture?  No — the control says so",
        fontsize=13.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}  (control {vals[2]:.1f} vs baseline {vals[1]:.1f}; gain {gain:.1f} nT)")


def modality_ablation(ev: dict[tuple[str, str, str], float], out: Path) -> None:
    ref = {
        k: {m: ev[(run, var, m)] for m in ("rmse", "ssim")} for k, (_, run, var, _) in MODAL_REFS.items()
    }
    base, bic = ref["baseline"], ref["bicubic"]

    # Worst first: the eye reads left to right as "raw DEM -> the combined model", so the
    # bars descend toward the champion instead of away from it.
    bars = sorted(
        ((lab, ev[(run, var, "rmse")], ev[(run, var, "ssim")]) for lab, run, var in MODAL),
        key=lambda b: b[1],
        reverse=True,
    )
    labs = [b[0] for b in bars]
    x = np.arange(len(bars))
    champ = len(bars) - 1  # champion is now the rightmost bar
    # Champion dark; anything that beats the reproduced baseline blue; the rest red.
    cols = [
        COMBO if i == champ else (BASE if r < base["rmse"] else CONTROL) for i, (_, r, _) in enumerate(bars)
    ]

    # SSIM on top, RMSE below: the model tags then sit directly under the RMSE bars, which
    # are the ones the audience reads.
    fig, (ax_s, ax_r) = plt.subplots(2, 1, figsize=(15.6, 7.6), sharex=True, height_ratios=[1, 1.15])

    # Right-hand margin (in bar units) that the reference labels live in, clear of every bar.
    # The bars keep the same per-bar width they had before the gutter was added, so the tick
    # labels below them still fit.
    REF_GUTTER = 2.6
    X0, X1 = -0.7, len(bars) - 0.4  # span the dashed lines run over, ending inside the gutter

    def refs(ax, key: str, fmt: str) -> None:
        # Line stops short of its own label, which sits past the last bar: nothing overlaps a
        # bar, and the eye reads the dashed line straight into its name.
        for k, (name, _, _, c) in MODAL_REFS.items():
            v = ref[k][key]
            ax.plot([X0, X1], [v, v], color=c, lw=0.9, ls=(0, (6, 4)), alpha=0.55, zorder=1)
            ax.text(
                X1 + 0.18,
                v,
                f"{name} {v:{fmt}}",
                fontsize=9.5,
                color=c,
                va="center",
                ha="left",
                fontweight="bold",
                alpha=0.85,
            )

    # -- top: SSIM (higher better). Truncated axis — 0.85 vs 0.87 is invisible on 0-1 — so
    # the bars carry a break mark and the ylabel says where the axis starts.
    v = [b[2] for b in bars]
    lo = 0.830
    ax_s.bar(x, [vi - lo for vi in v], 0.68, bottom=lo, color=cols, edgecolor="white", zorder=2)
    for xi, vi in zip(x, v):
        ax_s.text(
            xi,
            vi + 0.0007,
            f"{vi:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if xi == champ else "normal",
        )
    refs(ax_s, "ssim", ".3f")
    ax_s.set_ylim(lo, 0.872)
    # The ylabel is rotated 90 deg, so the arrow glyph rotates with it: -> reads as "up".
    ax_s.set_ylabel("Test SSIM $\\rightarrow$\n(axis starts at 0.83)")
    for xy in (0, 1):  # break marks: the bars do not start at zero
        ax_s.plot(
            [xy - 0.006, xy + 0.006],
            [-0.016, 0.016],
            transform=ax_s.transAxes,
            color="black",
            lw=1.1,
            clip_on=False,
            zorder=5,
        )

    # -- bottom: RMSE, zero-based (the honest scale; the dashed refs carry the comparison)
    v = [b[1] for b in bars]
    ax_r.bar(x, v, 0.68, color=cols, edgecolor="white", zorder=2)
    for xi, vi in zip(x, v):
        ax_r.text(
            xi,
            vi + 0.6,
            f"{vi:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if xi == champ else "normal",
        )
    refs(ax_r, "rmse", ".1f")
    ax_r.set_ylim(0, bic["rmse"] * 1.12)
    ax_r.set_ylabel("Test RMSE (nT) $\\leftarrow$")  # rotated: <- reads as "down"

    for ax in (ax_s, ax_r):
        ax.set_xlim(X0, len(bars) - 0.3 + REF_GUTTER)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_r.set_xticks(x, labs, fontsize=9.5)
    ax_r.tick_params(axis="x", length=0)

    hand = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COMBO, BASE, CONTROL)]
    # Colour is fixed by the RMSE ranking, so the SSIM panel can (and does) disagree — raw
    # DEM lifts SSIM while it costs RMSE. Say so in the legend rather than hide it.
    fig.legend(
        hand,
        ["combined model", "beats the RDN++ baseline on RMSE", "no better than baseline on RMSE"],
        fontsize=10,
        frameon=False,
        loc="upper left",
        ncols=3,
        bbox_to_anchor=(0.008, 0.995),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}   (champion {bars[champ][1]:.1f} nT vs baseline {base['rmse']:.1f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=ROOT_FOLDER / "results" / "ablation_summary.csv")
    p.add_argument(
        "--eval-csv",
        type=Path,
        nargs="+",
        default=[
            ROOT_FOLDER / "results" / "all_models_eval.csv",
            ROOT_FOLDER / "results" / "_demrelief.csv",
        ],
    )
    p.add_argument("--outdir", type=Path, default=ROOT_FOLDER / "figures" / "talk")
    args = p.parse_args()

    rows = load(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    baseline_wa_ksa(rows, args.outdir / "baseline_wa_ksa.png")
    cv_summary(rows, args.outdir / "cv_summary.png")
    arch_transfer(rows, args.outdir / "arch_transfer.png")
    ev = by_variant(args.eval_csv)
    gain_attribution(ev, args.outdir / "gain_attribution.png")
    modality_ablation(ev, args.outdir / "modality_ablation.png")


if __name__ == "__main__":
    main()

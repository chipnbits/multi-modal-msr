"""Wrong-head robustness: compare wrong-head vs correct-head, soup vs base.

Joins the wrong-head passes (from the now-removed wrong-head eval driver, which
wrote results/cv_folds_wronghead_eval/<prefix>wronghead.csv) against the existing
correct-head (in-domain) metrics from experiments/eval_cv_folds.py:
    base variants  -> results/cv_folds_eval/<prefix>inference.csv
    soup variants  -> results/cv_folds_soups_eval/<prefix>inference.csv

For each (fold, variant, split, scope, metric) the two wrong-head passes are
averaged into a single wrong-head value, then compared to the correct-head value
of the SAME model on the SAME patches. Robustness degradation:
    pct = 100 * (wrong - correct) / correct
(RMSE: positive = worse; SSIM/MS-SSIM: negative = worse). Smaller magnitude =
more robust to using the wrong domain head. Aggregated as the cross-fold mean with
[min, max] range; plots overlay each of the 5 folds as a tick rather than an std
error bar (K-fold CV has no unbiased variance estimator, Bengio & Grandvalet 2004).

Outputs (results/cv_folds_wronghead_eval/plots/):
    robustness_net.png   - overall (Net) %-degradation per variant, by split.
    robustness_block.png - per-block (B1/B2/B3) %-degradation, by split.
and prints the summary tables.

Usage:
    uv run python experiments/plots/plot_wronghead_robustness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # noqa: E402

    sns.set_theme(style="whitegrid", context="notebook")
    _PAL = sns.color_palette("deep")
    _BLUE, _RED, _PURPLE = _PAL[0], _PAL[3], _PAL[4]
except ImportError:
    # seaborn absent: replicate its "deep" palette + whitegrid look via matplotlib.
    for _s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if _s in plt.style.available:
            plt.style.use(_s)
            break
    _BLUE, _RED, _PURPLE = "#4C72B0", "#C44E52", "#8172B3"  # seaborn "deep" idx 0/3/4

# Horizontal (y) gridlines only — no vertical x gridlines.
matplotlib.rcParams["axes.grid.axis"] = "y"

from magsr import ROOT_FOLDER  # noqa: E402

VARIANTS = ["best", "soup_ssim_rmse", "best_rmse"]  # soup in the middle (it's a mix of the two)
VLABEL = {
    "best": "Best SSIM model",
    "best_rmse": "Best RMSE model",
    "soup_ssim_rmse": "RMSE-SSIM soup model",
}
VCOLOR = {"best": _BLUE, "best_rmse": _RED, "soup_ssim_rmse": _PURPLE}  # blue / red / purple (seaborn deep)
SPLITS = ["train", "val", "test"]


def load_correct(base_csv: Path, soup_csv: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(base_csv), pd.read_csv(soup_csv)], ignore_index=True)
    df = df[(df["method"] == "RDN++") & (df["variant"].isin(VARIANTS))]
    return df[["fold", "variant", "split", "scope", "metric", "mean"]].rename(columns={"mean": "correct"})


def load_wrong(wrong_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(wrong_csv)
    df = df[df["variant"].isin(VARIANTS)]
    # Average the two wrong-head passes (wrongA/wrongB) per cell.
    g = (
        df.groupby(["fold", "variant", "split", "scope", "metric"])["mean"]
        .mean()
        .reset_index()
        .rename(columns={"mean": "wrong"})
    )
    return g


def build(wrong_csv: Path, base_csv: Path, soup_csv: Path) -> pd.DataFrame:
    m = load_wrong(wrong_csv).merge(
        load_correct(base_csv, soup_csv), on=["fold", "variant", "split", "scope", "metric"], how="inner"
    )
    m["pct"] = 100.0 * (m["wrong"] - m["correct"]) / m["correct"]
    return m


def agg(df: pd.DataFrame, scope_sel) -> pd.DataFrame:
    sub = (
        df[df["scope"].isin(scope_sel)]
        if isinstance(scope_sel, (list, tuple))
        else df[df["scope"] == scope_sel]
    )
    # Cross-fold mean + [min, max] range (NOT std: K-fold CV has no unbiased variance
    # estimator, Bengio & Grandvalet 2004). Plots overlay each fold as a tick.
    return (
        sub.groupby(["variant", "split", "scope", "metric"])
        .agg(
            pct_m=("pct", "mean"),
            pct_lo=("pct", "min"),
            pct_hi=("pct", "max"),
            correct_m=("correct", "mean"),
            wrong_m=("wrong", "mean"),
        )
        .reset_index()
    )


def _fold_ticks(ax, xc, per):
    """Overlay each fold's |Δ%| as a tick at x=xc (n=5; honest small-n spread, no std)."""
    if len(per):
        ax.plot(
            [xc] * len(per),
            np.abs(per.to_numpy(float)),
            marker="_",
            linestyle="none",
            color="#222222",
            markersize=7,
            markeredgewidth=1.1,
            zorder=5,
        )


def _grouped_bars(ax, a, df, metric, scope="Net"):
    """Grouped bars: x = train/val/test groups, 3 variant bars each (one axes ->
    one shared y-scale across the splits). Plots |Δ%| so bars point up; each fold
    overlaid as a tick instead of an std error bar."""
    am = a[a["metric"] == metric]
    x = np.arange(len(SPLITS))
    w = 0.22
    for i, v in enumerate(VARIANTS):
        ys = []
        for s_i, split in enumerate(SPLITS):
            row = am[(am["split"] == split) & (am["variant"] == v)]
            ys.append(abs(float(row["pct_m"].iloc[0])) if len(row) else 0.0)
            per = df[
                (df["scope"] == scope)
                & (df["split"] == split)
                & (df["variant"] == v)
                & (df["metric"] == metric)
            ]["pct"]
            _fold_ticks(ax, s_i + (i - 1) * w, per)
        ax.bar(x + (i - 1) * w, ys, w, color=VCOLOR[v], label=VLABEL[v])
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)


def plot_net(df: pd.DataFrame, out: Path) -> None:
    a = agg(df, "Net")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), squeeze=False)
    for c, metric in enumerate(("rmse", "ssim")):
        ax = axes[0][c]
        _grouped_bars(ax, a, df, metric, "Net")
        ax.set_title(f"{metric.upper()} — Net wrong-head |Δ%|", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} wrong-head |Δ%| (vs correct head)")
    axes[0][0].legend(fontsize=9, title="variant")
    fig.suptitle(
        "Wrong-head robustness (overall/Net): Δ% vs correct head, "
        "bar=mean, ticks=each of 5 folds  —  smaller |Δ| = more robust",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_block(df: pd.DataFrame, out: Path, metric: str = "rmse") -> None:
    blocks = ["B1", "B2", "B3"]
    a = agg(df, blocks)
    a = a[a["metric"] == metric]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), squeeze=False)
    x = np.arange(len(blocks))
    w = 0.22
    for c, split in enumerate(SPLITS):
        ax = axes[0][c]
        for i, v in enumerate(VARIANTS):
            ys = []
            for b_i, b in enumerate(blocks):
                row = a[(a["split"] == split) & (a["variant"] == v) & (a["scope"] == b)]
                ys.append(abs(float(row["pct_m"].iloc[0])) if len(row) else 0.0)
                per = df[
                    (df["scope"] == b)
                    & (df["split"] == split)
                    & (df["variant"] == v)
                    & (df["metric"] == metric)
                ]["pct"]
                _fold_ticks(ax, b_i + (i - 1) * w, per)
            ax.bar(x + (i - 1) * w, ys, w, color=VCOLOR[v], label=VLABEL[v])
        ax.set_xticks(x)
        ax.set_xticklabels(blocks)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{split}", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"{metric.upper()} wrong-head |Δ%|")
            ax.legend(fontsize=8)
    fig.suptitle(
        f"Wrong-head robustness per block ({metric.upper()}): |Δ%| vs "
        "correct head, bar=mean, ticks=each of 5 folds",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="rdnpp_x3_16mp_cellgrid8_f")
    p.add_argument("--results-root", type=Path, default=ROOT_FOLDER / "results")
    args = p.parse_args()

    rr = args.results_root
    wrong_csv = rr / "cv_folds_wronghead_eval" / f"{args.prefix}wronghead.csv"
    base_csv = rr / "cv_folds_eval" / f"{args.prefix}inference.csv"
    soup_csv = rr / "cv_folds_soups_eval" / f"{args.prefix}inference.csv"
    for f in (wrong_csv, base_csv, soup_csv):
        if not f.exists():
            raise SystemExit(f"missing input: {f}")

    df = build(wrong_csv, base_csv, soup_csv)
    out_dir = rr / "cv_folds_wronghead_eval" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_net(df, out_dir / "robustness_net.png")
    plot_block(df, out_dir / "robustness_block.png")
    print(f"Wrote plots to {out_dir}")

    # Console summary: Net %-degradation, RMSE + SSIM, per split x variant.
    net = agg(df, "Net")
    for metric in ("rmse", "ssim"):
        print(f"\n=== Net wrong-head Δ% ({metric.upper()}), mean [min, max] over 5 folds ===")
        print(f"  {'split':>5}  " + "  ".join(f"{VLABEL[v]:>22}" for v in VARIANTS))
        for split in SPLITS:
            cells = []
            for v in VARIANTS:
                row = net[(net["split"] == split) & (net["variant"] == v) & (net["metric"] == metric)]
                if len(row):
                    cells.append(
                        f"{row['pct_m'].iloc[0]:+.2f} [{row['pct_lo'].iloc[0]:+.2f},"
                        f"{row['pct_hi'].iloc[0]:+.2f}]"
                    )
                else:
                    cells.append("--")
            print(f"  {split:>5}  " + "  ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()

"""Summarize the cellgrid8 CV soup analysis: cross-fold Net RMSE / SSIM for the
bicubic baseline, the three single checkpoints (best/best_rmse/last) and the
three 50/50 soups, on the val and test splits.

Aggregation: each fold contributes one per-fold Net mean (from the inference
CSVs); we report the mean over the 5 folds with the cross-fold std as the error
bar / the +- term — i.e. how the benchmark varies across folds.

Inputs:
    results/cv_folds_eval/<prefix>inference.csv         bicubic + best/best_rmse/last
    results/cv_folds_soups_eval/<prefix>inference.csv   the 3 soups

Outputs (results/cv_folds_soups_eval/plots/):
    soup_summary_bars.png   RMSE + SSIM panels, val/test bars, cross-fold error bars
    soup_summary.tex        LaTeX table, mean +- std at 3 significant figures

Usage:
    uv run python experiments/plots/plot_soup_summary.py
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
except ImportError:
    pass
matplotlib.rcParams["axes.grid.axis"] = "y"  # horizontal gridlines only

from magsr import ROOT_FOLDER  # noqa: E402

# Display order and labels for every variant in the summary.
VARIANTS = ["bicubic", "best", "best_rmse", "last", "soup_ssim_rmse", "soup_ssim_last", "soup_rmse_last"]
VLABEL = {
    "bicubic": "Bicubic",
    "best": "best (val SSIM)",
    "best_rmse": "best (val RMSE)",
    "last": "last",
    "soup_ssim_rmse": "soup (SSIM+RMSE)",
    "soup_ssim_last": "soup (SSIM+last)",
    "soup_rmse_last": "soup (RMSE+last)",
}
SPLIT_COLOR = {"val": "#4C72B0", "test": "#55A868"}  # blue / green
METRICS = ("rmse", "ssim")
MUNIT = {"rmse": "Net RMSE (nT)", "ssim": "Net SSIM"}


def load_all(prefix: str) -> pd.DataFrame:
    """Concatenate the main fold CSV and the soups CSV, Net scope only."""
    main = ROOT_FOLDER / "results" / "cv_folds_eval" / f"{prefix}inference.csv"
    soups = ROOT_FOLDER / "results" / "cv_folds_soups_eval" / f"{prefix}inference.csv"
    frames = [pd.read_csv(main)]
    if soups.exists():
        frames.append(pd.read_csv(soups))
    df = pd.concat(frames, ignore_index=True)
    return df[df.scope == "Net"].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-fold mean/std per (variant, split, metric) from the per-fold means."""
    rows = []
    for variant in VARIANTS:
        for split in ("val", "test"):
            for metric in METRICS:
                d = df[(df.variant == variant) & (df.split == split) & (df.metric == metric)]
                if d.empty:
                    continue
                vals = d["mean"].to_numpy(float)
                # Report mean + [min, max] range across folds, NOT std: K-fold CV has no
                # unbiased variance estimator (Bengio & Grandvalet 2004), so a cross-fold
                # std / standard error would be statistically unfounded.
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "metric": metric,
                        "mean": vals.mean(),
                        "lo": vals.min(),
                        "hi": vals.max(),
                        "n": len(vals),
                    }
                )
    return pd.DataFrame(rows)


# Decimals per metric so the mean reads as ~3 significant figures with trailing
# zeros kept (RMSE ~70 nT -> 1 dp = 70.3; SSIM ~0.86 -> 3 dp = 0.865); the range
# uses the same decimals so the [min, max] lines up with the mean.
METRIC_DP = {"rmse": 1, "ssim": 3}


def _cell(mean: float, lo: float, hi: float, dp: int) -> str:
    """mean with [min, max] cross-fold range, all at `dp` decimal places."""
    return f"{mean:.{dp}f} [{lo:.{dp}f}, {hi:.{dp}f}]"


def write_latex(summ: pd.DataFrame, out: Path) -> str:
    def get(variant, split, metric):
        r = summ[(summ.variant == variant) & (summ.split == split) & (summ.metric == metric)]
        return None if r.empty else (r["mean"].iloc[0], r["lo"].iloc[0], r["hi"].iloc[0])

    # Best (bolded) cell per column: RMSE min / SSIM max, excluding bicubic.
    best = {}
    for metric in ("rmse", "ssim"):
        for split in ("val", "test"):
            cand = [
                (get(v, split, metric)[0], v)
                for v in VARIANTS
                if v != "bicubic" and get(v, split, metric) is not None
            ]
            if cand:
                best[(metric, split)] = (min if metric == "rmse" else max)(cand)[1]

    n = int(summ["n"].max())
    body = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{RMSE (nT)} & \\multicolumn{2}{c}{SSIM} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Method & Val & Test & Val & Test \\\\",
        "\\midrule",
    ]
    for v in VARIANTS:
        cells = []
        for metric in ("rmse", "ssim"):
            for split in ("val", "test"):
                g = get(v, split, metric)
                if g is None:
                    cells.append("$-$")
                    continue
                c = _cell(g[0], g[1], g[2], METRIC_DP[metric])
                if best.get((metric, split)) == v:
                    c = f"\\textbf{{{c}}}"
                cells.append(c)
        if all(c == "$-$" for c in cells):
            continue
        body.append(f"{VLABEL[v]} & " + " & ".join(cells) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]

    # Wrap as a complete, compilable standalone document.
    lines = [
        "% cellgrid8 CV soup analysis -- cross-fold mean with [min, max] range (Net scope).",
        "% Compile with pdflatex; self-contained.",
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage[margin=1in]{geometry}",
        "\\begin{document}",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Cellgrid8 5-fold CV soup analysis: cross-fold mean with "
        f"[min, max] range (Net scope, $n={n}$ folds). Range, not std: K-fold CV "
        "has no unbiased variance estimator (Bengio \\& Grandvalet 2004).}}",
        "\\label{tab:soup-summary}",
        *body,
        "\\end{table}",
        "\\end{document}",
    ]
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    return text


def plot_bars(summ: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    variants = [v for v in VARIANTS if not summ[summ.variant == v].empty]
    x = np.arange(len(variants))
    w = 0.26  # narrow bars
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    for c, metric in enumerate(METRICS):
        ax = axes[0][c]
        means_by_split = {}
        for i, split in enumerate(("val", "test")):
            means = []
            for j, v in enumerate(variants):
                r = summ[(summ.variant == v) & (summ.split == split) & (summ.metric == metric)]
                m = float(r["mean"].iloc[0]) if not r.empty else np.nan
                means.append(m)
                # Overlay EVERY fold's value as a tick (n=5) instead of an std/SE error
                # bar — K-fold CV has no unbiased variance estimator (Bengio 2004), so the
                # 5 points are the honest summary.
                per = df[(df.variant == v) & (df.split == split) & (df.metric == metric)]["mean"]
                xc = x[j] + (i - 0.5) * w
                ax.plot(
                    [xc] * len(per),
                    per.to_numpy(float),
                    marker="_",
                    linestyle="none",
                    color="#222222",
                    markersize=7,
                    markeredgewidth=1.2,
                    zorder=5,
                )
            ax.bar(x + (i - 0.5) * w, means, w, color=SPLIT_COLOR[split], label=split)
            means_by_split[split] = means
        ax.set_xticks(x)
        ax.set_xticklabels([VLABEL[v] for v in variants], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(MUNIT[metric])
        ax.set_title(f"{metric.upper()} across 5 folds (ticks = each fold, bar = mean)", fontsize=10)
        ax.legend(title="split")
        if metric == "ssim":  # bounded near 1 — zoom so differences are visible
            vals = df[df.metric == "ssim"]["mean"].to_numpy(float)
            lo, hi = vals.min(), vals.max()
            pad = (hi - lo) * 0.15 + 1e-4
            ax.set_ylim(lo - pad, hi + pad)

        # Highlight the best model by test split (RMSE: min, SSIM: max),
        # excluding bicubic, by bolding its x-axis tick label.
        test = means_by_split["test"]
        cand = [(t, j) for j, (v, t) in enumerate(zip(variants, test)) if v != "bicubic" and np.isfinite(t)]
        best_j = (min if metric == "rmse" else max)(cand)[1]
        ax.get_xticklabels()[best_j].set_fontweight("bold")
    fig.suptitle("Cellgrid8 5-fold CV — soup analysis summary (Net scope)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="rdnpp_x3_16mp_cellgrid8_f")
    args = p.parse_args()

    df = load_all(args.prefix)
    summ = summarize(df)
    out_dir = ROOT_FOLDER / "results" / "cv_folds_soups_eval" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_bars(summ, df, out_dir / "soup_summary_bars.png")
    tex = write_latex(summ, out_dir / "soup_summary.tex")

    print(f"Wrote {out_dir}/soup_summary_bars.png")
    print(f"Wrote {out_dir}/soup_summary.tex\n")
    print(tex)


if __name__ == "__main__":
    main()

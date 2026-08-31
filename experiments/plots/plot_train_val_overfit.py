"""Training vs validation loss for the two no-adapter baselines: WA and KSA.

Shows the overfitting contrast — WA generalizes (val tracks train), KSA overfits (val
bottoms out early then rises while train keeps dropping). Masked L1 loss, per epoch.
Two-panel figure (side by side); the datasets live on ~14x different loss scales, so
each panel keeps its own y-axis. Also saves each panel as a standalone image.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from magsr import ROOT_FOLDER

ROOT = ROOT_FOLDER
OUT = ROOT / "figures/overfit"
OUT.mkdir(parents=True, exist_ok=True)

# Typography is sized for a HALF-COLUMN slot: the canvas is authored at ~2x the
# final print width, so every point size is scaled up by the same factor and lands
# back at a readable ~7-8 pt after \includegraphics scales it down.
SCALE = 1.6

# Academic style: serif typography, restrained grid, thin spines.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10 * SCALE,
        "axes.titlesize": 11 * SCALE,
        "axes.labelsize": 10.5 * SCALE,
        "axes.linewidth": 0.9,
        "axes.axisbelow": True,
        "legend.fontsize": 8.0 * SCALE,
        "xtick.labelsize": 9 * SCALE,
        "ytick.labelsize": 9 * SCALE,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "figure.dpi": 200,
    }
)

# (panel tag, title, run, colour, steps_per_epoch, legend loc)
# Titles stay abbreviated so they don't collide at half-column type sizes; the
# LaTeX caption spells out Western Australia / Arabian Shield.
RUNS = [
    ("(a)", "WA", "rdnpp_x4_wa_baseline", "#1f6feb", 5071 / 64, "upper right"),
    ("(b)", "KSA", "rdnpp_x3_nb23_noadapt_cellgrid8_f3", "#d1495b", 18967 / 64, "upper right"),
]


def onecycle_lr(p, max_lr, pct_start, div_factor=25.0, final_div_factor=1e4):
    """OneCycleLR (cos) learning rate at schedule fraction p in [0, 1] — torch defaults."""
    p = np.asarray(p, float)
    init_lr = max_lr / div_factor
    min_lr = init_lr / final_div_factor
    up = init_lr + (max_lr - init_lr) * (1 - np.cos(np.pi * np.clip(p / pct_start, 0, 1))) / 2
    dn = (
        max_lr
        + (min_lr - max_lr) * (1 - np.cos(np.pi * np.clip((p - pct_start) / (1 - pct_start), 0, 1))) / 2
    )
    return np.where(p <= pct_start, up, dn)


def load(run):
    d = torch.load(ROOT / f"checkpoints/{run}_last.pt", map_location="cpu", weights_only=False)
    h, cfg = d["hist"], d.get("config", {})
    tr = np.asarray(h["train_loss"], float)
    va = np.asarray(h["val_loss"], float)
    ep = np.asarray(h.get("epochs", np.arange(1, len(tr) + 1)), float)
    m = np.isfinite(va)  # keep validated epochs
    return ep, tr, va, m, cfg


def draw(ax, tag, title, run, color, spe, legend_loc):
    ep, tr, va, m, cfg = load(run)
    step = ep * spe / 1000  # optimizer step, in thousands

    # OneCycle LR schedule, lightly overlaid in the background on a twin axis.
    # Fraction of the schedule at epoch e is e/num_epochs (steps_per_epoch cancels).
    lr_ax = ax.twinx()
    p = ep / cfg.get("num_epochs", ep[-1])
    lr = onecycle_lr(p, cfg.get("lr_start", np.nan), cfg.get("pct_start", 0.3))
    lr_ax.plot(step, lr, color="0.55", lw=1.1 * SCALE, alpha=0.55, zorder=0, label=f"LR (peak 3e-4)")
    lr_ax.set_ylim(0, cfg.get("lr_start", 1) * 1.35)
    lr_ax.set_yticks([])  # no LR scale — schedule shape only
    lr_ax.spines[["right", "top"]].set_visible(False)
    ax.set_zorder(lr_ax.get_zorder() + 1)  # loss curves draw above the LR overlay
    ax.patch.set_visible(False)

    ax.plot(step, tr, color=color, lw=1.6 * SCALE, label="train")
    ax.plot(step[m], va[m], color=color, lw=1.4 * SCALE, ls="--", alpha=0.9, label="validation")

    ibest = int(np.nanargmin(va))
    ax.axvline(step[ibest], color="0.5", lw=0.9 * SCALE, ls=":", zorder=1)
    ax.scatter(
        [step[ibest]],
        [va[ibest]],
        color=color,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        s=42 * SCALE**2,
        label="best validation",
    )

    # No panel titles — the LaTeX caption identifies the panels.
    ax.set_xlabel("training step (thousands)")
    ax.set_ylabel("masked L1 loss")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = lr_ax.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc=legend_loc,
        frameon=False,
        handlelength=1.5,
        labelspacing=0.3,
        borderaxespad=0.3,
        handletextpad=0.5,
    )
    ax.grid(alpha=0.2, lw=0.6)
    ax.margins(x=0.02)
    ax.spines["top"].set_visible(False)
    return ep, tr, va, m, ibest


def main():
    # No suptitle — the LaTeX caption carries it. Canvas is ~2x the half-column
    # print width so the SCALE'd type lands readable after \includegraphics shrinks it.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), constrained_layout=True)
    for ax, cfg in zip(axes, RUNS):
        draw(ax, *cfg)
    fig.savefig(OUT / "train_val_overfit_wa_ksa.png", bbox_inches="tight")
    plt.close(fig)

    for cfg in RUNS:
        _, title, run, _, spe, _ = cfg
        f, a = plt.subplots(figsize=(4.9, 3.9), constrained_layout=True)
        draw(a, *cfg)
        t = "wa" if "wa" in run else "ksa"
        f.savefig(OUT / f"train_val_overfit_{t}.png", bbox_inches="tight")
        plt.close(f)
        ep, tr, va, m, _ = load(run)
        ib = int(np.nanargmin(va))
        print(
            f"{title}: {len(tr)} ep = {ep[-1]*spe/1000:.0f}k steps, best val {va[ib]:.4f} "
            f"@ {ep[ib]*spe/1000:.1f}k steps, final train {tr[-1]:.4f} / val {va[m][-1]:.4f}"
        )


if __name__ == "__main__":
    main()

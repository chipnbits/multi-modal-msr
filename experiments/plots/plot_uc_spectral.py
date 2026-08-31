"""What the survey height destroys — the UC multiplier at KSA's actual numbers.

Upward continuation is a low-pass filter with the Fourier multiplier exp(-|k| dz)
(report Eq. 9); inverting it multiplies by exp(+|k| dz), which is unbounded as |k|
grows, with condition number exp(pi dz / ds) (report Eq. 12). This plots the
multiplier against wavelength, so the axis is physical, and marks the three
wavelengths worth naming out loud.

Everything here is the report's own arithmetic: Eq. 9 for the curve, Eq. 12 for the
condition number, and the two attenuation figures the report already quotes (~6% at
lambda = 500 m, ~2% at the LR Nyquist). Deliberately makes NO claim about how the
measured 14-34% survey inconsistency propagates through an inversion -- that would
need the spectrum of epsilon, which was never measured, and epsilon is known to be
structured rather than white.

Run:  uv run python experiments/plots/plot_uc_spectral.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magsr import ROOT_FOLDER

HR_PX = 60.0  # m, KSA HR grid spacing
LR_PX = 180.0  # m, KSA LR grid (3:1)
DZ_LO, DZ_HI = 200.0, 250.0  # m, continuation gap measured by the EL / spectral fit

BAND = "#1F4E79"
FILL = "#8CB4CE"
WARN = "#B5493A"

# Wavelengths worth naming on a slide, with hand-placed label offsets (log-x, so the
# 360 m and 500 m marks sit close together and need pushing apart).
MARKS = [
    (2 * LR_PX, "LR Nyquist\n360 m", 0.72, 20),
    (500.0, "500 m", 1.42, 11),
    (2 * HR_PX, "HR Nyquist\n120 m", 1.30, 46),
]


def multiplier(lam: np.ndarray, dz: float) -> np.ndarray:
    """UC attenuation exp(-|k| dz) at wavelength lam, with |k| = 2*pi/lam."""
    return np.exp(-(2 * np.pi / lam) * dz)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT_FOLDER / "figures" / "uc_operator" / "uc_spectral.png")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lam = np.logspace(np.log10(100), np.log10(6000), 600)
    lo, hi = multiplier(lam, DZ_HI), multiplier(lam, DZ_LO)  # dz_hi attenuates more
    mid = multiplier(lam, (DZ_LO + DZ_HI) / 2)
    kappa = np.exp(np.pi * (DZ_LO + DZ_HI) / 2 / HR_PX)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.fill_between(
        lam,
        lo * 100,
        hi * 100,
        color=FILL,
        alpha=0.55,
        label=f"$\\Delta z \\sim U[{DZ_LO:.0f}, {DZ_HI:.0f}]$ m",
    )
    ax.plot(lam, mid * 100, color=BAND, lw=2.2)
    ax.set_xscale("log")
    ax.set_xlabel("Anomaly wavelength $\\lambda$ (m)")
    ax.set_ylabel("Amplitude surviving upward continuation (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(100, 6000)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")

    for x, lab, fx, fy in MARKS:
        v = multiplier(np.array([x]), (DZ_LO + DZ_HI) / 2)[0] * 100
        ax.axvline(x, color="#999999", lw=0.8, ls=(0, (3, 3)))
        ax.plot([x], [v], "o", color=WARN, ms=6, zorder=5)
        ax.annotate(
            f"{lab}\n{v:.1f}% left",
            xy=(x, v),
            xytext=(x * fx, v + fy),
            fontsize=9.5,
            color="#333333",
            ha="center",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8),
        )

    ax.grid(alpha=0.22, lw=0.6, which="both")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        "Upward continuation is a low-pass filter  $\\exp(-|\\mathbf{k}|\\,\\Delta z)$\n"
        f"Inverting it is ill-posed:  $\\kappa(\\mathbf{{A}}) = e^{{\\pi \\Delta z/\\Delta s}} \\approx$ {kappa:,.0f}",
        fontsize=12.5,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  kappa(A) = {kappa:,.0f}  (dz={((DZ_LO + DZ_HI) / 2):.0f} m, ds={HR_PX:.0f} m)")
    for x, lab, _, _ in MARKS:
        m = multiplier(np.array([x]), (DZ_LO + DZ_HI) / 2)[0]
        print(f"  lambda={x:6.0f} m -> {m * 100:7.3f}% survives")


if __name__ == "__main__":
    main()

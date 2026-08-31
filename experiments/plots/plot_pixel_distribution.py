"""
Raw pixel-value distributions of the WA and KSA aeromagnetic HR surveys, against a matched Gaussian Output:
figures/pixel_distribution/pixel_distribution_hist.png  (+ printed stats)

Run:  python experiments/plots/plot_pixel_distribution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.stats import kurtosis, norm, skew

from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig

SURVEYS = [
    ("WA Goldfields (HR, 20 m)", "TMI anomaly (nT)", "#3A7CA5", Path("data/WA/Goldfields_20m_HR.tif")),
    (
        "KSA Shield (HR, 60 m)",
        "RTP anomaly (nT)",
        "#B5493A",
        KSAAlignedConfig.default().hr_normalized_product_path("AMF_RTP"),
    ),
]
TARGET_PX = 80_000_000  # subsample cap (preserves the marginal distribution)


def read_finite(path: Path) -> np.ndarray:
    """Nearest-subsampled finite pixel values (nT) from a single-band raster."""
    with rasterio.open(path) as ds:
        k = max(1, round(((ds.width * ds.height) / TARGET_PX) ** 0.5))
        arr = (
            ds.read(1, out_shape=(ds.height // k, ds.width // k), resampling=Resampling.nearest)
            .astype(np.float64)
            .ravel()
        )
        nod = ds.nodata
    m = np.isfinite(arr) & (arr > -9e4)  # drop NaN + the -99999 sentinel
    if nod is not None and np.isfinite(nod):
        m &= arr != nod
    return arr[m]


def describe(x: np.ndarray) -> dict:
    mu, sd = float(x.mean()), float(x.std())
    z = (x - mu) / sd
    return dict(
        n=x.size,
        mean=mu,
        std=sd,
        median=float(np.median(x)),
        skew=float(skew(x)),
        exkurt=float(kurtosis(x, fisher=True)),  # excess (Gaussian=0)
        p3=float((np.abs(z) > 3).mean()),
        p5=float((np.abs(z) > 5).mean()),
        vmin=float(x.min()),
        vmax=float(x.max()),
    )


def main():
    out = Path("figures/pixel_distribution")
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(SURVEYS), figsize=(10, 4.2), constrained_layout=True)

    for ax, (name, xlabel, color, path) in zip(axes, SURVEYS):
        x = read_finite(path)
        s = describe(x)
        print(f"\n{name}  [{path.name}]")
        print(
            f"  n={s['n']:,}  mean={s['mean']:.1f}  std={s['std']:.1f}  "
            f"median={s['median']:.1f} nT  range=[{s['vmin']:.0f}, {s['vmax']:.0f}]"
        )
        print(f"  skew={s['skew']:.2f}  EXCESS KURTOSIS={s['exkurt']:.1f} (Gaussian=0)")
        print(
            f"  P(|z|>3)={100*s['p3']:.2f}% (Gaussian 0.27%)   "
            f"P(|z|>5)={100*s['p5']:.4f}% (Gaussian 6e-5%)"
        )

        # x-window at the matched sigma so the leptokurtic signature is visible:
        # a sharper-than-Gaussian peak AND heavier tails that cross the bell.
        lo, hi = s["mean"] - 6 * s["std"], s["mean"] + 6 * s["std"]
        ctr = 0.5 * (bins := np.linspace(lo, hi, 140))[:-1] + 0.5 * bins[1:]
        dens, _ = np.histogram(x, bins=bins, density=True)
        good = dens > 0
        ax.fill_between(ctr, dens, 1e-30, where=good, color=color, alpha=0.12)
        ax.plot(ctr[good], dens[good], color=color, lw=2.0, label="survey pixels")
        xs = np.linspace(lo, hi, 600)
        ax.plot(
            xs, norm.pdf(xs, s["mean"], s["std"]), "k--", lw=1.6, label=r"Gaussian ($\mu,\sigma$ matched)"
        )
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(dens[good].min() * 0.5, dens.max() * 3)  # bound to the data
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("density (log)", fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.tick_params(labelsize=9)
        ax.legend(loc="upper right", fontsize=9)

    p = out / "pixel_distribution_hist.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

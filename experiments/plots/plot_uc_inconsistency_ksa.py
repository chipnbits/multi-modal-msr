"""Physics-consistency check for KSA: apply the (draped, best-fit) upward-
continuation operator to real HR RTP and compare against the real LR survey.

For each patch we
  1. read an oversized 264 px HR context (+ 30 m DEM) and the co-registered
     44 px real LR tile,
  2. fit a draped equivalent layer on the HR survey drape z = -(60 m + DEM),
  3. forward-continue to z - dz and *search dz* to minimise the demeaned
     residual against the real LR (best-case altitude; DC bias ignored, per
     the request — we do not care about a constant level shift),
  4. decimate the continued field 3:1 with an IDEAL band-limited (sinc /
     Fourier-truncation) decimator, not a 3x3 box average,
  5. crop the central region so FFT edge effects stay in the discarded ring.

The residual eps = (LR - mean) - (LR_UC - mean) is the operator inconsistency
that eq:inverse_problem carries: even at the best-fit altitude and after
removing bias, real HR and LR are far from operator-consistent. This is the
figure that motivates the eps term.

Outputs two PNGs (examples grid + population stats).

Run:  python experiments/plots/plot_uc_inconsistency_ksa.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from magsr.datasets import build_ksa_aligned_datasets
from magsr.fourier import fit_equivalent_layer, upward_continue

# ----------------------------- geometry / operator constants -----------------
HR_PROD = "AMF_RTP"  # HR reduced-to-pole anomaly (nT)
LR_PROD = "RTP"  # real LR regional survey (nT)
DX = 60.0  # HR pixel size (m)
CLEARANCE_M = 60.0  # nominal HR terrain clearance (m)
LAYER_DEPTH_M = 300.0  # equivalent layer below the deepest sensor (m)
LR_SCALE = 3  # HR->LR decimation factor
DEM_SCALE = 2  # DEM (30 m) pixels per HR (60 m) pixel

CTX_PX = 264  # oversized HR context for the fit/continuation
OUT_PX = 132  # central crop compared (66 px ~= 4 km guard ring)
MARGIN = (CTX_PX - OUT_PX) // 2  # 66 HR px
LR_OUT = OUT_PX // LR_SCALE  # 44 LR px

PAD_TO = 384
N_LAYERS = 24
CG_ITERS = 10
# Per-patch dz is searched only within a physically-plausible survey gap: the
# gain-aware EL / spectral study brackets the real HR->LR clearance at
# ~149-268 m. Allowing dz to run away turns it into a free over-smoothing knob
# (the coarse LR survey is smoother than UC at any plausible altitude), which
# would artificially deflate eps. We keep generous headroom: [100, 400] m.
DZ_GRID = np.arange(100.0, 401.0, 10.0)  # continuation-gap search (m)
EL_DZ_BRACKET = (149.0, 268.0)  # independent EL/spectral estimate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------- ideal decimation ----------------------------
def fourier_decimate(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Ideal band-limited M:1 decimation by spectral truncation (sinc kernel).

    Brick-wall low-pass to the LR Nyquist, then resample -- the sampling-theorem
    ideal, unlike a box `avg_pool2d` whose sinc passband droops and whose
    stopband leaks. `x` is (B, H, W) real with H == W divisible by `scale`.
    Amplitude is scaled by 1/scale**2 so the field mean is preserved.
    """
    b, h, w = x.shape
    h2, w2 = h // scale, w // scale
    X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    r0, c0 = (h - h2) // 2, (w - w2) // 2
    Xc = torch.fft.ifftshift(X[..., r0 : r0 + h2, c0 : c0 + w2], dim=(-2, -1))
    return torch.fft.ifft2(Xc).real / (scale * scale)


def _pool2(a: torch.Tensor) -> torch.Tensor:
    """NaN-aware 2x2 average pool (30 m DEM -> 60 m HR grid)."""
    m = torch.isfinite(a)
    num = F.avg_pool2d(torch.where(m, a, torch.zeros_like(a))[None, None], DEM_SCALE)[0, 0]
    cnt = F.avg_pool2d(m.float()[None, None], DEM_SCALE)[0, 0]
    return num / cnt.clamp_min(1e-6)


def _fill_mean(a: torch.Tensor) -> torch.Tensor:
    """Replace non-finite pixels with the finite mean (per tile)."""
    m = torch.isfinite(a)
    mean = torch.where(m, a, torch.zeros_like(a)).sum() / m.sum().clamp_min(1)
    return torch.where(m, a, mean)


def demeaned_rmse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-tile RMS of (pred - tgt) after removing the residual's mean. (B,)."""
    n = mask.float().sum(dim=(-2, -1)).clamp_min(1)
    r = torch.where(mask, pred - tgt, torch.zeros_like(pred))
    r = r - (r.sum(dim=(-2, -1)) / n).view(-1, 1, 1)
    r = torch.where(mask, r, torch.zeros_like(r))
    return torch.sqrt((r * r).sum(dim=(-2, -1)) / n)


# ------------------------------- data reading --------------------------------
def read_context(ds, patch):
    """Return (hr_ctx 264, dem_hr 264, lr_real 44) numpy tiles in nT / m, or None.

    Skips patches whose oversized context is not (almost) fully on-survey.
    """
    r, c = patch.row_px, patch.col_px
    if r % LR_SCALE or c % LR_SCALE:  # LR grid must align
        return None
    hr = ds._hr[HR_PROD].read_window(r - MARGIN, c - MARGIN, CTX_PX, CTX_PX)
    dem30 = ds._dem.read_window(
        (r - MARGIN) * DEM_SCALE, (c - MARGIN) * DEM_SCALE, CTX_PX * DEM_SCALE, CTX_PX * DEM_SCALE
    )
    lr = ds._lr[LR_PROD].read_window(r // LR_SCALE, c // LR_SCALE, LR_OUT, LR_OUT)
    if np.isfinite(hr).mean() < 0.995 or np.isfinite(dem30).mean() < 0.99:
        return None
    if np.isfinite(lr).mean() < 0.98:
        return None
    return hr.astype(np.float32), dem30.astype(np.float32), lr.astype(np.float32)


def collect(ds, want: int):
    """Stride across the index so the sample spans the whole test region."""
    patches = ds.index.patches
    step = max(1, len(patches) // (want * 8))
    out = []
    for p in patches[::step]:
        try:
            t = read_context(ds, p)
        except Exception:
            t = None
        if t is not None:
            out.append((p, *t))
        if len(out) >= want:
            break
    return out


# ------------------------------- core operator -------------------------------
@torch.no_grad()
def best_fit_uc(hr_ctx, dem_hr, lr_real):
    """Fit drape EL, search dz per tile, return best LR_UC / eps / dz / rms.

    hr_ctx (B,264,264) nT, dem_hr (B,264,264) m, lr_real (B,44,44) nT — tensors.
    """
    z_obs = -(CLEARANCE_M + dem_hr)  # z-down survey drape
    z_layer = float(z_obs.max()) + LAYER_DEPTH_M
    layer = fit_equivalent_layer(
        hr_ctx, dx=DX, dy=DX, z_obs=z_obs, z_layer=z_layer, pad_to=PAD_TO, cg_iters=CG_ITERS
    )

    # Decimate the FULL oversized field, then crop the central LR tile, so the
    # decimator's own periodic-FFT ringing stays in the discarded guard ring.
    off_lr = MARGIN // LR_SCALE  # 22

    def uc_of(z_t):
        u = upward_continue(
            layer, z_target=z_t, n_layers=N_LAYERS, z_min=float(z_t.min()), z_max=float(z_t.max())
        )
        lr_full = fourier_decimate(u, LR_SCALE)
        return lr_full[:, off_lr : off_lr + LR_OUT, off_lr : off_lr + LR_OUT]

    lr_mask = torch.isfinite(lr_real)
    lr_t = torch.nan_to_num(lr_real)
    best_rms = torch.full((hr_ctx.shape[0],), float("inf"), device=DEVICE)
    best_dz = torch.zeros_like(best_rms)
    for dz in DZ_GRID:
        rms = demeaned_rmse(uc_of(z_obs - float(dz)), lr_t, lr_mask)
        better = rms < best_rms
        best_rms = torch.where(better, rms, best_rms)
        best_dz = torch.where(better, torch.full_like(best_dz, float(dz)), best_dz)

    # recompute final maps at each tile's own best dz (one forward)
    lr_uc = uc_of(z_obs - best_dz.view(-1, 1, 1))
    # demeaned residual (bias removed) on the LR grid
    n = lr_mask.float().sum(dim=(-2, -1)).clamp_min(1)
    resid = torch.where(lr_mask, lr_uc - lr_t, torch.zeros_like(lr_uc))
    resid = resid - (resid.sum(dim=(-2, -1)) / n).view(-1, 1, 1)
    eps = torch.where(lr_mask, resid, torch.tensor(float("nan"), device=DEVICE))
    return lr_uc, eps, best_dz, best_rms, lr_mask


def process(items, chunk=40):
    """Run best_fit_uc over all collected items in memory-bounded chunks."""
    res = {k: [] for k in ("lr_uc", "eps", "dz", "rms")}
    hr_c, lr_c = [], []
    for i in range(0, len(items), chunk):
        sl = items[i : i + chunk]
        hr = torch.stack([_fill_mean(torch.from_numpy(x[1])) for x in sl]).to(DEVICE)
        dem = torch.stack([_pool2(_fill_mean(torch.from_numpy(x[2]))) for x in sl]).to(DEVICE)
        lr = torch.stack([torch.from_numpy(x[3]) for x in sl]).to(DEVICE)
        lr_uc, eps, dz, rms, _ = best_fit_uc(hr, dem, lr)
        res["lr_uc"].append(lr_uc.cpu())
        res["eps"].append(eps.cpu())
        res["dz"].append(dz.cpu())
        res["rms"].append(rms.cpu())
        hr_c.append(hr[:, MARGIN : MARGIN + OUT_PX, MARGIN : MARGIN + OUT_PX].cpu())
        lr_c.append(lr.cpu())
        print(f"  processed {min(i + chunk, len(items))}/{len(items)}", flush=True)
    return (
        torch.cat(res["lr_uc"]),
        torch.cat(res["eps"]),
        torch.cat(res["dz"]),
        torch.cat(res["rms"]),
        torch.cat(hr_c),
        torch.cat(lr_c),
    )


# ------------------------------- plotting ------------------------------------
def _rng(*arrs, lo=2, hi=98):
    v = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs])
    return np.percentile(v, lo), np.percentile(v, hi)


def figure_examples(hr, lr, lr_uc, eps, dz, rms, idx, path):
    n = len(idx)
    fig, ax = plt.subplots(n, 4, figsize=(7.2, 1.95 * n), constrained_layout=True)
    ax = np.atleast_2d(ax)
    cols = [
        "HR RTP (60 m)",
        "LR survey (180 m)",
        r"LR$_{\rm UC}$=dec(UC(HR))",
        r"$\epsilon$ = LR $-$ LR$_{\rm UC}$",
    ]
    for j, t in enumerate(cols):
        ax[0, j].set_title(t, fontsize=11.25)
    for row, k in enumerate(idx):
        hr_i, lr_i, uc_i, ep_i = (a[k].numpy() for a in (hr, lr, lr_uc, eps))
        gv = _rng(lr_i, uc_i)  # LR pair share a grey scale
        hv = _rng(hr_i)
        ev = np.nanpercentile(np.abs(ep_i), 98)
        for a, img, vr, cm in (
            (ax[row, 0], hr_i, hv, "gray"),
            (ax[row, 1], lr_i, gv, "gray"),
            (ax[row, 2], uc_i, gv, "gray"),
        ):
            a.imshow(np.ma.masked_invalid(img), cmap=cm, vmin=vr[0], vmax=vr[1])
            a.set_xticks([])
            a.set_yticks([])
        im = ax[row, 3].imshow(np.ma.masked_invalid(ep_i), cmap="RdBu_r", vmin=-ev, vmax=ev)
        ax[row, 3].set_xticks([])
        ax[row, 3].set_yticks([])
        cb = fig.colorbar(im, ax=ax[row, 3], fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label("nT", fontsize=8.75)
        sig = np.nanstd(lr_i)
        ax[row, 0].set_ylabel(f"$\\Delta z^*$={dz[k]:.0f} m", fontsize=10)
        ax[row, 3].set_title(
            f"$\\sigma_\\epsilon$={rms[k]:.0f} nT ({100*rms[k]/sig:.0f}% $\\sigma_{{LR}}$)", fontsize=10
        )
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def figure_stats(rms, dz, path, n_shown):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    r = rms.numpy()
    ax[0].hist(r, bins=30, color="#4062BB", alpha=0.85)
    ax[0].axvline(np.median(r), color="crimson", lw=2, label=f"median {np.median(r):.1f} nT")
    ax[0].set_xlabel(r"residual std $\sigma_\epsilon$ (nT)")
    ax[0].set_ylabel("patches")
    ax[0].legend()
    ax[0].set_title("Irreducible inconsistency (best-fit altitude)")
    d = dz.numpy()
    ax[1].axvspan(*EL_DZ_BRACKET, color="0.75", alpha=0.5, label="EL/spectral estimate")
    ax[1].hist(d, bins=20, color="#3A9D4F", alpha=0.85)
    ax[1].axvline(np.median(d), color="crimson", lw=2, label=f"median {np.median(d):.0f} m")
    ax[1].set_xlabel(r"best-fit $\Delta z$ (m), searched in [100, 400]")
    ax[1].set_ylabel("patches")
    ax[1].legend(loc="upper left")
    ax[1].set_title("Fitted continuation gap")
    sat = 100.0 * (d >= DZ_GRID[-1] - 1e-6).mean()
    ax[1].text(
        0.97,
        0.5,
        f"{sat:.0f}% saturate the ceiling:\nLR is smoother than UC\n" "at any plausible altitude",
        transform=ax[1].transAxes,
        ha="right",
        va="center",
        fontsize=8,
        color="0.25",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    fig.suptitle(f"KSA fold-3 test  —  {n_shown} patches, ideal decimation", fontsize=12)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------- main ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="data/processed/ksa_aligned/patch_indices_cellgrid8s12_fold3")
    ap.add_argument("--n", type=int, default=160, help="patches to analyse")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", default="figures/uc_inconsistency")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={DEVICE}  index={args.index_dir}")
    ds = build_ksa_aligned_datasets(index_dir=Path(args.index_dir), load_dem=True)["test"]
    print(f"test patches: {len(ds.index.patches)}")
    items = collect(ds, args.n)
    print(f"collected {len(items)} valid oversized-context patches")

    lr_uc, eps, dz, rms, hr, lr = process(items)
    print(
        f"eps RMS  median={rms.median():.1f}  mean={rms.mean():.1f} nT | " f"dz median={dz.median():.0f} m"
    )

    # example rows: strong-signal patches spanning the eps range
    sig = torch.tensor([np.nanstd(lr[i].numpy()) for i in range(len(items))])
    strong = torch.argsort(sig, descending=True)[: max(args.examples * 4, 20)]
    order = strong[torch.argsort(rms[strong])]
    idx = order[torch.linspace(0, len(order) - 1, args.examples).long()].tolist()

    figure_examples(hr, lr, lr_uc, eps, dz, rms, idx, out / "uc_inconsistency_ksa_examples.png")
    figure_stats(rms, dz, out / "uc_inconsistency_ksa_stats.png", len(items))
    print(f"wrote {out}/uc_inconsistency_ksa_examples.png")
    print(f"wrote {out}/uc_inconsistency_ksa_stats.png")


if __name__ == "__main__":
    main()

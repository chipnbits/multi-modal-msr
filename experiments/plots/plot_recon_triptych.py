"""Paper-ready qualitative reconstruction comparison for KSA (x3) and WA (x4).

One 3x2 figure per example patch::

    LR  (input)   |  Bicubic
    HR  (target)  |  Baseline RDN
    DEM (terrain) |  Best RDN

Left column = the available data (LR input, HR target, and the DEM terrain modality
the best model exploits); right column = the three reconstructions, all on the SAME
diverging blue->red nT scale as HR (white pinned to 0 nT via ``TwoSlopeNorm``) so the
sharpening is read directly. The DEM gets its own elevation colourbar. RMSE / SSIM per
patch annotate each method title (drop with ``--no-metrics`` for a pure-qualitative
figure). ``--errors`` adds a signed-error column in a distinct purple/orange map.

Models (champion variant per model):
    KSA  baseline = mag-only  rdnpp_x3_ksa_f3_nb13_k0_dwd2e-3_s12  (soup_ssim_rmse)
    KSA  best     = mag+1VD+DEMgrad+FVD ..._1vd_demgrad_fvd5           (soup_ssim_rmse)
    WA   baseline = rdnpp_x4_wa_baseline (best)
    WA   best     = rdnpp_x4_wa_1vd_demgrad_fvd5 (best)

All KSA scores are on the canonical fold-3 benchmark test split. Auxiliary input
channels (1VD, DEM gradient) are assembled exactly as training/eval does.

    uv run python experiments/plots/plot_recon_triptych.py --dataset ksa --auto 4
    uv run python experiments/plots/plot_recon_triptych.py --dataset wa  --indices 7 300 780
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from magsr import ROOT_FOLDER
from magsr.datasets import build_ksa_aligned_datasets, build_wa_datasets, pool_collate
from magsr.metrics import apply_hr_nan_mask, per_patch_rmse, per_patch_ssim
from magsr.models import BicubicModel, load_checkpoint, rdnpp_default_x4

NAN_GREY = "#8a8a8a"

KSA_INDEX_DIR = Path("data/processed/ksa_aligned/patch_indices_cellgrid8_fold3")

KSA_MODELS = {
    "baseline": ("RDN++ baseline", "checkpoints/rdnpp_x3_nb23_noadapt_cellgrid8_f3_soup_ssim_rmse.pt"),
    "control": ("Mag-only control", "checkpoints/rdnpp_x3_ksa_f3_nb13_k0_dwd2e-3_s12_soup_ssim_rmse.pt"),
}
KSA_BEST = "checkpoints/rdnpp_x3_ksa_f3_nb13_k0_dwd2e-3_s12_1vd_demgrad_fvd5_soup_ssim_rmse.pt"
WA_BASELINE = "checkpoints/rdnpp_x4_wa_baseline_best.pt"
WA_BEST = "checkpoints/rdnpp_x4_wa_1vd_demgrad_fvd5_best.pt"


@dataclass
class Method:
    label: str
    img: np.ndarray  # reconstruction in nT
    rmse: float  # nT, masked
    ssim: float | None  # None when the patch is not fully valid


@dataclass
class Patch:
    subtitle: str
    lr: np.ndarray  # low-res mag input, nT
    hr: np.ndarray  # target, nT
    dem: np.ndarray  # terrain elevation, m
    methods: list[Method]  # [bicubic, baseline, best]
    tag: str  # short filename tag


# --------------------------------------------------------------------------- #
# KSA backend
# --------------------------------------------------------------------------- #
def _ksa_channels(cfg: dict) -> tuple[bool, str, tuple[str, ...]]:
    """(use_dem, dem_mode, lr_aux) from a checkpoint config (handles use_1vd flag)."""
    aux = list(cfg.get("lr_aux") or [])
    if cfg.get("use_1vd") and "1VD" not in aux:
        aux = ["1VD", *aux]
    return bool(cfg.get("use_dem")), cfg.get("dem_mode") or "relief", tuple(aux)


def prepare_ksa(
    indices: list[int] | None, n_auto: int, device: torch.device, ref: str = "baseline"
) -> list[Patch]:
    ref_label, ref_ckpt = KSA_MODELS[ref]
    best, ckb = load_checkpoint(KSA_BEST, device=device, return_ckpt=True)
    base = load_checkpoint(ref_ckpt, device=device)
    bicubic = BicubicModel(upscale_factor=3).to(device)

    use_dem, dem_mode, aux = _ksa_channels(ckb.get("config") or {})
    ds = build_ksa_aligned_datasets(
        index_dir=KSA_INDEX_DIR, load_dem=use_dem, dem_mode=dem_mode, lr_aux_products=aux
    )["test"]
    hr_prod = ds.config.hr_products[0]

    if not indices:
        # Stratify by survey block so B1/B2/B3 are all represented.
        by_block: dict[int, list[int]] = {}
        for i, t in enumerate(ds.index.patches):
            if t.valid_frac >= 0.999:
                by_block.setdefault(ds._block_from_patch(t), []).append(i)
        indices = _auto_select_per_group(lambda i: ds[i]["hr"][hr_prod].numpy(), by_block, n_auto)
        print(f"[ksa] auto-selected patches: {indices}  (blocks {sorted(by_block)})")

    results: list[Patch] = []
    for idx in indices:
        batch = pool_collate(
            [ds[idx]], hr_products=ds.config.hr_products, lr_products=ds.config.lr_products
        )
        blocks = [m["block"] for m in batch["meta"]]
        blocks_t = torch.tensor(blocks, dtype=torch.long, device=device)
        name = batch["meta"][0]["name"]

        lr_t = batch["lr"].to(device)
        hr_t = batch["hr"][:, :1].to(device)
        lr_n = ds.normalize(lr_t, blocks=blocks).nan_to_num(0.0)
        if ds.config.load_dem:
            lr_n = torch.cat([lr_n, ds.dem_features(batch["dem"].to(device))], dim=1)
        if ds.lr_aux_products:
            lr_n = torch.cat([lr_n, ds.lr_aux_to_channels(batch["lr_aux"].to(device))], dim=1)
        mag_n = lr_n[:, :1]
        hr_n = ds.normalize(hr_t, blocks=blocks)

        with torch.no_grad():
            preds = {
                "Bicubic": bicubic(mag_n).clamp(0, 1),
                ref_label: base(mag_n, blocks_t).clamp(0, 1),
                "Best RDN": best(lr_n, blocks_t).clamp(0, 1),
            }

        def to_nt(t: torch.Tensor) -> torch.Tensor:
            return ds.denormalize(t, blocks=blocks)  # [0,1] -> nT (per survey block)

        results.append(
            Patch(
                subtitle=f"KSA  x3  |  B{blocks[0]}  patch #{idx}  ({name})",
                lr=_np(to_nt(mag_n)),
                hr=_np(to_nt(hr_n)),
                dem=batch["dem"].squeeze().numpy(),  # raw elevation (m), same footprint
                methods=_methods(preds, hr_n, to_nt=to_nt),
                tag=f"ksa_B{blocks[0]}_p{idx}",
            )
        )
    return results


# --------------------------------------------------------------------------- #
# WA backend
# --------------------------------------------------------------------------- #
def _load_wa(path: str, device: torch.device) -> torch.nn.Module:
    """WA checkpoints predate model_spec — rebuild by inferring in_channels."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    in_ch = int(state["head.weight"].shape[1])
    model = rdnpp_default_x4(in_channels=in_ch).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_wa(indices: list[int] | None, n_auto: int, device: torch.device) -> list[Patch]:
    base = _load_wa(WA_BASELINE, device)  # in=1
    best = _load_wa(WA_BEST, device)  # in=4 (MAG,1VD,DEMGX,DEMGY)
    bicubic = BicubicModel(upscale_factor=4).to(device)

    ds = build_wa_datasets(lr_aux=("1VD",), use_dem=True, dem_mode="grad")["test"]
    vmin, vmax = ds.vmin, ds.vmax
    dr = vmax - vmin

    if not indices:
        indices = _auto_select(
            lambda i: ds[i]["hr"]["MAG"].numpy(), candidates=list(range(len(ds))), n=n_auto
        )
        print(f"[wa] auto-selected patches: {indices}")

    def to_nt(t: torch.Tensor) -> torch.Tensor:
        return t * dr + vmin  # [0,1] -> nT (global WA span)

    dem_idx = ds.lr_channels.index("DEM")  # raw elevation channel in the stored LR .npy

    results: list[Patch] = []
    for idx in indices:
        batch = pool_collate([ds[idx]])
        name = batch["meta"][0]["name"]
        lr = batch["lr"].to(device)  # (1,4,h,w)
        hr = batch["hr"].to(device)  # (1,1,H,W)
        mag = lr[:, :1]
        dem_raw = np.load(ds.patch_dir / f"{name}_lr.npy")[dem_idx].astype(np.float32)

        with torch.no_grad():
            preds = {
                "Bicubic": bicubic(mag).clamp(0, 1),
                "Baseline RDN": base(mag).clamp(0, 1),
                "Best RDN": best(lr).clamp(0, 1),
            }

        results.append(
            Patch(
                subtitle=f"WA  x4  |  patch #{idx}  ({name})",
                lr=_np(to_nt(mag)),
                hr=_np(to_nt(hr)),
                dem=dem_raw,
                methods=_methods(preds, hr, to_nt=to_nt, ssim_range=dr),
                tag=f"wa_p{idx}",
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Metrics + selection
# --------------------------------------------------------------------------- #
def _np(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().detach().cpu().numpy()


def _metrics(sr, hr, *, to_nt, ssim_range: float = 1.0):
    """RMSE on denormalized nT (masked). SSIM with `ssim_range`: 1.0 scores the [0,1] tensors
    (KSA convention), a nT span scores the denormalized tensors (WA). `to_nt(x)` maps [0,1] -> nT."""
    mask = hr.isfinite().to(sr.dtype)
    sr_m, hr_m = apply_hr_nan_mask(sr, hr)
    sr_nt, hr_nt = to_nt(sr_m) * mask, to_nt(hr_m) * mask
    rmse = per_patch_rmse(sr_nt, hr_nt, mask).item()
    if not bool(mask.all()):
        return rmse, None
    a, b = (sr_m, hr_m) if ssim_range == 1.0 else (sr_nt, hr_nt)
    return rmse, per_patch_ssim(a, b, data_range=ssim_range).item()


def _methods(preds: dict[str, torch.Tensor], hr, *, to_nt, ssim_range: float = 1.0) -> list[Method]:
    """Score each [0,1] prediction into a Method (nT image + RMSE/SSIM) through the one metric path."""
    return [
        Method(label, _np(to_nt(sr)), *_metrics(sr, hr, to_nt=to_nt, ssim_range=ssim_range))
        for label, sr in preds.items()
    ]


def _rank_by_contrast(read_hr, candidates: list[int], *, seed: int = 0, pool: int = 96) -> list[int]:
    """Fully-valid patches from a random probe of ``candidates``, highest HR std first.

    Selection is on HR contrast only — never on which model wins — so the rendered
    examples are not cherry-picked in favour of any method.
    """
    rng = np.random.default_rng(seed)
    probe = rng.permutation(candidates)[:pool]
    scored: list[tuple[float, int]] = []
    for i in probe:
        a = read_hr(int(i))
        if np.isfinite(a).all():
            scored.append((float(a.std()), int(i)))
    scored.sort(reverse=True)
    return [i for _, i in scored]


def _auto_select(read_hr, *, candidates: list[int], n: int, seed: int = 0, pool: int = 96) -> list[int]:
    """Top ``n`` highest-contrast, fully-valid HR patches."""
    return _rank_by_contrast(read_hr, candidates, seed=seed, pool=pool)[:n]


def _auto_select_per_group(read_hr, groups: dict[int, list[int]], n: int, seed: int = 0) -> list[int]:
    """Round-robin the highest-contrast patch from each group until ``n`` are picked.

    Guarantees every KSA survey block appears in the appendix rather than letting a
    single high-contrast block monopolise the top-n.
    """
    ranked = {g: _rank_by_contrast(read_hr, idxs, seed=seed, pool=64) for g, idxs in groups.items()}
    out: list[int] = []
    while len(out) < n and any(ranked.values()):
        for g in sorted(ranked):
            if ranked[g] and len(out) < n:
                out.append(ranked[g].pop(0))
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render(patch: Patch, out_path: Path, *, show_metrics: bool = True, show_errors: bool = False) -> None:
    """3x2 qualitative (LR/HR/DEM | Bicubic/Baseline/Best), + optional 3rd error column.

    On WA the baseline->best difference is ~0.5% of the greyscale ramp and is therefore
    invisible without the error column; on KSA it is 3-7% and reads qualitatively.
    """
    ncols = 3 if show_errors else 2
    fig, axes = plt.subplots(3, ncols, figsize=(3.6 * ncols + 0.6, 10.4), constrained_layout=True)

    # Magnetic anomaly is signed, so it gets a diverging blue->red map. The field is
    # strongly asymmetric (p1/p99 of ~-300/+790 nT), so symmetric limits about zero
    # would throw away 17-86% of the contrast; TwoSlopeNorm keeps the full percentile
    # range while pinning white to exactly 0 nT. Same norm on LR/HR/all recons.
    hv = patch.hr[np.isfinite(patch.hr)]
    vmin, vmax = np.percentile(hv, [1, 99])
    if vmin < 0.0 < vmax:
        fnorm = TwoSlopeNorm(vmin=float(vmin), vcenter=0.0, vmax=float(vmax))
    else:  # all-positive or all-negative patch: no zero to centre on
        fnorm = Normalize(vmin=float(vmin), vmax=float(vmax))
    field = plt.get_cmap("RdBu_r").copy()
    field.set_bad(NAN_GREY)
    # Truncate `terrain` to its land band (skip the 0-0.25 below-sea-level blue,
    # which would falsely read as water on an all-land DEM).
    terrain = LinearSegmentedColormap.from_list(
        "terrain_land", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256))
    )
    terrain.set_bad(NAN_GREY)

    def show(ax, arr, cmap, norm, title):
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("black")
            sp.set_linewidth(0.8)
        return im

    # Left column: the available data — LR input, HR target, DEM (extra modality).
    im_mag = show(axes[0, 0], patch.lr, field, fnorm, "LR (input)")
    show(axes[1, 0], patch.hr, field, fnorm, "HR (target)")
    dv = patch.dem[np.isfinite(patch.dem)]
    dmin, dmax = np.percentile(dv, [1, 99]) if dv.size else (0.0, 1.0)
    im_dem = show(axes[2, 0], patch.dem, terrain, Normalize(dmin, dmax), "DEM")

    # Middle column: the three reconstructions, same field norm as HR.
    for row, m in enumerate(patch.methods):
        title = m.label
        if show_metrics:
            title += f"\nRMSE {m.rmse:.1f} nT" + (f"   SSIM {m.ssim:.3f}" if m.ssim is not None else "")
        show(axes[row, 1], m.img, field, fnorm, title)

    if show_errors:
        # Right column: signed error HR - SR, symmetric about zero. Deliberately a
        # DIFFERENT diverging map (purple/orange) so it cannot be mistaken for the
        # red/blue field panels.
        div = plt.get_cmap("PuOr").copy()
        div.set_bad(NAN_GREY)
        errs = [patch.hr - m.img for m in patch.methods]
        flat = np.concatenate([e[np.isfinite(e)].ravel() for e in errs])
        emax = float(np.percentile(np.abs(flat), 99)) if flat.size else 1.0
        enorm = Normalize(vmin=-emax, vmax=emax)
        for row, (m, e) in enumerate(zip(patch.methods, errs)):
            im_err = show(axes[row, 2], e, div, enorm, f"{m.label} error")

    # Colourbars along the bottom of their own columns.
    cbd = fig.colorbar(im_dem, ax=axes[2, 0], location="bottom", shrink=0.9, aspect=22, pad=0.03)
    cbd.set_label("Elevation (m)", fontweight="bold")
    if show_errors:
        cb = fig.colorbar(
            im_mag, ax=axes[:, 1].tolist(), location="bottom", shrink=0.9, aspect=26, pad=0.03
        )
        cb.set_label("Magnetic field (nT)", fontweight="bold")
        cbe = fig.colorbar(
            im_err, ax=axes[:, 2].tolist(), location="bottom", shrink=0.9, aspect=26, pad=0.03
        )
        cbe.set_label("Error  HR − SR (nT)", fontweight="bold")
    else:
        cb = fig.colorbar(
            im_mag, ax=axes[:, 1].tolist(), location="right", shrink=0.82, aspect=34, pad=0.02
        )
        cb.set_label("Magnetic field (nT)", fontweight="bold")

    fig.suptitle(patch.subtitle, fontweight="bold", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    scores = "  ".join(f"{m.label.split()[0]} {m.rmse:.1f}" for m in patch.methods)
    print(f"wrote {out_path}   [RMSE nT: {scores}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ksa", "wa"], required=True)
    ap.add_argument(
        "--indices", type=int, nargs="*", default=None, help="Explicit test-patch indices (else auto)."
    )
    ap.add_argument(
        "--auto", type=int, default=4, help="How many patches to auto-select when --indices is omitted."
    )
    ap.add_argument(
        "--no-metrics", action="store_true", help="Drop the RMSE/SSIM annotations (pure qualitative)."
    )
    ap.add_argument(
        "--errors",
        action="store_true",
        help="Add a signed-error column. Required for WA, where baseline->best is "
        "~0.5%% of the greyscale ramp and invisible in the recon panels.",
    )
    ap.add_argument(
        "--ksa-ref",
        choices=["baseline", "control"],
        default="baseline",
        help="Which KSA reference model fills the middle recon panel: the published "
        "RDN++ baseline (nb23, 73.5 nT) or our mag-only control (nb13, 73.2 nT).",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches = (
        prepare_ksa(args.indices, args.auto, device, ref=args.ksa_ref)
        if args.dataset == "ksa"
        else prepare_wa(args.indices, args.auto, device)
    )

    out_dir = args.out_dir or (ROOT_FOLDER / "figures" / "recon_compare" / args.dataset)
    for p in patches:
        render(p, out_dir / f"recon_{p.tag}.png", show_metrics=not args.no_metrics, show_errors=args.errors)


if __name__ == "__main__":
    main()

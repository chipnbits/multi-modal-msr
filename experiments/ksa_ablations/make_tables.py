"""Regenerate the KSA ablation LaTeX tables from the eval CSVs.

Reads `latex_tables.yaml` and the eval CSVs, selects each run's checkpoint by argmin validation RMSE per
entry with the lowest test RMSE bolded.

Run:  uv run python experiments/ksa_ablations/make_tables.py            # -> stdout
      uv run python experiments/ksa_ablations/make_tables.py --out t.tex
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import yaml

from magsr import ROOT_FOLDER

EVAL_CSVS = ["results/ksa_ablations_eval.csv"]  # produced by experiments/ksa_ablations/evaluate.py
VARIANT_TAG = {"best_rmse": "rmse", "best": "ssim", "soup_ssim_rmse": "soup"}


def load_evals(paths: list[Path]) -> dict:
    """(run, variant) -> {(split, metric): mean}, merged across CSVs."""
    D: dict = collections.defaultdict(dict)
    for p in paths:
        if not Path(p).exists():
            continue
        for r in csv.DictReader(open(p)):
            if r.get("scope") == "Net":
                D[(r["run"], r["variant"])][(r["split"], r["metric"])] = float(r["mean"])
    return D


def val_selected(D: dict, run: str) -> tuple[str, dict] | None:
    """Return (variant_tag, metrics) for the argmin-val-RMSE checkpoint of `run`."""
    cands = {v: d for (rn, v), d in D.items() if rn == run and v != "last" and ("val", "rmse") in d}
    if not cands:
        return None
    best = min(cands, key=lambda v: cands[v][("val", "rmse")])
    return VARIANT_TAG.get(best, best), cands[best]


def esc(label: str) -> str:
    """Row label as normal text, with LaTeX-safe underscores."""
    return label.replace("_", r"\_")


def fmt_row(label: str, tag: str, ch: int, m: dict, bold: bool, dp: int = 1) -> str:
    v, t = m[("val", "rmse")], m[("test", "rmse")]
    s, ms = m[("test", "ssim")], m[("test", "msssim")]
    tval = f"\\best{{{t:.{dp}f}}}" if bold else f"{t:.{dp}f}"
    return f"{esc(label)} & {tag} & {ch} & {v:.{dp}f} & {tval} & {s:.3f} & {ms:.3f} \\\\"


def render_fold_table(cfg: dict, D: dict) -> str:
    """5-fold summary: mean over folds 0-4 with fold min--max, one `\\mr{mean}{lo--hi}` per cell."""

    def agg(stem: str, split: str, metric: str):
        vals = []
        for f in range(5):
            sel = val_selected(D, f"{stem}_f{f}")
            if sel:
                vals.append(sel[1][(split, metric)])
        if not vals:
            return None
        return sum(vals) / len(vals), min(vals), max(vals)

    def cell(stem, split, metric, dp, best=False):
        a = agg(stem, split, metric)
        if a is None:
            return "---"
        mean = f"\\best{{{a[0]:.{dp}f}}}" if best else f"{a[0]:.{dp}f}"
        return f"\\mr{{{mean}}}{{{a[1]:.{dp}f}--{a[2]:.{dp}f}}}"

    # combo (last model) has the lowest test RMSE — bold its test mean
    combo = cfg["models"][-1]["stem"]
    body = []
    for mdl in cfg["models"]:
        st, is_combo = mdl["stem"], mdl["stem"] == combo
        body.append(
            f"{mdl['label']} & {mdl['ch']} & {cell(st,'val','rmse',1)} & "
            f"{cell(st,'test','rmse',1,best=is_combo)} & {cell(st,'test','ssim',3)} & "
            f"{cell(st,'test','msssim',3)} \\\\"
        )
    bic = cfg["bicubic_stem"]
    body.append(
        f"bicubic & 1 & {cell(bic,'val','rmse',1)} & {cell(bic,'test','rmse',1)} & "
        f"{cell(bic,'test','ssim',3)} & {cell(bic,'test','msssim',3)} \\\\"
    )

    return f"""% ---- auto-generated: fivefold ----
\\begin{{table}}[H]
\\centering
\\caption{{{cfg['caption']}}}
\\label{{tab:exp_v2_ksa}}
\\begin{{tabular}}{{l c R R R R}}
\\toprule
 & & \\multicolumn{{1}}{{c}}{{\\textbf{{Val}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Test}}}} \\\\
\\cmidrule(lr){{3-3}}\\cmidrule(lr){{4-6}}
Model & Ch & RMSE & RMSE & SSIM & MS-SSIM \\\\
\\midrule
{chr(10).join(body[:-1])}
\\midrule
{body[-1]}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def render_table(name: str, spec: dict, D: dict, refs: dict) -> str:
    rows = []
    for r in spec["rows"]:
        sel = val_selected(D, r["run"])
        if sel is None:
            rows.append((r["label"], None, r["ch"], None))
            continue
        tag, m = sel
        rows.append((r["label"], tag, r["ch"], m))
    # bold the lowest test RMSE among the (non-reference) rows
    tests = [m[("test", "rmse")] for _, _, _, m in rows if m]
    tmin = min(tests) if tests else None
    dp = spec.get("rmse_dp", 1)

    mid = set(spec.get("midrule_after", []))
    body = []
    for i, (label, tag, ch, m) in enumerate(rows):
        if m is None:
            body.append(f"{esc(label)} & --- & {ch} & --- & --- & --- & --- \\\\  % MISSING")
        else:
            body.append(fmt_row(label, tag, ch, m, bold=(m[("test", "rmse")] == tmin), dp=dp))
        if (i + 1) in mid:
            body.append("\\midrule")

    ref_labels = {
        "baseline": "baseline (nb23)",
        "control": "control",
        "bicubic": "bicubic",
        "bicubic_wa": "bicubic",
    }
    ref_lines = []
    for ref_key in spec.get("include_refs", []):
        sel = val_selected(D, refs[ref_key])
        if not sel:
            continue
        tag, m = sel
        ref_lines.append(fmt_row(ref_labels[ref_key], tag, 1, m, bold=False, dp=dp))
    ref_block = ("\\midrule\n" + "\n".join(ref_lines) + "\n") if ref_lines else ""

    return f"""% ---- auto-generated: {name} ----
\\begin{{table}}[H]
\\centering
\\caption{{{spec['caption']}}}
\\label{{tab:exp_{name}}}
\\begin{{tabular}}{{M l c R R R R}}
\\toprule
 & & & \\multicolumn{{1}}{{c}}{{\\textbf{{Val}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Test}}}} \\\\
\\cmidrule(lr){{4-4}}\\cmidrule(lr){{5-7}}
Model & Ckpt & Ch & RMSE & RMSE & SSIM & MS-SSIM \\\\
\\midrule
{chr(10).join(body)}
{ref_block}\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


IDEAL_STACKS = [
    ("mag only", "mag"),
    ("+1VD", "1vd"),
    ("+DEM-grad", "demgrad"),
    ("+1VD +DEM-grad", "1vd_demgrad"),
    ("+DEM-relief", "demrelief"),
    ("+DEM-relief +1VD", "relief_1vd"),
]


def render_ideal_table(csv_path: Path) -> str:
    """Synthetic-operator study: flat + drape families, per-block RMSE, recovery %.

    Reads all scopes (load_evals keeps only Net), computes coupling-recovered from the
    flat mag-only floor and the drape mag-only floor:  1 - (net^2 - flat0^2)/(drape0^2 - flat0^2).
    """
    M: dict = collections.defaultdict(dict)  # (run) -> {(scope, metric): val}
    for r in csv.DictReader(open(csv_path)):
        if r["variant"] == "synuc" and r["split"] == "test":
            M[r["run"]][(r["scope"], r["metric"])] = float(r["mean"])

    def g(pfx, suf, scope, metric):
        return M.get(f"rdnpp_x3_ksa_f3_nb13_k0_p264_{pfx}_{suf}", {}).get((scope, metric))

    flat0 = g("synuc", "mag", "Net", "rmse")
    drape0 = g("synucd", "mag", "Net", "rmse")

    def line(pfx, label, suf, recovery):
        n = g(pfx, suf, "Net", "rmse")
        if n is None:
            return f"{label} & --- & --- & --- & --- & --- & --- & --- \\\\  % MISSING"
        s, ms = g(pfx, suf, "Net", "ssim"), g(pfx, suf, "Net", "msssim")
        b1, b2, b3 = (g(pfx, suf, f"B{i}", "rmse") for i in (1, 2, 3))
        rec = "---"
        if recovery and flat0 and drape0:
            rec = f"{100 * (1 - (n**2 - flat0**2) / (drape0**2 - flat0**2)):.0f}\\%"
        return f"{label} & {n:.2f} & {s:.3f} & {ms:.3f} & " f"{b1:.2f} & {b2:.2f} & {b3:.2f} & {rec} \\\\"

    flat = "\n".join(line("synuc", lab, suf, False) for lab, suf in IDEAL_STACKS)
    drape = "\n".join(line("synucd", lab, suf, True) for lab, suf in IDEAL_STACKS)
    return f"""% ---- auto-generated: ideal ----
\\begin{{table}}[H]
\\centering
\\caption{{Idealized operator study (nb13 k0, fold 3): test metrics when the LR is
generated by a known upward-continuation operator ($\\Delta z \\sim \\mathcal{{U}}[200,250]$~m,
$3{{:}}1$ decimation, $1$~nT noise), removing survey inconsistency. Flat is terrain-blind;
the drape operator injects the true terrain coupling, concentrated in the rugged block B2.}}
\\label{{tab:exp_ideal}}
\\begin{{tabular}}{{l r c c r r r r}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{\\textbf{{Net}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Per-block RMSE}}}} & \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}
Input & RMSE & SSIM & MS-SSIM & B1 & B2 & B3 & Recov. \\\\
\\midrule
\\multicolumn{{8}}{{l}}{{\\emph{{Flat (terrain-blind) operator}}}}\\\\
{flat}
\\midrule
\\multicolumn{{8}}{{l}}{{\\emph{{Drape-aware operator}}}}\\\\
{drape}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, default=ROOT_FOLDER / "experiments/ksa_ablations/latex_tables.yaml"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    D = load_evals([ROOT_FOLDER / p for p in EVAL_CSVS])
    refs = {**cfg["references"], "bicubic_wa": "bicubic_wa"}

    out = [render_table(name, spec, D, refs) for name, spec in cfg["tables"].items()]
    if "fivefold" in cfg:
        out.append(render_fold_table(cfg["fivefold"], D))
    out.append(render_ideal_table(ROOT_FOLDER / EVAL_CSVS[0]))
    text = "\n\n".join(out) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

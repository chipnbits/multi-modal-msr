"""50/50 weight-average "model soup" of a run's two best checkpoints.

One recipe is in use across the eval scripts and tables:

    <run>_soup_ssim_rmse.pt = 0.5*<run>_best.pt + 0.5*<run>_best_rmse.pt

i.e. best val SSIM averaged with best val RMSE. Both come from the same
training trajectory, so it is a plain two-point uniform soup.
"""

from __future__ import annotations

from pathlib import Path

import torch

SOUP_VARIANT = "soup_ssim_rmse"
SOUP_RECIPE = "0.5*best_ssim + 0.5*best_rmse"


def ensure_soup(ckpt_dir: Path | str, run: str, *, force: bool = False) -> Path | None:
    """Build `<run>_soup_ssim_rmse.pt` from the run's `_best` / `_best_rmse` pair.

    Returns the soup path, or None if either input is missing (run still training).
    An existing soup is reused unless `force`.
    """
    ckpt_dir = Path(ckpt_dir)
    a_path = ckpt_dir / f"{run}_best.pt"
    b_path = ckpt_dir / f"{run}_best_rmse.pt"
    out = ckpt_dir / f"{run}_{SOUP_VARIANT}.pt"
    if not (a_path.exists() and b_path.exists()):
        return None
    if out.exists() and not force:
        return out

    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    sd_a, sd_b = a["model"], b["model"]
    if sd_a.keys() != sd_b.keys():
        raise ValueError(f"state_dict keys differ between {a_path} and {b_path}")

    ck = dict(a)  # keep model_spec, config, wandb_run_id, etc. from the _best ckpt
    ck["model"] = {
        k: (
            (t.float() + sd_b[k].float()).mul_(0.5).to(t.dtype) if torch.is_floating_point(t) else t.clone()
        )  # int buffers: not averageable, keep _best's
        for k, t in sd_a.items()
    }
    ck["soup"] = {"recipe": SOUP_RECIPE, "epoch_a": a["epoch"], "epoch_b": b["epoch"]}
    ck["epoch"] = f"{a['epoch']}+{b['epoch']}"  # provenance shown in eval tables
    for key in ("optim", "sched"):
        ck.pop(key, None)
    torch.save(ck, out)
    return out

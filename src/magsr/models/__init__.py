"""Models + portable checkpoint loading.

Adding a new architecture:
  1. In its `__init__`, set `self.build_kwargs = {...}` (kwargs needed to rebuild).
  2. Re-export the class below and add it to `__all__`.
Train/eval/reconstruct scripts pick it up automatically via `load_checkpoint`.
"""

import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from magsr import ROOT_FOLDER
from magsr.models.bicubic import BicubicModel
from magsr.models.rdnpp import (
    RDNpp,
    rdnpp_default_x4,
    rdnpp_large_x4,
    rdnpp_small_x4,
)
from magsr.models.soup import SOUP_VARIANT, ensure_soup

__all__ = [
    "BicubicModel",
    "RDNpp",
    "SOUP_VARIANT",
    "rdnpp_default_x4",
    "rdnpp_large_x4",
    "rdnpp_small_x4",
    "build_from_spec",
    "ensure_soup",
    "git_sha",
    "load_checkpoint",
]


def git_sha() -> str | None:
    """Short commit of the repo, with a `-dirty` suffix if tracked files differ
    from HEAD. Stamp into checkpoints so weights trace to their source
    (`git checkout <sha>`). Returns None outside a git checkout."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT_FOLDER,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.call(
                ["git", "diff", "--quiet", "HEAD"],
                cwd=ROOT_FOLDER,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            != 0
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return f"{sha}-dirty" if dirty else sha


def build_from_spec(spec: dict) -> nn.Module:
    """Instantiate a model from a {'name': str, 'kwargs': dict} spec.

    `spec['name']` must resolve to a class/factory exported from `magsr.models`.
    """
    cls = getattr(sys.modules[__name__], spec["name"])
    return cls(**spec["kwargs"])


def load_checkpoint(
    path: Path | str,
    device: torch.device | str | None = None,
    *,
    return_ckpt: bool = False,
) -> nn.Module | tuple[nn.Module, dict]:
    """Load a checkpoint and return the rebuilt model in eval mode on `device`.

    `ckpt['model_spec']` is required (run `scripts/migrate_checkpoint.py` on
    pre-spec files). If `return_ckpt=True`, also returns the full ckpt dict
    (useful for reading metadata like `wandb_run_id`, `epoch`, or `config`).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_from_spec(ckpt["model_spec"])
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        # Prefer the construction-time sha in model_spec; fall back to the older
        # top-level git_sha stamped by pre-source_sha checkpoints.
        sha = ckpt["model_spec"].get("source_sha") or ckpt.get("git_sha")
        hint = (
            f"check out the source that trained it: `git checkout {sha}` (or migrate it)"
            if sha
            else "checkpoint predates git-sha stamping; run scripts/migrate_checkpoint.py "
            "or bisect to the training commit"
        )
        raise RuntimeError(
            f"state_dict from {path} does not match the architecture rebuilt from its "
            f"model_spec — the model code likely changed since training. {hint}.\n"
            f"Original error: {e}"
        ) from e
    if device is not None:
        model.to(device)
    model.eval()
    if return_ckpt:
        return model, ckpt
    return model

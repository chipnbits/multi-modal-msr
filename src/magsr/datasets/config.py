"""Shared loader for `configs/datasets.yaml`.

`WAConfig` and `KSAAlignedConfig` each call `load_yaml_section` with
their own section key (`"wa"`, `"ksa_aligned"`). The single YAML file
is the source of truth for patch-build + load parameters across every
build script and the dataloaders notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from magsr import ROOT_FOLDER

DEFAULT_CONFIG_PATH = ROOT_FOLDER / "configs" / "datasets.yaml"


def load_yaml_section(section: str, path: Path | None = None) -> dict[str, Any]:
    """Read the given top-level key out of `configs/datasets.yaml`."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}
    if section not in data:
        raise KeyError(f"Section {section!r} missing from {cfg_path}")
    return dict(data[section])

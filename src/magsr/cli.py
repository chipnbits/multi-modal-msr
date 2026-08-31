"""Shared argparse fragments for the experiment entry points.

Lives in the package (not `experiments/`) so scripts in subdirectories
(`experiments/wa_dataset/`, etc.) can import it without path tricks.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence


def add_channel_args(
    parser: argparse.ArgumentParser,
    *,
    defaults: Any = None,
    dem_modes: Sequence[str] = ("relief", "grad"),
) -> None:
    """Add the multichannel model-input flags shared by the train/eval/reconstruct scripts.

    The flags select the extra input channels concatenated after the magnetic channel:
    `--use-dem` (+ `--dem-mode`) feeds the DEM channel(s); `--lr-aux` appends LR product
    channels (e.g. 1VD). Omit all three for the in=1 mag-only path. At eval/reconstruct
    time the choice must match the channels the checkpoint was trained with.

    `defaults` is an optional object (a trainer's `TrainConfig`) supplying per-script
    defaults via its `use_dem` / `dem_mode` / `lr_aux` attributes; without it the flags
    default to the mag-only path. `dem_modes` restricts the accepted DEM representations
    (the WA pipeline only builds "grad").
    """
    parser.add_argument(
        "--use-dem",
        action="store_true",
        default=getattr(defaults, "use_dem", False),
        help="Feed the DEM channel(s) as extra model input. Off => in=1 mag-only, unchanged.",
    )
    parser.add_argument(
        "--dem-mode",
        choices=tuple(dem_modes),
        default=getattr(defaults, "dem_mode", "grad"),
        help="DEM representation when --use-dem: 'relief' (1ch mean-removed elevation) or "
        "'grad' (2ch slope dz/dx,dz/dy).",
    )
    parser.add_argument(
        "--lr-aux",
        nargs="*",
        default=list(getattr(defaults, "lr_aux", ())),
        metavar="PRODUCT",
        help="Auxiliary LR product channels to concatenate (e.g. 1VD ANS), each normalized "
        "independently.",
    )

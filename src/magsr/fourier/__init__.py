"""Fourier-domain potential-field tools for RTP magnetic data.

Pure-PyTorch, batched, differentiable end-to-end: upward continuation
(:func:`upward_continue_field`), equivalent-layer inversion
(:func:`fit_equivalent_layer` / :func:`upward_continue`, flat and draped),
the first-vertical-derivative operator and :class:`FVDLoss`
(:mod:`~magsr.fourier.fvd`), and operator-consistent synthetic survey
pairs (:mod:`~magsr.fourier.synth`). See ``README.md`` in this package
for the physics, numerics, usage examples, and validation notes.

Public API
----------
    EquivalentLayer            dataclass holding cached F[σ] + grid info
    fit_equivalent_layer       invert RTP tile  →  single-layer σ
    upward_continue            forward continue from a fitted layer
    upward_continue_field      level-to-level continuation primitive
    first_vertical_derivative  1VD of a potential-field grid (|k| kernel)
    FVDLoss                    L1 on the 1VD of prediction vs target
    synth_uc_pair              synthetic HR/LR pair, level-to-level UC
    synth_uc_pair_drape        synthetic HR/LR pair, draped EL continuation
"""

from magsr.fourier.equivalent_layer import (
    EquivalentLayer,
    fit_equivalent_layer,
    upward_continue,
    upward_continue_field,
)
from magsr.fourier.fvd import FVDLoss, first_vertical_derivative
from magsr.fourier.synth import synth_uc_pair, synth_uc_pair_drape

__all__ = [
    "EquivalentLayer",
    "FVDLoss",
    "first_vertical_derivative",
    "fit_equivalent_layer",
    "synth_uc_pair",
    "synth_uc_pair_drape",
    "upward_continue",
    "upward_continue_field",
]

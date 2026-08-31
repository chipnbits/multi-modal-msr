"""End-to-end correctness tests for the equivalent-layer inversion.

The cleanest test is a *round trip*:
1. Synthesise an analytic RTP dipole anomaly at observation altitude z_obs.
2. Fit a single-layer equivalent at depth z_layer (well below the dipole).
3. Use the fitted layer to forward-continue to a *different* altitude z_target.
4. Compare against the analytic dipole evaluated at z_target.

If the entire chain (detrend → taper → pad → FFT → Tikhonov inverse → forward
kernel → IFFT → un-pad → re-add plane) is correct, the residual on the
central window should be at the few-percent level — limited only by the
finite-domain truncation of the dipole's tail.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from magsr.fourier._reference import Dipole, analytic_dipole_rtp
from magsr.fourier.equivalent_layer import (
    EquivalentLayer,
    fit_equivalent_layer,
    upward_continue,
)


def _grid(n: int, dx: float) -> np.ndarray:
    return np.linspace(-(n - 1) * dx / 2, (n - 1) * dx / 2, n)


@pytest.mark.parametrize(
    "z_obs,z_layer,z_target,depth_m,max_rel_err",
    [
        # Shallow dipole, layer just below it, continue upward.
        (-100.0, 200.0, -300.0, 400.0, 0.05),
        # Deeper dipole, larger continuation hop.
        (0.0, 500.0, -500.0, 800.0, 0.05),
        # Continue from above the surface to a much higher altitude.
        (-50.0, 100.0, -800.0, 250.0, 0.05),
    ],
)
def test_round_trip_synthetic_dipole(z_obs, z_layer, z_target, depth_m, max_rel_err):
    """Fit at z_obs, continue to z_target, compare against analytic.

    Both observation and target are flat (no DEM). The equivalent layer is
    pure FFT inversion; this test gates the inversion kernel + forward
    operator chain.
    """
    n = 132
    dx = dy = 60.0
    coords = _grid(n, dx)
    d = Dipole(x=200.0, y=-150.0, z=depth_m, moment=1.0e10)

    # Convert input nT — the API expects nT.
    field_obs_T = analytic_dipole_rtp(coords, coords, z_obs, [d])
    field_obs_nT = field_obs_T * 1e9
    rtp = torch.from_numpy(field_obs_nT).to(torch.float64)

    layer = fit_equivalent_layer(
        rtp,
        dx=dx,
        dy=dy,
        z_obs=z_obs,
        z_layer=z_layer,
        eps=1e-3,
        pad_to=256,
        taper_frac=0.15,
        detrend=True,
    )
    assert isinstance(layer, EquivalentLayer)

    # Forward continuation back to z_obs should reproduce the input (sanity).
    self_consistent = upward_continue(layer, z_target=z_obs).numpy()
    centre = slice(20, 112)
    self_err = np.abs(self_consistent[centre, centre] - field_obs_nT[centre, centre]) / np.max(
        np.abs(field_obs_nT[centre, centre])
    )
    assert self_err.max() < max_rel_err, f"self-consistency failed: max rel err {self_err.max():.3%}"

    # Forward continuation to a different altitude must match analytic.
    field_target_nT = analytic_dipole_rtp(coords, coords, z_target, [d]) * 1e9
    out = upward_continue(layer, z_target=z_target).numpy()
    rel_err = np.abs(out[centre, centre] - field_target_nT[centre, centre]) / np.max(
        np.abs(field_target_nT[centre, centre])
    )
    assert rel_err.max() < max_rel_err, (
        f"z_obs={z_obs} z_layer={z_layer} z_target={z_target}: "
        f"max rel err {rel_err.max():.3%} > {max_rel_err:.1%}"
    )


def test_layer_depth_invariance():
    """Different layer depths must produce different sigma maps but the
    *forward continuation* at the original observation altitude must agree
    (the equivalent layer is non-unique; only the field above is observable).
    """
    n = 132
    dx = dy = 60.0
    coords = _grid(n, dx)
    d = Dipole(x=0.0, y=0.0, z=600.0, moment=1.0e10)
    field_obs_nT = analytic_dipole_rtp(coords, coords, 0.0, [d]) * 1e9
    rtp = torch.from_numpy(field_obs_nT).to(torch.float64)

    outs = []
    for z_layer in (200.0, 400.0, 500.0):
        layer = fit_equivalent_layer(rtp, dx=dx, dy=dy, z_obs=0.0, z_layer=z_layer, eps=1e-3, pad_to=256)
        # Re-forward at observation altitude.
        outs.append(upward_continue(layer, z_target=0.0).numpy())

    centre = slice(20, 112)
    base = outs[0][centre, centre]
    base_max = np.max(np.abs(base))
    for o in outs[1:]:
        rel = np.abs(o[centre, centre] - base) / base_max
        assert rel.max() < 0.01, f"layer-depth invariance broken: max rel err {rel.max():.3%}"


def test_invalid_layer_above_observer():
    rtp = torch.zeros(132, 132)
    with pytest.raises(ValueError, match="z_layer"):
        fit_equivalent_layer(rtp, dx=60.0, dy=60.0, z_obs=0.0, z_layer=-100.0)


def test_invalid_target_below_layer():
    rtp = torch.zeros(132, 132)
    layer = fit_equivalent_layer(rtp, dx=60.0, dy=60.0, z_obs=0.0, z_layer=500.0)
    with pytest.raises(ValueError, match="z_target"):
        upward_continue(layer, z_target=600.0)


def test_batch_dim_round_trip():
    """A `(B=3, H, W)` batch must produce a `(B, H, W)` output that matches
    per-tile fitting."""
    n = 132
    dx = dy = 60.0
    coords = _grid(n, dx)
    dipoles = [
        Dipole(x=0.0, y=0.0, z=400.0, moment=1.0e10),
        Dipole(x=200.0, y=200.0, z=600.0, moment=-2.0e10),
        Dipole(x=-100.0, y=300.0, z=300.0, moment=5.0e9),
    ]
    fields = torch.stack(
        [
            torch.from_numpy(analytic_dipole_rtp(coords, coords, 0.0, [d])).to(torch.float64) * 1e9
            for d in dipoles
        ]
    )
    batched = fit_equivalent_layer(fields, dx=dx, dy=dy, z_obs=0.0, z_layer=700.0)
    out_batched = upward_continue(batched, z_target=-200.0)

    # Per-tile.
    for i, f in enumerate(fields):
        single = fit_equivalent_layer(f, dx=dx, dy=dy, z_obs=0.0, z_layer=700.0)
        out_single = upward_continue(single, z_target=-200.0)
        torch.testing.assert_close(out_batched[i], out_single, atol=1e-9, rtol=1e-7)

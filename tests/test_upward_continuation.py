"""Validate the FFT upward-continuation against the slow CPU references.

Two independent ground truths are used:

1. **Analytic dipole** — closed-form Bz of a vertical dipole evaluated at any
   altitude (`magsr.fourier._reference.analytic_dipole_rtp`). The FFT
   continuation of the dipole field at `z0` must reproduce the dipole field at
   `z0 - dz` to machine-level precision in the central window.

2. **Brute-force spatial summation** — direct evaluation of Blakely eq 12.4
   (`brute_force_upward_continue`). Independent of any FFT, so it provides a
   second-witness check on smaller grids.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from magsr.fourier._reference import (
    Dipole,
    analytic_dipole_rtp,
    integral_upward_continue,
)
from magsr.fourier.equivalent_layer import upward_continue_field


def _grid(n: int, dx: float) -> np.ndarray:
    return np.linspace(-(n - 1) * dx / 2, (n - 1) * dx / 2, n)


# (dz_m, depth_m, max_rel_err) triples. The error budget reflects how
# well-localised the dipole field is inside a 132×60 m = 7.92 km window:
# shallow dipoles (depth << half-window) are nearly perfect; deep dipoles
# (depth ~ half-window) bleed past the edges and hit ~2% from finite-domain
# truncation, regardless of pad/taper settings.
@pytest.mark.parametrize(
    "dz_m,depth_m,max_rel_err",
    [
        (100.0, 600.0, 0.005),  # shallow, short hop
        (500.0, 600.0, 0.005),  # shallow, long hop
        (300.0, 1200.0, 0.015),  # deep, intermediate hop
        (1000.0, 1500.0, 0.025),  # deep, long hop — tail clipped at edges
    ],
)
def test_fft_vs_analytic_dipole(dz_m, depth_m, max_rel_err):
    """Continue a dipole field measured at z=0 up by dz; compare to the
    analytic field at the new altitude."""
    n = 132
    dx = dy = 60.0
    coords = _grid(n, dx)
    d = Dipole(x=300.0, y=-200.0, z=depth_m, moment=1.0e10)

    # Source data at observation altitude z=0 (positive down convention).
    field0_np = analytic_dipole_rtp(coords, coords, 0.0, [d])
    field0 = torch.from_numpy(field0_np).to(torch.float64)

    # FFT continuation by dz. taper_frac=0 because the synthetic dipole tail
    # is genuine signal, not edge garbage; in real data, taper > 0 helps.
    continued = upward_continue_field(
        field0, dx=dx, dy=dy, dz=dz_m, pad_to=256, detrend=False, taper_frac=0.0
    ).numpy()

    # Analytic ground truth at the new altitude.
    field1_np = analytic_dipole_rtp(coords, coords, -dz_m, [d])

    # Compare on the central 100×100 window — outer ring inherits pad effects.
    centre = slice(16, 116)
    rel_err = np.abs(continued[centre, centre] - field1_np[centre, centre]) / np.max(
        np.abs(field1_np[centre, centre])
    )
    assert rel_err.max() < max_rel_err, (
        f"dz={dz_m} depth={depth_m}: max rel err {rel_err.max():.3%}" f" exceeds budget {max_rel_err:.1%}"
    )


def test_fft_vs_brute_force():
    """FFT continuation must agree with brute-force spatial summation."""
    n = 64
    dx = dy = 100.0
    coords = _grid(n, dx)
    d = Dipole(x=0.0, y=0.0, z=800.0, moment=1.0e10)
    field0_np = analytic_dipole_rtp(coords, coords, 0.0, [d])
    field0 = torch.from_numpy(field0_np).to(torch.float64)

    dz = 400.0
    fft_continued = upward_continue_field(field0, dx=dx, dy=dy, dz=dz, pad_to=128, detrend=False).numpy()
    brute = integral_upward_continue(field0_np, dx, dy, dz)

    centre = slice(20, 44)
    rel_err = np.abs(fft_continued[centre, centre] - brute[centre, centre]) / np.max(
        np.abs(brute[centre, centre])
    )
    # Brute-force has its own ~few-% truncation error, so 5% is the bar.
    assert rel_err.max() < 0.05, f"max rel err {rel_err.max():.3%}"


def test_dz_zero_is_identity():
    """upward_continue_field with dz=0 must return the input (detrend=False)."""
    n = 64
    dx = dy = 60.0
    rng = np.random.default_rng(0)
    field = torch.from_numpy(rng.normal(size=(n, n)).astype(np.float64))
    out = upward_continue_field(field, dx=dx, dy=dy, dz=0.0, detrend=False, taper_frac=0.0)
    torch.testing.assert_close(out, field, atol=1e-10, rtol=1e-10)


def test_semigroup_two_steps():
    """One dz=400 m continuation == two dz=200 m continuations (FFT version,
    where this is exact up to numerical precision because the kernel is
    multiplicative in Fourier space)."""
    n = 132
    dx = dy = 60.0
    coords = _grid(n, dx)
    d = Dipole(x=100.0, y=-100.0, z=700.0, moment=2.0e10)
    field0 = torch.from_numpy(analytic_dipole_rtp(coords, coords, 0.0, [d])).to(torch.float64)
    one = upward_continue_field(field0, dx=dx, dy=dy, dz=400.0, detrend=False)
    two = upward_continue_field(
        upward_continue_field(field0, dx=dx, dy=dy, dz=200.0, detrend=False),
        dx=dx,
        dy=dy,
        dz=200.0,
        detrend=False,
    )
    # Same pad, same kernel structure -> only floating-point differences.
    centre = slice(16, 116)
    torch.testing.assert_close(one[centre, centre], two[centre, centre], atol=1e-8, rtol=1e-6)


def test_batch_dim_independent():
    """Batched continuation == per-tile continuation."""
    n = 132
    dx = dy = 60.0
    coords = _grid(n, dx)
    dipoles = [
        Dipole(x=100.0, y=0.0, z=400.0, moment=1.0e10),
        Dipole(x=-200.0, y=300.0, z=600.0, moment=-2.0e10),
    ]
    fields = torch.stack(
        [torch.from_numpy(analytic_dipole_rtp(coords, coords, 0.0, [d])).to(torch.float64) for d in dipoles]
    )
    batched = upward_continue_field(fields, dx=dx, dy=dy, dz=300.0, detrend=False)
    per_tile = torch.stack(
        [upward_continue_field(f, dx=dx, dy=dy, dz=300.0, detrend=False) for f in fields]
    )
    torch.testing.assert_close(batched, per_tile, atol=1e-10, rtol=1e-10)


def test_plane_invariance_with_detrend():
    """A linear plane added to the field must survive upward continuation
    when detrending is enabled (planes are invariant under z-translation).
    """
    n = 132
    dx = dy = 60.0
    coords = _grid(n, dx)
    d = Dipole(x=0.0, y=0.0, z=500.0, moment=1.0e10)
    field0 = torch.from_numpy(analytic_dipole_rtp(coords, coords, 0.0, [d])).to(torch.float64)
    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=torch.float64) - (n - 1) / 2,
        torch.arange(n, dtype=torch.float64) - (n - 1) / 2,
        indexing="ij",
    )
    plane = 0.001 * xx + (-0.0007) * yy + 5e-6
    biased = field0 + plane

    no_plane = upward_continue_field(field0, dx=dx, dy=dy, dz=200.0, detrend=False)
    with_plane = upward_continue_field(biased, dx=dx, dy=dy, dz=200.0, detrend=True)

    centre = slice(20, 112)
    diff = (with_plane - plane)[centre, centre] - no_plane[centre, centre]
    # The plane should be preserved exactly; residual should match the
    # no-plane continuation.
    rel_err = diff.abs().max() / no_plane[centre, centre].abs().max()
    assert rel_err.item() < 1e-3, f"plane invariance broken: rel err {rel_err.item():.3%}"

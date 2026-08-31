"""Sanity tests for the slow CPU references themselves.

These don't test our FFT operators; they verify the two ground-truth
implementations agree with each other. If they do, both can be trusted as
references for downstream FFT validation.
"""

import math

import numpy as np
import pytest

from magsr.fourier._reference import (
    CM,
    Dipole,
    analytic_dipole_rtp,
    integral_upward_continue,
)


def test_analytic_dipole_axial_value():
    """At (x=xd, y=yd, z=zd-h) directly above the dipole, Bz reduces to a
    simple closed form: Bz = Cm * m * 2 / h^3 (with h > 0 = distance above).
    """
    d = Dipole(x=0.0, y=0.0, z=200.0, moment=1.0e9)  # 1e9 A m^2 (huge for sanity)
    x = np.array([0.0])
    y = np.array([0.0])
    z_obs = 100.0  # 100 m above ground; dipole is at z=200 m below ground
    h = d.z - z_obs  # 100 m vertical separation
    expected = CM * d.moment * 2.0 / h**3
    got = analytic_dipole_rtp(x, y, z_obs, [d])[0, 0]
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_analytic_dipole_far_field_decay():
    """At a fixed altitude, |ΔT| decays as 1/r^3 along the equatorial line of
    the dipole in the genuine far field (r >> dipole depth).
    """
    d = Dipole(x=0.0, y=0.0, z=100.0, moment=1.0)
    z_obs = 100.0  # equatorial (same altitude as dipole) -> simplest 1/r^3 form
    rs = np.array([5_000.0, 10_000.0, 20_000.0, 40_000.0])  # well beyond depth
    fields = analytic_dipole_rtp(rs, np.zeros(1), z_obs, [d])[0]
    # On the equatorial line at the dipole's altitude, Bz = -Cm * m / r^3 exactly.
    # Each doubling of r should drop |ΔT| by exactly 8.
    ratios = np.abs(fields[:-1] / fields[1:])
    np.testing.assert_allclose(ratios, 8.0, rtol=1e-6)


def test_brute_force_upward_continue_agrees_with_analytic():
    """Forward an analytic dipole at altitude z0, brute-force continue up by
    dz, compare against the analytic dipole at z0 - dz. This is the cross-check
    that validates the slow reference.
    """
    # Small grid for speed (32x32 is ~1 million ops, OK).
    n = 32
    dx = dy = 50.0  # 50 m pixels => 1.6 km square
    extent = (n - 1) * dx / 2.0
    coords = np.linspace(-extent, extent, n)
    d = Dipole(x=200.0, y=-300.0, z=400.0, moment=5.0e9)

    z0 = 0.0  # surface
    dz = 200.0  # continue up 200 m
    field_z0 = analytic_dipole_rtp(coords, coords, z0, [d])
    field_z1_brute = integral_upward_continue(field_z0, dx, dy, dz)
    field_z1_analytic = analytic_dipole_rtp(coords, coords, z0 - dz, [d])

    # Compare central window — edges ringing from finite domain truncation
    # is unavoidable for the brute-force version.
    central = slice(8, 24)
    rel_err = np.abs(field_z1_brute[central, central] - field_z1_analytic[central, central]) / np.max(
        np.abs(field_z1_analytic[central, central])
    )
    # Brute-force has truncation error from the finite domain. 5% is a
    # generous bound for a 1.6 km square grid + 200 m continuation.
    assert rel_err.max() < 0.05, f"max rel err {rel_err.max():.3%}"


def test_brute_force_self_consistency_two_step():
    """Two successive 200 m continuations should equal one 400 m continuation
    (semigroup property of upward continuation).

    Brute-force convolution requires ``dz / dx`` large enough that the
    Poisson kernel is resolved on the grid: a too-small `dz` undersamples
    the kernel and the second step amplifies the error. Here `dx=100 m`,
    `dz=200 m` (ratio 2). The dipole sits at 800 m depth so its field
    varies on a scale much larger than the pixel.
    """
    n = 64
    dx = dy = 100.0  # 100 m pixels, total 6.4 km grid
    coords = np.linspace(-(n - 1) * dx / 2, (n - 1) * dx / 2, n)
    d = Dipole(x=0.0, y=0.0, z=800.0, moment=1.0e10)
    field0 = analytic_dipole_rtp(coords, coords, 0.0, [d])
    one_step = integral_upward_continue(field0, dx, dy, 400.0)
    two_step = integral_upward_continue(integral_upward_continue(field0, dx, dy, 200.0), dx, dy, 200.0)
    centre = slice(n // 2 - 12, n // 2 + 12)
    rel_err = np.abs(one_step[centre, centre] - two_step[centre, centre]) / np.max(
        np.abs(one_step[centre, centre])
    )
    assert rel_err.max() < 0.05, f"max rel err {rel_err.max():.3%}"

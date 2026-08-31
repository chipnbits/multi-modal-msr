"""CPU references implementations for upward continuation and RTP anomalies.

Two ground-truth implementations:

1. `analytic_dipole_rtp` — closed-form RTP anomaly of a vertical magnetic
   dipole. RTP means both magnetization and ambient field point along +z
   (down), so the total-field anomaly equals B_z (Blakely eq 12.23 with
   `Theta_m = Theta_f = 1` and Cm = mu0/4pi). Used to validate both the
   forward upward continuation and the equivalent-layer inversion: any
   tile generated this way at altitude z0 must be reproducible at any
   other altitude z1 via continuation.

2. `integral_upward_continue` — direct evaluation of Blakely's
   upward-continuation integral (eq 12.4):

       U(x, y, z0 - dz) = (dz / 2pi) * ∬ U(x', y', z0) /
                          [(x-x')^2 + (y-y')^2 + dz^2]^(3/2) dx' dy'

   This is O(N^4) on an N×N grid but is independent of any FFT, so it
   provides an independent check of the FFT-based continuation. We use it
   on small (e.g. 32×32) grids where the cost is acceptable.

Both routines run on numpy / CPU and return numpy arrays in nT (when the
input is in nT) or Tesla (when in Tesla); they are unit-agnostic — what
goes in is what comes out, scaled by the kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Cm = mu0 / (4 pi) in SI units (T m / A).
CM = 1e-7


@dataclass
class Dipole:
    """Vertical magnetic dipole for RTP synthetics.

    Position is in metres in a right-handed (x east, y north, z down) frame.
    Moment is in ampere * metre^2 (A m^2). For a uniformly magnetised sphere
    of radius a (m) and magnetization M (A/m), `moment = (4/3) pi a^3 M`.
    """

    x: float  # m
    y: float  # m
    z: float  # m, +down (positive = below surface)
    moment: float  # A m^2, +z direction (pointing down)


def analytic_dipole_rtp(
    x: np.ndarray,
    y: np.ndarray,
    z_obs: float | np.ndarray,
    dipoles: list[Dipole],
) -> np.ndarray:
    """Closed-form RTP total-field anomaly for vertical dipoles.

    Parameters
    ----------
    x, y : np.ndarray
        Observation grid coordinates in metres. Either 1-D arrays that get
        broadcast (`np.meshgrid`-style) or already-broadcast 2-D arrays.
    z_obs : float or np.ndarray
        Observation altitude in metres, +down convention. Can be a scalar
        (flat surface) or an array matching the broadcast shape of x/y
        (draped surface).
    dipoles : list[Dipole]
        Sources. Their fields are summed.

    Returns
    -------
    np.ndarray
        Anomaly in Tesla. Multiply by 1e9 to get nT.

    Notes
    -----
    For a vertical dipole with moment m̂ = +ẑ at position (xd, yd, zd), the
    magnetic induction at observation point (x, y, z) is::

        B = Cm * m * [3(m̂·r̂)r̂ - m̂] / r^3

    so the z-component (the RTP anomaly) is::

        Bz = Cm * m * (2*(z-zd)^2 - (x-xd)^2 - (y-yd)^2) / r^5

    All distances in metres, moment in A m^2, output in Tesla.
    """
    if x.ndim == 1 and y.ndim == 1:
        xx, yy = np.meshgrid(x, y, indexing="xy")
    else:
        xx, yy = x, y
    zz = np.broadcast_to(np.asarray(z_obs, dtype=float), xx.shape)
    out = np.zeros(xx.shape, dtype=float)
    for d in dipoles:
        dx = xx - d.x
        dy = yy - d.y
        dz = zz - d.z
        r2 = dx * dx + dy * dy + dz * dz
        r5 = r2 * r2 * np.sqrt(r2)
        out += CM * d.moment * (2.0 * dz * dz - dx * dx - dy * dy) / r5
    return out


def integral_upward_continue(
    field: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Spatial-domain upward continuation (Blakely eq 12.4).

    Convolves the input grid with the upward-continuation Green's function

        psi_u(x, y, dz) = (dz / 2pi) / (x^2 + y^2 + dz^2)^(3/2)

    via direct double summation (no FFT). O(N^4) on an N×N grid.

    Parameters
    ----------
    field : np.ndarray
        Potential-field values on a regular `(H, W)` grid at altitude z0.
        Units are arbitrary; output uses the same units.
    dx, dy : float
        Pixel size in metres along x (last axis) and y (second-to-last).
    dz : float
        Continuation distance in metres, +up. Must be > 0.

    Returns
    -------
    np.ndarray
        `(H, W)` continued field at altitude `z0 - dz` (i.e. higher).

    Notes
    -----
    This is a brute-force reference for validating the FFT version; it has
    no edge handling and assumes the field is zero outside the input window.
    Use it on small grids (≤ 64×64) only.
    """
    if dz <= 0:
        raise ValueError("dz must be > 0 (upward continuation)")
    h, w = field.shape
    out = np.zeros_like(field, dtype=float)
    cell_area = dx * dy
    coef = dz / (2.0 * math.pi)
    # Precompute relative offsets to all source pixels from a single target
    # pixel. We then shift these by the target index to get distances.
    j_grid, i_grid = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    for ti in range(h):
        for tj in range(w):
            ddx = (j_grid - tj) * dx
            ddy = (i_grid - ti) * dy
            denom = (ddx * ddx + ddy * ddy + dz * dz) ** 1.5
            out[ti, tj] = coef * np.sum(field / denom) * cell_area
    return out

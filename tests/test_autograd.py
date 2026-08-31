"""End-to-end differentiability tests.

The whole point of staying torch-native (instead of harmonica / SimPEG) is so
the inversion + continuation chain can sit inside a learning loop and have
useful gradients flow through it. These tests gate that contract.

Currently differentiable:
- upward_continue_field (the level-to-level primitive)
- fit_equivalent_layer in BOTH flat and drape paths, w.r.t. ``rtp_tile``
- upward_continue, w.r.t. the cached ``sigma_hat`` produced by the fit

Currently *not* differentiable (all are float scalars cast at the kernel
boundary; can be lifted by accepting tensor versions if needed):
- z_obs, z_layer, eps, dx, dy
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from magsr.fourier._reference import Dipole, analytic_dipole_rtp
from magsr.fourier.equivalent_layer import (
    fit_equivalent_layer,
    upward_continue,
    upward_continue_field,
)


def _grid(n: int, dx: float) -> np.ndarray:
    return np.linspace(-(n - 1) * dx / 2, (n - 1) * dx / 2, n)


def _dipole_field(n: int = 64) -> np.ndarray:
    coords = _grid(n, 60.0)
    return analytic_dipole_rtp(coords, coords, 0.0, [Dipole(0.0, 0.0, 500.0, 1e10)]) * 1e9


def test_upward_continue_field_gradient_flows():
    field = torch.from_numpy(_dipole_field()).to(torch.float64).requires_grad_(True)
    out = upward_continue_field(field, dx=60.0, dy=60.0, dz=300.0, pad_to=128, detrend=False)
    out.pow(2).sum().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    # Most pixels should have nonzero gradient (the few zero ones live in the
    # padded zero ring, since their value contributes only via cropping).
    assert (field.grad.abs() > 0).float().mean().item() > 0.9


def test_fit_flat_gradient_flows():
    field = torch.from_numpy(_dipole_field()).to(torch.float64).requires_grad_(True)
    layer = fit_equivalent_layer(field, dx=60.0, dy=60.0, z_obs=0.0, z_layer=600.0, pad_to=128)
    out = upward_continue(layer, z_target=-200.0)
    out.pow(2).sum().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    assert field.grad.abs().max().item() > 0.0


def test_fit_drape_pcg_gradient_flows():
    """The PCG inner loop uses .item() for stopping criteria *only* — the
    actual σ updates are tensor ops, so autograd traces through them.
    """
    field = torch.from_numpy(_dipole_field()).to(torch.float64).requires_grad_(True)
    coords = _grid(64, 60.0)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    ext = coords.max() - coords.min()
    drape = -100.0 + 50.0 * np.cos(2 * np.pi * xx / ext) * np.sin(2 * np.pi * yy / ext)
    z_obs = torch.from_numpy(drape).to(torch.float64)

    layer = fit_equivalent_layer(
        field, dx=60.0, dy=60.0, z_obs=z_obs, z_layer=600.0, pad_to=128, cg_iters=20
    )
    out = upward_continue(layer, z_target=z_obs - 200.0)
    out.pow(2).sum().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    assert field.grad.abs().max().item() > 0.0


def test_n_layer_drape_forward_gradient_flows():
    """N-layer chessboard forward must keep the autograd graph clean.

    The gather + linear-interp blend is differentiable via standard
    linear-interp behavior — the integer slice index is quantised but
    `frac` carries the z-derivative. Gradient w.r.t. ``rtp_tile`` must
    flow through the entire fit + N-layer forward chain.
    """
    n = 64
    coords = _grid(n, 60.0)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    ext = coords.max() - coords.min()
    drape_a = -100.0 + 60.0 * np.cos(2 * np.pi * xx / ext) * np.sin(2 * np.pi * yy / ext)
    drape_b = -150.0 + 100.0 * np.sin(2 * np.pi * xx / ext) * np.cos(2 * np.pi * yy / ext)
    field_np = analytic_dipole_rtp(coords, coords, drape_a, [Dipole(0, 0, 600, 1e10)]) * 1e9

    field = torch.from_numpy(field_np).to(torch.float64).requires_grad_(True)
    z_obs = torch.from_numpy(drape_a).to(torch.float64)
    z_target = torch.from_numpy(drape_b).to(torch.float64)

    layer = fit_equivalent_layer(
        field, dx=60.0, dy=60.0, z_obs=z_obs, z_layer=900.0, pad_to=128, cg_iters=15
    )
    out = upward_continue(layer, z_target=z_target, n_layers=8)
    out.pow(2).sum().backward()

    assert field.grad is not None
    assert torch.isfinite(field.grad).all()
    # > 90% of pixels should have nonzero gradient (some edge pixels at the
    # tukey-zero ring contribute via the FFT but their gradient may be tiny).
    nonzero_frac = (field.grad.abs() > 0).float().mean().item()
    assert nonzero_frac > 0.9, f"only {nonzero_frac:.1%} of grad entries nonzero"
    assert field.grad.abs().max().item() > 0.0


def _drape_setup(n: int = 64):
    """Shared synthetic drape geometry for the implicit-backward tests."""
    coords = _grid(n, 60.0)
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    ext = coords.max() - coords.min()
    drape = -100.0 + 50.0 * np.cos(2 * np.pi * xx / ext) * np.sin(2 * np.pi * yy / ext)
    field_np = analytic_dipole_rtp(coords, coords, drape, [Dipole(0, 0, 600, 1e10)]) * 1e9
    field = torch.from_numpy(field_np).to(torch.float64)
    z_obs = torch.from_numpy(drape).to(torch.float64)
    return field, z_obs


def _drape_grad(field: torch.Tensor, z_obs: torch.Tensor, *, implicit: bool, cg_iters: int):
    x = field.clone().requires_grad_(True)
    layer = fit_equivalent_layer(
        x,
        dx=60.0,
        dy=60.0,
        z_obs=z_obs,
        z_layer=600.0,
        pad_to=128,
        cg_iters=cg_iters,
        implicit_grad=implicit,
    )
    out = upward_continue(layer, z_target=z_obs - 200.0, n_layers=8)
    out.pow(2).mean().backward()
    return x.grad.clone()


def test_drape_implicit_matches_unrolled_gradient():
    """Implicit-function backward vs the unrolled-CG backward.

    The implicit gradient is that of the *converged* solution map, the
    unrolled one is exact for the truncated iteration; they differ by
    O(CG residual) plus fp32 iteration noise. At the calibrated iteration
    counts the two must agree closely in direction and to ~1e-2 in norm.
    """
    field, z_obs = _drape_setup()
    g_imp = _drape_grad(field, z_obs, implicit=True, cg_iters=30)
    g_unr = _drape_grad(field, z_obs, implicit=False, cg_iters=30)
    rel = (g_imp - g_unr).norm() / g_unr.norm()
    cos = (g_imp * g_unr).sum() / (g_imp.norm() * g_unr.norm())
    assert torch.isfinite(g_imp).all()
    assert rel < 2e-2, f"implicit vs unrolled grad rel err {rel:.3e}"
    assert cos > 0.999, f"implicit vs unrolled grad cosine {cos:.6f}"


def test_drape_implicit_forward_identical_to_unrolled():
    """The implicit Function must not change forward values at all — it runs
    the very same solver, only outside the autograd tape."""
    field, z_obs = _drape_setup()
    outs = []
    for implicit in (True, False):
        x = field.clone().requires_grad_(True)
        layer = fit_equivalent_layer(
            x,
            dx=60.0,
            dy=60.0,
            z_obs=z_obs,
            z_layer=600.0,
            pad_to=128,
            cg_iters=15,
            implicit_grad=implicit,
        )
        outs.append(upward_continue(layer, z_target=z_obs - 200.0, n_layers=8).detach())
    torch.testing.assert_close(outs[0], outs[1], atol=0.0, rtol=0.0)


def test_drape_directional_fd():
    """Directional finite difference through the FULL drape pipeline.

    The pipeline is exactly linear in the input tile (fixed-iteration CG on
    a linear system), so a quadratic loss gives central differences with
    zero truncation error at any step — a large step is used to rise above
    the fp32-internal noise floor. The FD measures the *truncated* map, so
    it arbitrates the UNROLLED gradient tightly; the implicit gradient (of
    the converged map) is allowed its documented O(CG residual) deviation.
    """
    field, z_obs = _drape_setup()
    z_target = z_obs - 200.0

    def loss_of(x: torch.Tensor, implicit: bool = True) -> torch.Tensor:
        layer = fit_equivalent_layer(
            x,
            dx=60.0,
            dy=60.0,
            z_obs=z_obs,
            z_layer=600.0,
            pad_to=128,
            cg_iters=30,
            implicit_grad=implicit,
        )
        return upward_continue(layer, z_target=z_target, n_layers=8).pow(2).mean()

    g = torch.Generator().manual_seed(0)
    v = torch.randn(field.shape, generator=g, dtype=torch.float64)
    v /= v.norm()
    # Step sizing: the synthetic dipole field is O(5000 nT), so each fp32-internal
    # forward carries ~1e-7-relative rounding roughness on the loss. The FD
    # numerator must clear that floor; with exact linearity any h is truncation-
    # free, so pick h large enough that 2h·|∂_v L| >> fp32 noise.
    h = 100.0
    with torch.no_grad():
        fd = (loss_of(field + h * v) - loss_of(field - h * v)) / (2 * h)

    ads = {}
    for implicit in (False, True):
        x = field.clone().requires_grad_(True)
        loss_of(x, implicit=implicit).backward()
        ads[implicit] = (x.grad * v).sum()

    rel_unrolled = abs(fd - ads[False]) / abs(fd)
    rel_implicit = abs(fd - ads[True]) / abs(fd)
    # Unrolled differentiates exactly what FD measures — tight agreement.
    assert rel_unrolled < 5e-3, f"FD={fd:.4e} vs unrolled={ads[False]:.4e} rel={rel_unrolled:.3e}"
    # Implicit is the converged-map gradient — same direction, O(residual) offset.
    assert rel_implicit < 1e-1, f"FD={fd:.4e} vs implicit={ads[True]:.4e} rel={rel_implicit:.3e}"


def test_drape_implicit_falls_back_when_z_obs_requires_grad():
    """A differentiable z_obs must route to the unrolled solver so geometry
    gradients keep flowing (the implicit path treats the operator as
    constant)."""
    field, z_obs = _drape_setup()
    x = field.clone().requires_grad_(True)
    zg = z_obs.clone().requires_grad_(True)
    layer = fit_equivalent_layer(x, dx=60.0, dy=60.0, z_obs=zg, z_layer=600.0, pad_to=128, cg_iters=10)
    upward_continue(layer, z_target=float(z_obs.min()) - 200.0).pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().max() > 0
    assert zg.grad is not None and zg.grad.abs().max() > 0


def test_finite_difference_check_on_flat_path():
    """Compare autograd vs central finite differences at one entry of
    rtp_tile. They must agree to a few percent (small grid, double precision)."""
    n = 32
    coords = _grid(n, 100.0)
    field_np = analytic_dipole_rtp(coords, coords, 0.0, [Dipole(0.0, 0.0, 600.0, 1e10)]) * 1e9
    field = torch.from_numpy(field_np).to(torch.float64).requires_grad_(True)

    def loss_of(x: torch.Tensor) -> torch.Tensor:
        layer = fit_equivalent_layer(x, dx=100.0, dy=100.0, z_obs=0.0, z_layer=900.0, pad_to=64)
        return upward_continue(layer, z_target=-300.0).pow(2).sum()

    loss = loss_of(field)
    loss.backward()
    # Central FD at (16, 16).
    eps = 1.0
    f_plus = field.detach().clone()
    f_plus[16, 16] += eps
    f_minus = field.detach().clone()
    f_minus[16, 16] -= eps
    # Gradient values are large here (data scaled in nT); FD with 1 nT step.
    fd_grad = (loss_of(f_plus) - loss_of(f_minus)).item() / (2 * eps)
    ad_grad = field.grad[16, 16].item()
    rel_err = abs(fd_grad - ad_grad) / max(abs(fd_grad), 1e-30)
    assert rel_err < 1e-3, f"FD={fd_grad:.4e} vs autograd={ad_grad:.4e} rel_err={rel_err:.3e}"

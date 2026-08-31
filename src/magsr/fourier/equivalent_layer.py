"""Equivalent-layer inversion + upward continuation for RTP magnetic data.

PyTorch implementation of Blakely (1996) Ch. 12, specialised to
reduced-to-pole geometry.

1. :func:`upward_continue_field` — level-to-level continuation primitive
   (Blakely eq 12.6 / 12.8). Pad → rfft2 → ``exp(-|k|·dz)`` → irfft2.
2. Drape-PCG operators (private helpers) — ``A`` / ``A^T`` of the 3-level
   Lagrange-quadratic chessboard, plus the preconditioned-CG solver.
3. :class:`EquivalentLayer` + :func:`fit_equivalent_layer` — the inversion.
4. :func:`upward_continue` (with N-layer linear chessboard helpers) —
   the user-facing forward operator.

Numerical convention
--------------------
Every routine works in **nT-native** units so the kernel prefactor
``2π · μ₀/(4π) · 1e9 ≈ 6.28e-1`` (encoded in ``_CM_NT``) lives at O(1)
instead of O(1e-7). This keeps every CG inner product well inside
float64's dynamic range — the SI version silently stagnates because
inner products fall below float64 precision.

Conjugate-gradient solver
-------------------------
Drape inversions solve ``(AᵀA + λI) σ = Aᵀ b`` by preconditioned CG.
The preconditioner approximates the diagonal of ``AᵀA + λI`` by the
simple average of the three flat-kernel squares,
``⟨K²⟩ = (K_lo² + K_mid² + K_hi²) / 3`` — exact in the flat limit
(1 iter, all three K's coincide) and gives geometric convergence at
~10× per iter on rich-source drape tiles. The earlier ``K_mid²`` variant
was replaced because ``⟨K²⟩`` cuts the CG residual by ~4-5 orders of
magnitude at iter 15. The Cordell-Grauch Picard iteration was also
tried and rejected: it diverges for ``|k|·max_dz > 0.7`` which is most
of the spectrum on a 60 m / 132×132 tile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from magsr.fourier._fft_utils import (
    apply_plane,
    bilinear_fit,
    crop_centered,
    extend_feathered,
    make_wavenumbers_rfft,
    pad_centered,
    tukey2d,
)

__all__ = [
    "EquivalentLayer",
    "fit_equivalent_layer",
    "upward_continue",
    "upward_continue_field",
]


# ---------------------------------------------------------------------------
# Level-to-level upward continuation (Blakely eq 12.6 / 12.8)
# ---------------------------------------------------------------------------


def upward_continue_field(
    field: Tensor,
    dx: float,
    dy: float,
    dz: float,
    *,
    pad_to: int | None = None,
    taper_frac: float = 0.15,
    detrend: bool = True,
) -> Tensor:
    """Upward-continue a potential-field grid by ``dz`` metres.

    Implements Blakely's level-to-level filter
    ``F[U_u] = F[U] · exp(-|k|·dz)`` with edge-handling consistent with the
    rest of this module (bilinear detrend → Tukey window → zero-pad → FFT).

    Parameters
    ----------
    field : Tensor
        ``(H, W)`` or ``(B, H, W)`` real grid in arbitrary units (the kernel
        is unitless).
    dx, dy : float
        Pixel size in metres.
    dz : float
        Continuation distance in metres. ``dz > 0`` is upward (away from
        sources). Downward continuation (``dz < 0``) is supported but is
        unstable at high wavenumbers — use the regularised
        :func:`fit_equivalent_layer` path instead when working in that
        direction.
    pad_to : int, optional
        Pad the grid to this size before the FFT. Defaults to the next power
        of two ≥ 1.5 × ``max(H, W)``.
    taper_frac : float
        Tukey-window fraction applied to the (un-padded) data window before
        zero-padding. ``0`` disables tapering. Default 0.15.
    detrend : bool
        Subtract a least-squares bilinear plane before the FFT and add it
        back afterwards. Default ``True``. Disable when the input is already
        zero-mean and zero-trend (synthetics, residuals).

    Returns
    -------
    Tensor
        Continued field, same shape and dtype as the input.
    """
    if field.ndim == 2:
        squeeze = True
        field = field.unsqueeze(0)
    else:
        squeeze = False
    *batch, h, w = field.shape
    device, dtype = field.device, field.dtype

    if pad_to is None:
        pad_to = _next_pow2(int(1.5 * max(h, w)))
    if pad_to < max(h, w):
        raise ValueError(f"pad_to={pad_to} smaller than grid {h}x{w}")

    # ----- bilinear detrend (in-units; restored at the end) -----
    if detrend:
        plane = bilinear_fit(field)
        plane_grid = apply_plane(plane, h, w, device=device, dtype=dtype)
        f = field - plane_grid
    else:
        plane_grid = None
        f = field

    # ----- Tukey + zero-pad -----
    if taper_frac > 0:
        win = tukey2d(h, w, taper_frac, device=device, dtype=dtype)
        f = f * win
    f_pad = pad_centered(f, pad_to)  # (B, pad_to, pad_to)

    # ----- rfft2 -> kernel -> irfft2 (real-FFT path: ~2x faster) -----
    _, _, k = make_wavenumbers_rfft(pad_to, pad_to, dx, dy, device=device, dtype=dtype)
    F = torch.fft.rfft2(f_pad)
    H = torch.exp(-k * dz)  # eq 12.8 — real-valued, no phase
    out_pad = torch.fft.irfft2(F * H, s=(pad_to, pad_to))
    out = crop_centered(out_pad, h, w)

    # ----- restore plane (a constant offset persists; tilt is invariant
    # under upward continuation, so re-adding the plane is exact). -----
    if plane_grid is not None:
        out = out + plane_grid

    if squeeze:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    """Smallest power of two that is >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Drape forward model + preconditioned CG inversion.
#
# Forward operator A: σ (real, padded) -> ΔT_drape (real, padded)
#
#     A(σ) = w_lo · M_lo(σ) + w_mid · M_mid(σ) + w_hi · M_hi(σ)
#
# where M_z(σ) = IFFT[ 2π·C_m_nT·|k|·exp(-|k|·(z_layer - z)) · FFT(σ) ] is the
# real, self-adjoint flat-to-flat forward kernel and (w_lo, w_mid, w_hi) are
# the Lagrange-quadratic weights for the three evenly-spaced flat altitudes
# z_lo = min(z_drape), z_mid = (z_lo + z_hi)/2, z_hi = max(z_drape) — the
# 3-level "chessboard" interpolation (Cordell 1992 / Blakely §12.1.2).
#
# Adjoint A^T (derived from <A σ, d> = <σ, A^T d>):
#
#     A^T(d) = M_lo(w_lo · d) + M_mid(w_mid · d) + M_hi(w_hi · d)
#
# The flat case (z_drape constant) reduces to A = M_z (self-adjoint, diagonal
# in spectral space) and gives the closed-form spectral-filter Tikhonov
# inversion used in `fit_equivalent_layer`'s flat branch.
# ---------------------------------------------------------------------------


def _build_flat_kernel(k: Tensor, d_gap: float | Tensor) -> Tensor:
    """Forward spectral kernel ``2π · C_m_nT · |k| · exp(-|k| · d_gap)``.

    Maps σ at the equivalent layer to ΔT (nT) at altitude
    ``z = z_layer - d_gap`` (z-down convention). Real-valued ⇒
    self-adjoint, so the same kernel applies in the adjoint direction.
    ``d_gap`` may be a scalar or a tensor that broadcasts against ``k``.
    """
    return 2.0 * math.pi * _CM_NT * k * torch.exp(-k * d_gap)


def _M_apply_kernel(field: Tensor, kernel: Tensor) -> Tensor:
    """Apply a *precomputed* spectral kernel to a real spatial tensor.

    Uses ``rfft2 / irfft2`` — the input is real, so we exploit Hermitian
    symmetry to halve the FFT cost. ``kernel`` is on the rfft layout
    ``(H, W//2 + 1)`` (or batch-broadcast variant).
    """
    *_, h, w = field.shape
    return torch.fft.irfft2(kernel * torch.fft.rfft2(field), s=(h, w))


def _quadratic_lagrange_weights(
    alpha: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lagrange-quadratic weights for 3 evenly-spaced levels.

    With ``alpha = (z - z_lo) / (z_hi - z_lo) ∈ [0, 1]`` and the middle
    sample at ``alpha = 0.5``::

        w_lo  = (1 - α)(1 - 2α)
        w_mid = 4 α (1 - α)
        w_hi  = α (2α - 1)

    Sum to 1 for any α; reduce to identity at the three nodes.
    """
    one_minus = 1.0 - alpha
    w_lo = one_minus * (1.0 - 2.0 * alpha)
    w_mid = 4.0 * alpha * one_minus
    w_hi = alpha * (2.0 * alpha - 1.0)
    return w_lo, w_mid, w_hi


def _drape_forward_kernel(
    sigma_pad: Tensor,
    K_lo: Tensor,
    K_mid: Tensor,
    K_hi: Tensor,
    weights: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """Batched 3-level chessboard forward with precomputed rfft kernels.

    All inputs are batch-compatible — ``sigma_pad``, ``K_*``, and ``weights``
    broadcast over the batch. Uses ``rfft2 / irfft2``; one ``rfft2`` is
    shared across the three altitude kernels (1 rfft + 3 irfft per call).
    """
    *_, h, w = sigma_pad.shape
    F_sigma = torch.fft.rfft2(sigma_pad)
    f_lo = torch.fft.irfft2(K_lo * F_sigma, s=(h, w))
    f_mid = torch.fft.irfft2(K_mid * F_sigma, s=(h, w))
    f_hi = torch.fft.irfft2(K_hi * F_sigma, s=(h, w))
    w_lo, w_mid, w_hi = weights
    return w_lo * f_lo + w_mid * f_mid + w_hi * f_hi


def _drape_adjoint_kernel(
    field_pad: Tensor,
    K_lo: Tensor,
    K_mid: Tensor,
    K_hi: Tensor,
    weights: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """Batched adjoint with precomputed kernels.

    From ``<A σ, d> = <σ, A^T d>`` and the symmetry of the flat kernel::

        A^T d = M_lo(w_lo · d) + M_mid(w_mid · d) + M_hi(w_hi · d)

    Three separate ``rfft2 / irfft2`` calls. A stacked variant was tried
    and rejected — it cuts CG-iter latency at B=1 but the extra
    ``torch.stack`` allocation regressed B≥32 by 10-28 %.
    """
    w_lo, w_mid, w_hi = weights
    return (
        _M_apply_kernel(w_lo * field_pad, K_lo)
        + _M_apply_kernel(w_mid * field_pad, K_mid)
        + _M_apply_kernel(w_hi * field_pad, K_hi)
    )


def _pcg_normal_eq(
    rhs: Tensor,  # (B, H_pad, W_pad) — RHS of (A^T A + λ I) x = rhs
    K_lo: Tensor,  # (B, H_pad, W_pad) or (H_pad, W_pad)
    K_mid: Tensor,
    K_hi: Tensor,
    weights: tuple[Tensor, Tensor, Tensor],
    M_inv_spec: Tensor,
    lam: Tensor,  # (B,) or scalar
    n_iter: int,
) -> Tensor:
    """Batched preconditioned CG solving ``(A^T A + λ I) x = rhs``.

    The raw Krylov iteration on the (self-adjoint, PSD) normal-equation
    operator, taking the RHS directly — used with ``rhs = A^T b`` for the
    fit and with ``rhs = ∂L/∂σ`` for the implicit-function backward (the
    operator is its own adjoint, so the same solve serves both directions).

    ``mask`` is the optional diagonal data-window weight ``W`` (0/1 indicator
    of the un-padded data window on the padded grid). With ``mask=None`` the
    misfit runs over the whole padded grid — which *forces the field to zero
    in the guard ring* and makes σ ring at the window boundary. With the
    window mask, σ outside the data window is constrained only by λ‖σ‖², so
    the fitted layer extends the field smoothly past the edges instead of
    slamming it to zero (measurably lower edge error on continued output).

    Every operation is over the trailing 2 dims (padded grid) and broadcasts
    over the leading batch dim. Inner-product reductions sum *only* the
    spatial dims, leaving ``(B,)`` per-tile residuals — so a single batched
    call replaces the per-tile Python loop the baseline used.

    Stopping is by **fixed iteration count**: empirically the convergence
    on rich-source fields is non-monotone for the first ~7 iters (the
    spectral preconditioner is exact only in the flat limit, so the
    Lagrange-quadratic perturbation introduces transient growth), then
    geometric at ~0.92 / iter. Reaching the documented ~5 % residual
    floor on a 132×132 / 30-prism / 270 m-relief tile takes ~30 iters,
    which is the project's calibrated default.

    Parameters
    ----------
    rhs : Tensor
        ``(B, H, W)`` real right-hand side (already in σ-space).
    K_lo, K_mid, K_hi : Tensor
        Precomputed flat kernels at the three altitudes; broadcast against
        the batch.
    weights : 3-tuple of Tensor
        Lagrange-quadratic weights ``(w_lo, w_mid, w_hi)`` on the padded
        grid; broadcast against the batch.
    M_inv_spec : Tensor
        Spectral preconditioner ``1 / (⟨K²⟩ + λ)`` already evaluated, where
        ``⟨K²⟩ = (K_lo² + K_mid² + K_hi²) / 3`` is the simple average of the
        three flat-kernel squares.
    lam : Tensor
        Per-tile Tikhonov ``λ`` scalar(s); shape ``(B,)`` or 0-D.
    n_iter : int
        Number of CG iterations to run.

    Returns
    -------
    Tensor
        ``(B, H, W)`` real solution on the padded grid.
    """
    # Broadcast helper for per-tile λ: shape (B,) -> (B, 1, 1) for spatial mul.
    if lam.ndim == 1:
        lam_bcast = lam.view(-1, 1, 1)
    else:
        lam_bcast = lam

    x = torch.zeros_like(rhs)
    r = rhs
    z = _M_apply_kernel(r, M_inv_spec)
    p = z

    # Reductions sum spatial dims only; per-tile scalars stay as (B,).
    rz_old = (r * z).sum(dim=(-2, -1))
    eps = torch.finfo(rhs.dtype).tiny

    for _ in range(n_iter):
        Ap = _drape_forward_kernel(p, K_lo, K_mid, K_hi, weights)
        AtAp = _drape_adjoint_kernel(Ap, K_lo, K_mid, K_hi, weights) + lam_bcast * p
        denom = (p * AtAp).sum(dim=(-2, -1)).clamp_min(eps)
        alpha_step = (rz_old / denom).view(-1, 1, 1)
        x = x + alpha_step * p
        r = r - alpha_step * AtAp
        z = _M_apply_kernel(r, M_inv_spec)
        rz_new = (r * z).sum(dim=(-2, -1))
        beta = (rz_new / rz_old.clamp_min(eps)).view(-1, 1, 1)
        p = z + beta * p
        rz_old = rz_new
    return x


def _solve_drape_pcg_batched(
    field_drape_pad: Tensor,  # (B, H_pad, W_pad)
    K_lo: Tensor,
    K_mid: Tensor,
    K_hi: Tensor,
    weights: tuple[Tensor, Tensor, Tensor],
    M_inv_spec: Tensor,
    lam: Tensor,
    n_iter: int,
) -> Tensor:
    """Solve ``(A^T A + λ I) σ = A^T b`` for the padded data tile ``b``.

    Thin wrapper: form the normal-equation RHS ``A^T b`` and run
    :func:`_pcg_normal_eq`. Kept separate so the implicit backward can call
    the raw solver with its own RHS.
    """
    AtB = _drape_adjoint_kernel(field_drape_pad, K_lo, K_mid, K_hi, weights)
    return _pcg_normal_eq(AtB, K_lo, K_mid, K_hi, weights, M_inv_spec, lam, n_iter)


class _ImplicitDrapePCG(torch.autograd.Function):
    """Drape PCG solve with an implicit-function backward.

    The unrolled solve is autograd-correct but stores every CG iterate
    (~``n_iter`` × several padded grids) and replays ~4 FFT applies per
    iteration in backward. This Function instead runs the forward solve
    under ``no_grad`` and differentiates the *solution map* analytically:
    with ``H = A^T A + λI`` (self-adjoint) and ``σ* = H⁻¹ A^T b``,

        ∂L/∂b = A · H⁻¹ · (∂L/∂σ*)

    i.e. backward is ONE more PCG solve with the same operator and
    preconditioner, plus one forward apply — O(1) graph memory and
    backward ≈ forward cost, independent of ``n_iter``.

    Caveat (documented, tested): the forward truncates CG at ``n_iter``,
    so the implicit gradient is that of the *converged* solution and
    differs from the exact unrolled gradient by O(CG residual). At the
    calibrated ``cg_iters`` the disagreement is far below the fp32 noise
    of the iteration itself (see ``tests/test_autograd.py``).

    Gradients are returned for the data tile only; kernels, weights,
    preconditioner and λ are treated as constants (``fit_equivalent_layer``
    falls back to the unrolled solver when any of them requires grad —
    e.g. a ``z_obs`` that itself needs gradients).
    """

    @staticmethod
    def forward(
        ctx,
        field_drape_pad: Tensor,
        K_lo: Tensor,
        K_mid: Tensor,
        K_hi: Tensor,
        w_lo: Tensor,
        w_mid: Tensor,
        w_hi: Tensor,
        M_inv_spec: Tensor,
        lam: Tensor,
        n_iter: int,
    ) -> Tensor:
        weights = (w_lo, w_mid, w_hi)
        with torch.no_grad():
            sigma = _solve_drape_pcg_batched(
                field_drape_pad, K_lo, K_mid, K_hi, weights, M_inv_spec, lam, n_iter
            )
        ctx.save_for_backward(K_lo, K_mid, K_hi, w_lo, w_mid, w_hi, M_inv_spec, lam)
        ctx.n_iter = n_iter
        return sigma

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_sigma: Tensor):  # noqa: D102 — see class docstring
        K_lo, K_mid, K_hi, w_lo, w_mid, w_hi, M_inv_spec, lam = ctx.saved_tensors
        weights = (w_lo, w_mid, w_hi)
        with torch.no_grad():
            y = _pcg_normal_eq(
                grad_sigma.contiguous(), K_lo, K_mid, K_hi, weights, M_inv_spec, lam, ctx.n_iter
            )
            grad_field = _drape_forward_kernel(y, K_lo, K_mid, K_hi, weights)
        return grad_field, None, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Equivalent-layer inversion (Blakely §12.4 + §12.1.2 specialised to RTP)
# ---------------------------------------------------------------------------

# Numerical convention: we keep the field in **nT** end-to-end so the kernel
# magnitudes stay in float64's sweet spot. The Blakely formula uses SI units
# (Tesla, A/m), but μ_0/(4π) = 1e-7 introduces seven orders of magnitude that
# drag every product down to ~1e-20 and break iterative solvers (CG silently
# stagnates because inner products fall below float64 precision). Scaling the
# kernel by 1e9 (T→nT on the field side) brings σ into a numerically friendly
# range without changing the physics.
#
# Effective constant in nT:  C_m_nT = (μ_0 / (4π)) · 1e9 = 1e2  (units: nT·m/A)
_CM_NT = 100.0


@dataclass
class EquivalentLayer:
    """Cached spectrum of a fitted single-layer equivalent magnetisation.

    The "thin sheet" magnetic-moment-density ``sigma(x, y)`` (units A·m
    per pixel, i.e. column moment per unit area integrated over the
    vanishing-thickness layer) is stored in Fourier space on the padded
    grid so that multiple :func:`upward_continue` calls reuse one FFT.

    Attributes
    ----------
    sigma_hat : Tensor
        ``(B?, H_pad, W_pad // 2 + 1)`` complex tensor — ``rfft2[sigma]`` on
        the padded grid (real-FFT half-spectrum exploiting Hermitian
        symmetry). The DC bin is forced to zero (regional plane is detrended
        out and re-added separately).
    k : Tensor
        ``(H_pad, W_pad // 2 + 1)`` real tensor — ``|k|`` in rad/m on the
        rfft2 layout. DC at ``[0, 0]``.
    z_layer : float
        Equivalent-layer depth, +down convention (m).
    z_ref : float
        Reference observation altitude used at fit time, +down (m). For
        flat fits this equals the (single) observation altitude; for
        draped fits it equals ``min(z_obs)`` — the highest sensor altitude
        in z-down — and is used for diagnostics (the actual chessboard
        endpoints are stored implicitly inside ``sigma_hat``).
    pad : tuple[int, int]
        ``(orig_h, orig_w)`` so the cached spectrum can be cropped back to
        the original tile shape after :func:`upward_continue`.
    plane : tuple[Tensor, Tensor, Tensor]
        Bilinear plane ``(a, b, c)`` removed at fit time and re-added at
        forward time. Stored in **nT** (matches the rest of the pipeline).
    dx, dy : float
        Pixel size in metres.
    """

    sigma_hat: Tensor
    k: Tensor
    z_layer: float
    z_ref: float
    pad: tuple[int, int]
    plane: tuple[Tensor, Tensor, Tensor]
    dx: float
    dy: float

    @property
    def device(self) -> torch.device:
        """Device the cached σ spectrum lives on."""
        return self.sigma_hat.device

    @property
    def dtype(self) -> torch.dtype:
        """Real dtype of the original tile (complex<float32> ⇒ float32)."""
        return torch.empty(0, dtype=self.sigma_hat.dtype).real.dtype


def fit_equivalent_layer(
    rtp_tile: Tensor,
    dx: float,
    dy: float,
    z_obs: float | Tensor,
    z_layer: float,
    *,
    eps: float = 1e-3,
    pad_to: int | None = None,
    taper_frac: float = 0.0,
    detrend: bool = True,
    cg_iters: int = 10,
    implicit_grad: bool = True,
    edge_ext: int | str | None = "auto",
) -> EquivalentLayer:
    """Invert an RTP magnetic tile into a single-layer equivalent source.

    Implements the flat-to-flat case of Blakely's equivalent-source method
    (§12.1.2 + §12.3 specialised to RTP, ``Theta_m = Theta_f = 1``). The
    forward kernel for a thin layer at depth ``z_layer`` observed from
    altitude ``z_obs`` (both +down, ``z_layer > z_obs``) is

        F[ΔT] = 2π · C_m · |k| · exp(-|k| · (z_layer - z_obs)) · F[σ]

    The inverse multiplies by ``exp(+|k|·d)`` (downward-continuation flavour,
    unstable at high |k|), so we damp with a Tikhonov / Wiener filter

        W(|k|) = |k|² / (|k|² + (ε · k_max)²)

    that simultaneously cancels the ``1/|k|`` singularity at low |k| and
    suppresses the ``exp(+|k|·d)`` blow-up at high |k|. ``ε`` is the
    dimensionless single knob (Tikhonov damping, Blakely Ch. 12);
    default ``1e-3``.

    Parameters
    ----------
    rtp_tile : Tensor
        ``(H, W)`` or ``(B, H, W)`` reduced-to-pole anomaly in **nT**.
    dx, dy : float
        Pixel size in metres.
    z_obs : float or Tensor
        Observation altitude(s), +down convention. Above ground is negative.

        - **Scalar** → flat survey: closed-form spectral-filter Tikhonov
          inversion (one rfft2 / multiply / irfft2 per fit).
        - **Tensor** matching the spatial shape of ``rtp_tile`` (or
          ``(B, H, W)``) → draped survey: preconditioned conjugate gradient
          on a 3-level Lagrange-quadratic chessboard forward, with each
          tile's chessboard endpoints set to its own
          ``(min(z_obs), max(z_obs))``.
    z_layer : float
        Equivalent-layer depth (m, +down). Must be > ``max(z_obs)``.
    eps : float
        Tikhonov regularisation parameter (dimensionless). Default 1e-3.
    pad_to : int, optional
        FFT grid size. Default = next power of two ≥ ``1.5 · max(H, W)``.
    taper_frac : float
        Tukey-window fraction applied before zero-padding (``0`` disables).
        Default **0** (no taper). Unlike the bare :func:`upward_continue_field`
        filter — which multiplies un-inverted data in the spectrum and needs a
        taper to suppress leakage — this is an *inversion*: the taper would
        attenuate the data toward zero in the boundary ring, so the fitted
        equivalent source reproduces attenuated edges and the forward fails
        self-consistency there (the residual shows a ~``taper_frac``-wide frame
        of large error). The zero-pad guard band plus Tikhonov regularisation
        already control wraparound, so the data is fit untapered by default.
    detrend : bool
        Subtract a least-squares bilinear plane before fitting and re-add
        it on every forward call. Default ``True``.
    cg_iters : int
        Number of preconditioned-CG iterations for the drape branch
        (ignored for scalar ``z_obs``). Default 10 — the simple-average
        spectral preconditioner ``M⁻¹ = 1 / (⟨K²⟩ + λ)`` (where
        ``⟨K²⟩ = (K_lo² + K_mid² + K_hi²) / 3``) gives PCG geometric
        convergence at ~10× per iter, so 10 iters reaches the
        regularisation-limited residual floor on rich-source 3D-model
        data (~0.75 % drape-to-drape rel err on the validation tile).
        CG runs to a fixed count (no early-exit sync) to keep autograd
        graphs deterministic and avoid GPU stalls.
    implicit_grad : bool
        Backward strategy for the drape CG solve (ignored for scalar
        ``z_obs``). ``True`` (default) differentiates the solution map via
        the implicit-function theorem — backward is one extra PCG solve
        instead of unrolling ``cg_iters`` iterations, giving O(1) graph
        memory and iteration-count-independent backward cost (see
        :class:`_ImplicitDrapePCG`). Falls back to the unrolled solver
        automatically when the survey geometry itself carries gradients
        (``z_obs.requires_grad``), since the implicit path only returns
        gradients w.r.t. the data tile. Set ``False`` to force the
        unrolled (exact-to-the-iteration) backward.
    edge_ext : int, "auto", or None
        Feathered edge extension (both branches). Before padding, the
        detrended tile is replicate-extended by this many pixels per side
        and the ring cosine²-feathered to zero (see
        :func:`~magsr.fourier._fft_utils.extend_feathered`). Rationale:
        with a bare zero pad the misfit treats the guard ring as real
        zero-valued data, so the fitted σ must explain an abrupt data→0
        step at the tile boundary and *rings* there on every continued
        output; the feathered ring gives the fit a plausible smooth
        continuation instead, cutting edge-band error severalfold with
        the tile centre unchanged. ``"auto"`` (default) uses
        ``max(8, min(H, W) // 10)``; ``None``/``0`` disables. (A hard
        data-window mask on the misfit is not a substitute: it makes the
        converged inverse map 1/λ-sensitive off the window, which breaks
        the implicit-function backward.) The extension is differentiable
        (ring gradients fold back onto the edge pixels).

    Notes
    -----
    For draped surveys we solve

        minimize_σ  ‖ A(σ) - data ‖²  +  λ ‖ σ ‖²

    by preconditioned CG on the normal equations, where ``A`` is the
    3-level Lagrange-quadratic chessboard forward operator (interpolating
    between flat continuations at ``min(z_obs)``, the midpoint, and
    ``max(z_obs)``). The Tikhonov parameter ``λ = (eps · K_max)²`` is
    calibrated to the dominant kernel magnitude at the equivalent-layer
    depth, so ``eps`` carries the same meaning whether the survey is flat
    or draped.

    The user-facing forward operator :func:`upward_continue` uses an
    independent, configurable **N-layer linear** chessboard, optimised for
    drape-to-drape evaluation accuracy. The fit's 3-level Lagrange variant
    is purely an internal regularisation choice — it gives the PCG
    preconditioner an exact diagonal in the flat limit.
    """
    z_obs_is_drape = torch.is_tensor(z_obs)
    if z_obs_is_drape:
        # one-shot validation sync (feeds the error below); not in the FFT/autograd hot path
        z_ref = float(torch.min(z_obs).item())
    else:
        z_ref = float(z_obs)
    if z_layer <= z_ref:
        raise ValueError(
            f"z_layer ({z_layer}) must be greater than max-altitude observer "
            f"z_ref ({z_ref}) (layer must be below all observers, z-down)"
        )

    if rtp_tile.ndim == 2:
        rtp_tile = rtp_tile.unsqueeze(0)
        squeeze_b = True
    else:
        squeeze_b = False
    *batch, h, w = rtp_tile.shape
    device, dtype = rtp_tile.device, rtp_tile.dtype

    # Feathered edge extension radius (see the `edge_ext` doc).
    if edge_ext == "auto":
        r_ext = max(8, min(h, w) // 10)
    else:
        r_ext = int(edge_ext) if edge_ext else 0

    if pad_to is None:
        pad_to = _next_pow2(int(1.5 * max(h, w)))
    if pad_to < max(h, w) + 2 * r_ext:
        raise ValueError(f"pad_to={pad_to} smaller than grid {h}x{w} plus 2x edge_ext={r_ext}")

    # Operate in **nT** end-to-end so the SI prefactor 2π·μ₀/(4π)·1e9 = 6.28e-1
    # keeps all kernel products O(1) (instead of O(1e-19) when working in T).
    field_nT = rtp_tile

    # Detrend (planes are harmonic and pass through continuation unchanged,
    # so we keep them out of the FFT path entirely).
    if detrend:
        plane = bilinear_fit(field_nT)
        plane_grid = apply_plane(plane, h, w, device=device, dtype=dtype)
        field_nT = field_nT - plane_grid
    else:
        plane = (
            torch.zeros(*batch, device=device, dtype=dtype),
            torch.zeros(*batch, device=device, dtype=dtype),
            torch.zeros(*batch, device=device, dtype=dtype),
        )

    # Tukey + feathered extension + zero-pad. The zero pad suppresses
    # high-|k| wraparound; the feathered replicate ring (edge_ext) keeps the
    # misfit from treating the abrupt data→0 step as real data (σ would ring
    # at the boundary otherwise). The original window stays centred, so the
    # post-forward crop recovers it exactly.
    if taper_frac > 0:
        win = tukey2d(h, w, taper_frac, device=device, dtype=dtype)
        field_nT = field_nT * win
    if r_ext > 0:
        field_nT = extend_feathered(field_nT, r_ext)
    field_pad = pad_centered(field_nT, pad_to)

    # Wavenumbers on the rfft2 half-spectrum layout: (pad_to, pad_to//2 + 1).
    # All real-valued FFTs use rfft2/irfft2 internally for ~2x speed.
    _, _, k = make_wavenumbers_rfft(pad_to, pad_to, dx, dy, device=device, dtype=dtype)
    k_max = math.pi / min(dx, dy)

    if z_obs_is_drape:
        if z_obs.shape[-2:] != (h, w):
            raise ValueError(f"z_obs spatial shape {tuple(z_obs.shape[-2:])} must match rtp_tile {(h, w)}")
        # Drape PCG always runs in float32 (validated equal to fp64, ~2× faster); the
        # EquivalentLayer spectrum is stored back at the caller's dtype.
        cg_dtype = torch.float32
        field_pad_cg = field_pad.to(cg_dtype) if dtype != cg_dtype else field_pad
        z_obs_cg = z_obs.to(device=device, dtype=cg_dtype)
        if z_obs_cg.ndim == 2:
            z_obs_cg = z_obs_cg.unsqueeze(0)
        if r_ext > 0:
            # Extend the observation surface to match the extended data ring
            # (replicate: no new extremes, so kernels and λ are unchanged).
            z_obs_cg = torch.nn.functional.pad(
                z_obs_cg.unsqueeze(1), (r_ext,) * 4, mode="replicate"
            ).squeeze(1)
        # Wavenumbers at the CG dtype — re-fetch (cached, so essentially free).
        if dtype != cg_dtype:
            _, _, k_cg = make_wavenumbers_rfft(pad_to, pad_to, dx, dy, device=device, dtype=cg_dtype)
        else:
            k_cg = k
        # Per-tile altitude extrema as (B,) tensors. AMIN/AMAX over the
        # spatial dims keep autograd happy and avoid any .item() sync.
        z_lo_t = z_obs_cg.amin(dim=(-2, -1))  # (B,)
        z_hi_t = z_obs_cg.amax(dim=(-2, -1))
        z_mid_t = 0.5 * (z_lo_t + z_hi_t)
        # Linear blend coefficient α = (z - z_lo) / (z_hi - z_lo), per tile.
        # Avoid divide-by-zero on flat-z tiles via clamp_min on the span.
        span = (z_hi_t - z_lo_t).clamp_min(torch.finfo(cg_dtype).eps)
        alpha = (z_obs_cg - z_lo_t.view(-1, 1, 1)) / span.view(-1, 1, 1)
        flat_mask = (z_hi_t == z_lo_t).view(-1, 1, 1)
        alpha = torch.where(flat_mask, torch.zeros_like(alpha), alpha)
        alpha_pad = pad_centered(alpha, pad_to, mode="replicate")  # (B, P, P)
        weights = _quadratic_lagrange_weights(alpha_pad)

        # Three flat kernels per tile, broadcast-ready: (B, 1, 1) * (P, P) -> (B, P, P).
        d_lo_t = (float(z_layer) - z_lo_t).view(-1, 1, 1)
        d_mid_t = (float(z_layer) - z_mid_t).view(-1, 1, 1)
        d_hi_t = (float(z_layer) - z_hi_t).view(-1, 1, 1)
        K_lo = _build_flat_kernel(k_cg, d_lo_t)
        K_mid = _build_flat_kernel(k_cg, d_mid_t)
        K_hi = _build_flat_kernel(k_cg, d_hi_t)

        # Tikhonov: λ = (eps · K_max(b))² with per-tile K_max from minimum
        # layer-to-sensor distance.  d_min = z_layer - z_hi (smallest gap).
        d_min_t = (float(z_layer) - z_hi_t).clamp_min(1.0)
        K_max_t = 2.0 * math.pi * _CM_NT / d_min_t * math.exp(-1.0)
        lam_t = (eps * K_max_t) ** 2  # (B,)

        # Spectral preconditioner M^-1 = 1 / (⟨K²⟩ + λ), ⟨K²⟩ = mean of the three flat-kernel
        # squares — a diagonal approximation of AᵀA across the chessboard altitude range.
        K2_avg = (K_lo * K_lo + K_mid * K_mid + K_hi * K_hi) / 3.0
        M_inv_spec = 1.0 / (K2_avg + lam_t.view(-1, 1, 1))

        # Implicit-function backward: O(1) graph memory (backward = one more
        # PCG solve). Only valid while the operator itself is constant w.r.t.
        # the autograd graph — kernels/weights/λ all derive from z_obs, so a
        # differentiable z_obs forces the unrolled path.
        use_implicit = implicit_grad and not any(
            t.requires_grad for t in (K_lo, K_mid, K_hi, *weights, M_inv_spec, lam_t)
        )
        if use_implicit:
            sigma_pad = _ImplicitDrapePCG.apply(
                field_pad_cg, K_lo, K_mid, K_hi, *weights, M_inv_spec, lam_t, cg_iters
            )
        else:
            sigma_pad = _solve_drape_pcg_batched(
                field_pad_cg,
                K_lo=K_lo,
                K_mid=K_mid,
                K_hi=K_hi,
                weights=weights,
                M_inv_spec=M_inv_spec,
                lam=lam_t,
                n_iter=cg_iters,
            )
        # Cast σ back to the caller's dtype before caching in the layer.
        if sigma_pad.dtype != dtype:
            sigma_pad = sigma_pad.to(dtype)
        sigma_hat = torch.fft.rfft2(sigma_pad)
    else:
        # Closed-form spectral-filter regularisation for the flat case. The
        # filter W = |k|² / (|k|² + (ε · k_max)²) shapes the transfer function
        # without depending on the layer depth, which preserves layer-depth
        # invariance of the forward continuation. The drape branch above uses
        # Tikhonov-on-σ via PCG (a slightly different regulariser); the two
        # agree to a few percent in the flat limit.
        d = float(z_layer - z_ref)
        F = torch.fft.rfft2(field_pad)
        W = k * k / (k * k + (eps * k_max) ** 2)
        safe_k = torch.where(k > 0, k, torch.ones_like(k))
        G_inv = torch.where(
            k > 0,
            torch.exp(k * d) / (2.0 * math.pi * _CM_NT * safe_k) * W,
            torch.zeros_like(k),
        )
        sigma_hat = F * G_inv

    return EquivalentLayer(
        sigma_hat=sigma_hat.squeeze(0) if squeeze_b else sigma_hat,
        k=k,
        z_layer=float(z_layer),
        z_ref=z_ref,
        pad=(h, w),
        plane=plane if not squeeze_b else tuple(p.squeeze(0) for p in plane),
        dx=float(dx),
        dy=float(dy),
    )


def upward_continue(
    layer: EquivalentLayer,
    z_target: float | Tensor,
    *,
    n_layers: int = 8,
    z_min: float | None = None,
    z_max: float | None = None,
) -> Tensor:
    """Forward the fitted equivalent layer to a target altitude.

    For a flat (scalar) target, implements

        F[ΔT(z_target)] = 2π · C_m · |k| · exp(-|k| · (z_layer - z_target)) · F[σ]

    For a draped (Tensor) target ``z_t(x, y)`` — which is the typical
    drape-to-drape workflow, e.g. fit on data sampled at ``DEM + 60 m``,
    forward to ``DEM + 100 m`` — uses an **N-layer linear chessboard**
    (Blakely §12.1.2 / Cordell 1992):

    1. ``n_layers`` flat upward continuations evenly spaced between
       ``z_min`` and ``z_max`` (defaults to ``z_target.min/max()``).
       The slice range must span the full topographic relief of the
       target — that's what sets the chessboard error, *not* the
       drape-clearance difference between source and target.
    2. Per-pixel piecewise-linear interpolation using ``z_target(x, y)``
       to blend the two bracketing slices.

    Linear interpolation error scales as
    ``((z_max - z_min) / (n_layers - 1))^2 · |k|^2``. Default
    ``n_layers=8`` matches harmonica's spatial-domain `predict` to ≤ 1 %
    on typical KSA-style geometry. Drop to 4 for cheaper / less accurate;
    raise to 16 for paranoid accuracy.

    Inverse-FFT, crop, and add the bilinear plane back. Output is **nT**.

    Parameters
    ----------
    layer : EquivalentLayer
        Fitted equivalent layer (cached σ spectrum).
    z_target : float or Tensor
        Scalar (flat target) or ``(H, W)`` / ``(B, H, W)`` real tensor of
        per-pixel target altitudes (z-down convention).
    n_layers : int
        Number of flat slices for the chessboard. Ignored if ``z_target``
        is a scalar. Default 8.
    z_min, z_max : float, optional
        Override the chessboard span. Default to ``z_target.min/max()``.
        Provide explicit bounds when batching tiles with very different
        DEM ranges to keep error bounds predictable.

    All target altitudes must be above the equivalent layer
    (``< z_layer`` in z-down).
    """
    if torch.is_tensor(z_target):
        if z_target.shape[-2:] != layer.pad:
            raise ValueError(
                f"z_target spatial shape {tuple(z_target.shape[-2:])} must match "
                f"original tile shape {layer.pad}"
            )
        max_target = float(z_target.max().item())
    else:
        max_target = float(z_target)
    if max_target >= layer.z_layer:
        raise ValueError(
            f"z_target ({max_target}) must be < z_layer ({layer.z_layer}) "
            "(target must be above the equivalent layer)"
        )

    sigma_hat = layer.sigma_hat
    if sigma_hat.ndim == 2:
        sigma_hat = sigma_hat.unsqueeze(0)
        squeeze_b = True
    else:
        squeeze_b = False

    # k is on rfft layout (pad_to, pad_to//2 + 1). The spatial padded size is
    # layer.k.shape[0] (rfft preserves the height).
    pad_to = layer.k.shape[0]

    real_dtype = torch.empty(0, dtype=sigma_hat.dtype).real.dtype
    h_orig, w_orig = layer.pad

    if torch.is_tensor(z_target):
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2, got {n_layers}")
        # Chessboard span = full topographic extent of the target surface;
        # this is what sets the linear-interp error, *not* the source/target
        # drape gap.
        z_lo = float(z_min if z_min is not None else z_target.min().item())
        z_hi = float(z_max if z_max is not None else z_target.max().item())
        span = z_hi - z_lo
        eps_z = float(torch.finfo(real_dtype).eps) * max(abs(z_lo), abs(z_hi), 1.0)

        if span < eps_z:
            # Genuinely flat target — single forward, no chessboard needed.
            out = _flat_forward(sigma_hat, layer, z_lo)
        else:
            out = _drape_forward_nlayer(sigma_hat, layer, z_target, z_lo, z_hi, n_layers)
    else:
        out = _flat_forward(sigma_hat, layer, float(z_target))

    plane_grid = apply_plane(layer.plane, h_orig, w_orig, device=out.device, dtype=out.dtype)
    out = out + plane_grid  # plane is in nT, like the rest of the pipeline
    return out.squeeze(0) if squeeze_b else out


def _flat_forward(sigma_hat: Tensor, layer: EquivalentLayer, z_target: float) -> Tensor:
    """Single flat forward continuation from cached σ to a scalar altitude.

    ``sigma_hat`` is expected with a leading batch dim already (broadcast or
    real). Returns the cropped, plane-free ΔT in nT.
    """
    pad_to = layer.k.shape[0]
    G_fwd = _build_flat_kernel(layer.k, float(layer.z_layer - z_target))
    out_pad = torch.fft.irfft2(sigma_hat * G_fwd, s=(pad_to, pad_to))
    return crop_centered(out_pad, layer.pad[0], layer.pad[1])


def _drape_forward_nlayer(
    sigma_hat: Tensor,
    layer: EquivalentLayer,
    z_target: Tensor,
    z_lo: float,
    z_hi: float,
    n_layers: int,
) -> Tensor:
    """N-layer linear-interp chessboard forward (drape-to-drape).

    Builds ``n_layers`` flat continuations evenly spaced between ``z_lo``
    and ``z_hi`` in a single batched ``irfft2``, then linearly interpolates
    per pixel using ``z_target``. Returns ΔT (nT) on the **original** tile
    grid, plane-free; the caller adds the bilinear plane back.
    """
    real_dtype = torch.empty(0, dtype=sigma_hat.dtype).real.dtype
    pad_to = layer.k.shape[0]
    h_orig, w_orig = layer.pad

    # 1. Slice z's evenly spaced over [z_lo, z_hi]. shape: (n_layers,).
    zs = torch.linspace(z_lo, z_hi, n_layers, device=layer.k.device, dtype=real_dtype)
    d_gaps = float(layer.z_layer) - zs  # (n_layers,)

    # 2. Stacked rfft kernels; broadcast against (B, H, W//2+1) σ_hat.
    K_stack = _build_flat_kernel(
        layer.k.unsqueeze(0), d_gaps.view(-1, 1, 1)
    )  # (n_layers, H_pad, W_pad//2+1)

    # 3. ONE batched irfft2 over the (B, n_layers) axes. The unsqueeze on
    # σ_hat creates the n_layers axis via broadcasting — only the spectrum
    # stack itself materialises in memory.
    F_stack = sigma_hat.unsqueeze(-3) * K_stack  # (B, n_layers, H, W//2+1)
    f_stack_pad = torch.fft.irfft2(F_stack, s=(pad_to, pad_to))

    # 4. Crop slices to original tile shape — (B, n_layers, H, W).
    f_stack = crop_centered(f_stack_pad, h_orig, w_orig)

    # 5. Per-pixel linear interp via gather along the n_layers dim.
    z_t = z_target.to(device=layer.k.device, dtype=real_dtype)
    if z_t.ndim == 2:
        z_t = z_t.unsqueeze(0)  # (1, H, W) — broadcasts with (B, …) f_stack
    idx_f = (z_t - z_lo) / (z_hi - z_lo) * (n_layers - 1)  # (B, H, W) ∈ [0, N-1]
    idx_lo = idx_f.floor().long().clamp(0, n_layers - 2)
    frac = idx_f - idx_lo.to(real_dtype)
    f_lo = torch.gather(f_stack, dim=-3, index=idx_lo.unsqueeze(-3)).squeeze(-3)
    f_hi = torch.gather(f_stack, dim=-3, index=(idx_lo + 1).unsqueeze(-3)).squeeze(-3)
    return (1.0 - frac) * f_lo + frac * f_hi

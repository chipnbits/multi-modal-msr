# `magsr.fourier`

Fourier-domain potential-field tools for reduced-to-pole aeromagnetic
data: equivalent-layer inversion + upward continuation
(Blakely 1996, *Potential Theory in Gravity and Magnetic Applications*,
Ch. 12), the first-vertical-derivative operator and its training loss
(`fvd.py`), and operator-consistent synthetic HR/LR survey-pair generation
(`synth.py`). Everything shares one FFT toolkit (`_fft_utils.py`); the
modules differ only in the spectral kernel applied.

The whole pipeline is **autograd-clean** — gradients flow through fit and
forward, so the operators sit naturally inside a learning loop and back-prop
to whatever produced the input tile (e.g. an SR network's output). The drape
CG solve uses an **implicit-function backward** by default (one extra PCG
solve instead of unrolling the iterations): O(1) graph memory — 7.7× lower
peak at B=32 — and backward cost independent of `cg_iters`. See
`implicit_grad` in `fit_equivalent_layer` (auto-falls back to the unrolled
graph when `z_obs` itself requires grad).

## Quick start

```python
import torch
from magsr.fourier import fit_equivalent_layer, upward_continue

# rtp_60m: (B, H, W) RTP anomaly in nT, sampled at flight altitude DEM + 60 m.
# dem:     (B, H, W) terrain elevation in metres (+up).
clearance = 60.0

z_obs   = -clearance - dem                           # z-down observation surface
z_layer = float(z_obs.max() + 300.0)                 # layer 300 m below the
                                                     # deepest sensor

layer = fit_equivalent_layer(
    rtp_60m, dx=60.0, dy=60.0, z_obs=z_obs, z_layer=z_layer, eps=1e-3,
)
rtp_100m = upward_continue(
    layer, z_target=-100.0 - dem, n_layers=8,
)
```

## Public API

| Symbol | Purpose |
|---|---|
| `fit_equivalent_layer(rtp_tile, dx, dy, z_obs, z_layer, …)` | Invert an RTP tile to a single-layer magnetic-moment density σ. |
| `upward_continue(layer, z_target, n_layers=8, …)` | Forward-continue a fitted layer to any (flat or draped) target altitude. |
| `upward_continue_field(field, dx, dy, dz, …)` | Standalone level-to-level continuation — pad → FFT → `exp(-|k|·dz)` → IFFT. No fit. |
| `EquivalentLayer` | Cached σ spectrum + grid metadata; device/dtype handles. |
| `first_vertical_derivative(field, dx, dy, …)` | 1VD of a potential-field grid (`|k|` kernel). |
| `FVDLoss(dx, dy, taper_frac)` | L1 between the 1VD of prediction and target — high-pass training penalty. |
| `synth_uc_pair(hr_full, dz, …)` | Synthetic HR/LR pair via level-to-level UC + decimation + noise. |
| `synth_uc_pair_drape(hr_full, dem_hr, dz, …)` | Drape-aware synthetic pair via EL fit on the survey drape. |

## Conventions

- **Coordinates**: metres, *z-down* (positive z = into the ground). Above-ground
  altitudes are negative.
- **Grid orientation**: `(H, W)` with H along north and W along east; pixel
  size `dx` (east) and `dy` (north). Padding is centred and symmetric.
- **Field units**: nT, end-to-end. The kernel prefactor
  `2π · μ₀/(4π) · 1e9 ≈ 0.628` (encoded in `_CM_NT`) keeps numerics at O(1).
- **Magnetisation σ**: column moment per unit area (A·m / m² = A), stored as
  `rfft2[σ]` on the padded grid.
- **Bilinear plane**: detrended out before the FFT path and re-added on every
  forward call — invariant under upward continuation, so this is exact.

## How it works

### Flat-survey inversion (closed-form)

For RTP geometry the forward kernel is
`F[ΔT] = 2π · C_m · |k| · exp(-|k| · d) · F[σ]` (Blakely eq 12.26 with
Θ_m = Θ_f = 1). Inversion is a Tikhonov spectral filter

```
F[σ] = F[ΔT] · exp(+|k|·d) · |k| / (2π · C_m · (|k|² + (ε · k_max)²))
```

evaluated in O(1) FFTs. ε ≈ 1e-3 keeps high-|k| content stable.

### Draped-survey inversion (preconditioned CG)

For varying-altitude `z_obs` we build a 3-level Lagrange-quadratic
chessboard forward operator A (interpolation between flat continuations
at `min(z_obs)`, the midpoint, and `max(z_obs)`). PCG solves the normal
equations `(AᵀA + λI) σ = Aᵀ b` in **5-15 iterations** thanks to a
spectral preconditioner that is exact in the flat limit.

The Cordell-Grauch Picard iteration (Blakely eq 12.14) was tried and
rejected — it diverges for `|k|·max_dz > 0.7`, which is most of the
spectrum on a typical 60 m / 132×132 tile.

### Forward to a draped target (N-layer linear chessboard)

`upward_continue(layer, z_target=Tensor, n_layers=N)` builds **N** flat
continuations evenly spaced over `[z_target.min(), z_target.max()]` in a
single batched `irfft2`, then linearly interpolates per pixel using the
actual `z_target(x, y)`. Linear interpolation has second-order convergence
in z, so error scales as `((z_max - z_min) / (n_layers - 1))² · |k|²`.

The default `n_layers=8` matches the canonical spatial-domain reference
(`harmonica.EquivalentSources.predict`) to ≤ 1 % on typical KSA-style
geometry. Drop to 4 for cheaper / less accurate; raise to 16 for paranoid
accuracy. Note the chessboard span must cover the **full topographic
relief** of the target surface, not the source-target altitude gap.

## Performance

CUDA fp32 / 132×132 tile / pad 256, batch B = 32:

| operation               | per tile |
|-------------------------|----------|
| `fit_equivalent_layer`  | ~3 ms (drape PCG, 15 iters)  |
| `upward_continue` (n=8) | ~0.13 ms (drape) / ~0.02 ms (flat) |

## Validation

The in-repo suite covers:

- **Synthetic dipole round-trips** vs analytic ground truth
  (`tests/test_equivalent_layer.py`), with the analytic references in
  `_reference.py` themselves gated by `tests/test_reference.py`.
- **Brute-force spatial summation** (no FFT) for the upward-continuation
  primitive (`tests/test_upward_continuation.py`).
- **Differentiability** end-to-end (`tests/test_autograd.py`) including a
  finite-difference vs autograd check.

Beyond the in-repo suite, the operator was validated during development
against `harmonica.EquivalentSources` (< 1 % agreement on flat and draped
geometry), an N-layer chessboard convergence study, a blind 3D-prism-model
drape-to-drape recovery, float32/float64 parity, and real-KSA-tile
integration.

## Limitations

- **RTP-only**. Non-RTP magnetic data (with horizontal-magnetization
  components) violates the Θ_m = Θ_f = 1 specialisation; results degrade
  proportionally to the horizontal magnetic moment.
- **Single equivalent layer**. The fit assumes one buried layer; you cannot
  recover depth-dependent source structure from above-ground data, only the
  smoothest layer that explains it (this is intrinsic to equivalent-source
  inversion, not a code limitation).
- **Edges**. The fit replicate-extends the tile with a cosine²-feathered
  ring (`edge_ext="auto"`, ~min(H,W)/10 px per side) so the misfit never
  treats the zero guard ring as real zero-valued data — on the 3D-model
  validation pair this cut the drape-B edge-band error from RMS 1.75 % /
  max 13.3 % of peak to RMS 0.48 % / max 2.1 %. A thin boundary frame is
  still the least trustworthy part of the tile; crop a few px for strict
  quantitative comparisons (`edge_ext=None` restores the old zero-ring
  behaviour).

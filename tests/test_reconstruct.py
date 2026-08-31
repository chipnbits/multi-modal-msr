"""Tests for the reconstruction pipeline + dataset normalization helpers.

These pin the behavior of the reconstruct-function unification (shared `_sweep_and_blend`
core), the `read_masked` nodata handling, the mag-input NaN-fill correctness fix, and the
KSA `_clip_scale01` / `_center_scale01` helpers.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from magsr.models import BicubicModel
from magsr.normalize import center_scale01, clip_scale01, denorm01, from_pm1, to_pm1
from magsr.reconstruct import (
    Normalizer,
    plan_lr_patches,
    reconstruct_region,
    reconstruct_region_multichannel,
)
from magsr.reconstruct.build import MAG_INPUT_NAN_FILL, _read_masked, infer_patch


def _write_raster(path, arr, *, res=180.0, x0=500_000.0, y0=4_000_000.0, nodata=None):
    """Write a single-band float32 GeoTIFF on a UTM-like grid, return its world bounds."""
    h, w = arr.shape
    transform = from_origin(x0, y0, res, res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:32637",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype("float32"), 1)
    return (x0, y0 - h * res, x0 + w * res, y0)  # (minx, miny, maxx, maxy)


# --------------------------------------------------------------------------- #
# read_masked (A2)
# --------------------------------------------------------------------------- #
def test_read_masked_maps_nodata_to_nan(tmp_path):
    arr = np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=np.float32)
    p = tmp_path / "r.tif"
    _write_raster(p, arr, nodata=-9999.0)
    with rasterio.open(p) as ds:
        out = _read_masked(ds, window=rasterio.windows.Window(0, 0, 2, 2))
    assert np.isnan(out[0, 1])
    assert out[0, 0] == 1.0 and out[1, 1] == 3.0


# --------------------------------------------------------------------------- #
# reconstruct unification (A1)
# --------------------------------------------------------------------------- #
def test_single_channel_matches_multichannel_none(tmp_path):
    """reconstruct_region == reconstruct_region_multichannel(aux_channels=None)."""
    rng = np.random.default_rng(0)
    lr = rng.random((20, 24)).astype(np.float32)
    p = tmp_path / "lr.tif"
    minx, miny, maxx, maxy = _write_raster(p, lr)

    plan = plan_lr_patches(
        polygon_world=(minx, miny, maxx, maxy),
        lr_path=str(p),
        patch_lr_px=8,
        stride_lr_px=4,
        buffer_lr_px=0,
        scale=3,
        min_valid_frac=0.0,
    )
    model = BicubicModel(upscale_factor=3)
    norm = Normalizer(vmin=0.0, vmax=1.0)

    a = reconstruct_region(model, str(p), plan, normalizer=norm, blend_kind="cosine", batch_size=4)
    b = reconstruct_region_multichannel(
        model, str(p), plan, normalizer=norm, aux_channels=None, blend_kind="cosine", batch_size=4
    )
    np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(b))
    assert np.array_equal(np.isnan(a), np.isnan(b))


def test_multichannel_stacks_aux_below_mag(tmp_path):
    """With a 1-extra-channel model that reads only the mag channel, output matches single-channel."""
    rng = np.random.default_rng(1)
    lr = rng.random((15, 15)).astype(np.float32)
    p = tmp_path / "lr.tif"
    minx, miny, maxx, maxy = _write_raster(p, lr)
    plan = plan_lr_patches(
        polygon_world=(minx, miny, maxx, maxy),
        lr_path=str(p),
        patch_lr_px=6,
        stride_lr_px=6,
        buffer_lr_px=0,
        scale=3,
        min_valid_frac=0.0,
    )
    norm = Normalizer(vmin=0.0, vmax=1.0)

    class MagOnly(torch.nn.Module):
        """Ignores aux channels: upsamples channel 0 only (like the mag-only baseline)."""

        def forward(self, x):
            return torch.nn.functional.interpolate(
                x[:, :1], scale_factor=3, mode="bicubic", align_corners=False
            )

    aux = np.full((1, *plan.lr_size), 0.5, dtype=np.float32)  # neutral extra channel
    single = reconstruct_region(BicubicModel(3), str(p), plan, normalizer=norm, blend_kind="ones")
    multi = reconstruct_region_multichannel(
        MagOnly(), str(p), plan, normalizer=norm, aux_channels=aux, blend_kind="ones"
    )
    np.testing.assert_allclose(np.nan_to_num(single), np.nan_to_num(multi), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# mag-input NaN-fill correctness fix (A4)
# --------------------------------------------------------------------------- #
def test_mag_input_nan_fill_matches_training():
    """Reconstruct fill default must equal training's 0.0 fill (not the old 0.5)."""
    assert MAG_INPUT_NAN_FILL == 0.0
    for fn in (reconstruct_region, reconstruct_region_multichannel):
        assert inspect.signature(fn).parameters["nan_fill"].default == MAG_INPUT_NAN_FILL


def test_infer_patch_fills_nan_with_given_value():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    patch = np.array([[0.2, np.nan], [np.nan, 0.8]], dtype=np.float32)
    out = infer_patch(Identity(), patch, nan_fill=0.0, device=torch.device("cpu"))
    assert out[0, 1] == 0.0 and out[1, 0] == 0.0  # NaNs filled with 0.0
    assert out[0, 0] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# normalization helpers (A3)
# --------------------------------------------------------------------------- #
def test_clip_scale01_scalar_and_tensor_bounds():
    x = torch.tensor([[-1.0, 0.0, 0.5, 1.0, 2.0]])
    got = clip_scale01(x, 0.0, 1.0)
    expected = torch.tensor([[0.0, 0.0, 0.5, 1.0, 1.0]])
    torch.testing.assert_close(got, expected)
    # broadcastable tensor bounds (the per-block case)
    vmin = torch.tensor([[0.0]])
    vmax = torch.tensor([[2.0]])
    torch.testing.assert_close(clip_scale01(x, vmin, vmax), (x.clamp(0.0, 2.0)) / 2.0)
    # numpy arrays go through the same implementation
    xn = np.array([-1.0, 0.5, 2.0])
    np.testing.assert_allclose(clip_scale01(xn, 0.0, 1.0), [0.0, 0.5, 1.0])


def test_center_scale01_maps_zero_to_half():
    v = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
    got = center_scale01(v, 5.0)
    torch.testing.assert_close(got, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_denorm01_inverts_clip_scale01():
    x = torch.tensor([[10.0, 55.0, 100.0]])
    vmin, vmax = torch.tensor([[10.0]]), torch.tensor([[100.0]])  # batched-bounds form
    torch.testing.assert_close(denorm01(clip_scale01(x, vmin, vmax), vmin, vmax), x)
    assert denorm01(np.float32(0.5), 0.0, 200.0) == 100.0


def test_pm1_round_trip_and_nan():
    x = torch.tensor([0.0, 0.5, 1.0, float("nan")])
    y = to_pm1(x)
    torch.testing.assert_close(y[:3], torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.isnan(y[3])  # NaN propagates for masked losses
    torch.testing.assert_close(from_pm1(y)[:3], x[:3])


def test_normalizer_eps_and_nan_fill_match_wa_loader():
    """Normalizer(eps=1e-8) reproduces the WA loader's historical arithmetic exactly."""
    arr = np.array([-500.0, 0.0, 250.0, np.nan], dtype=np.float32)
    vmin, vmax = -400.0, 300.0

    legacy_mag = ((np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)
    got_mag = Normalizer(vmin, vmax, eps=1e-8).normalize(arr)
    np.testing.assert_array_equal(np.nan_to_num(got_mag), np.nan_to_num(legacy_mag))

    legacy_robust = np.nan_to_num(legacy_mag, nan=0.5).astype(np.float32)
    got_robust = Normalizer(vmin, vmax, eps=1e-8).normalize(arr, nan_fill=0.5)
    np.testing.assert_array_equal(got_robust, legacy_robust)


# --------------------------------------------------------------------------- #
# Normalizer constructors (magsr.normalize)
# --------------------------------------------------------------------------- #
def test_normalizer_from_stats_and_json(tmp_path):
    import json

    stats = {
        "global": {"vmin": -100.0, "vmax": 300.0},
        "blocks": {"1": {"vmin": -50.0, "vmax": 150.0}},
    }
    p = tmp_path / "normalization.json"
    p.write_text(json.dumps(stats))

    g = Normalizer.from_json(p)
    assert (g.vmin, g.vmax, g.data_range) == (-100.0, 300.0, 400.0)

    b1 = Normalizer.from_json(p, block=1)
    assert (b1.vmin, b1.vmax) == (-50.0, 150.0)

    # from_stats accepts any vmin/vmax mapping (also the WA top-level layout).
    w = Normalizer.from_stats({"vmin": 0.0, "vmax": 2.0})
    arr = np.array([-1.0, 1.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(w.normalize(arr), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(w.denormalize(w.normalize(arr)), [0.0, 1.0, 2.0])


def test_normalizer_reexported_from_reconstruct():
    from magsr.normalize import Normalizer as CanonicalNormalizer

    assert Normalizer is CanonicalNormalizer  # magsr.reconstruct re-export stays the same class


# --------------------------------------------------------------------------- #
# shared CLI fragment (magsr.cli)
# --------------------------------------------------------------------------- #
def test_add_channel_args_defaults_and_dem_modes():
    import argparse
    from dataclasses import dataclass, field

    from magsr.cli import add_channel_args

    # No defaults object: mag-only path.
    p = argparse.ArgumentParser()
    add_channel_args(p)
    a = p.parse_args([])
    assert (a.use_dem, a.dem_mode, a.lr_aux) == (False, "grad", [])

    # Defaults flow from a TrainConfig-like object (the KSA rdn trainer uses relief).
    @dataclass
    class Cfg:
        use_dem: bool = True
        dem_mode: str = "relief"
        lr_aux: tuple = ("1VD",)

    p = argparse.ArgumentParser()
    add_channel_args(p, defaults=Cfg())
    a = p.parse_args([])
    assert (a.use_dem, a.dem_mode, a.lr_aux) == (True, "relief", ["1VD"])

    # Restricted dem_modes (WA pipeline): relief must be rejected.
    p = argparse.ArgumentParser()
    add_channel_args(p, dem_modes=("grad",))
    with pytest.raises(SystemExit):
        p.parse_args(["--dem-mode", "relief"])

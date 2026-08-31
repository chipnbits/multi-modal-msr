"""Stage-00 KSA preprocessing: snapping products onto the master grid.

These pin the behaviours the shipped dataset depends on — output geometry, the
validity mask, both mosaic paths, and the overlap rule — on synthetic rasters, so
they run without the (non-public) KSA data. `compare_rasters.py` covers the
against-the-real-dataset check; see README_ksa_data.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

PREP = Path(__file__).resolve().parents[1] / "scripts/build_ksa_dataset/00_ksa_preprocessing"


def _load(stem: str):
    spec = importlib.util.spec_from_file_location(stem.replace(".", "_"), PREP / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


snap = _load("03_snap_raster")
grid = _load("02_base_grid")

UTM37 = CRS.from_epsg(32637)
UTM38 = CRS.from_epsg(32638)


def write_raster(path: Path, arr, transform, crs, nodata=None, dtype=None):
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype or arr.dtype.name,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
    return path


@pytest.fixture
def master(tmp_path):
    """32x32 master at 60 m, all valid except a 4-column strip on the right."""
    a = np.ones((32, 32), dtype=np.uint8)
    a[:, 28:] = 0
    return write_raster(
        tmp_path / "master.tif", a, Affine(60, 0, 400_000, 0, -60, 2_700_000), UTM37, nodata=0
    )


def test_snap_matches_master_geometry_and_masks_outside(tmp_path, master):
    """Output lands on exactly the master grid and is NaN wherever the master is 0."""
    src = write_raster(
        tmp_path / "a.tif",
        np.full((32, 32), 5.0, dtype=np.float32),
        Affine(60, 0, 400_000, 0, -60, 2_700_000),
        UTM37,
    )
    out = snap.snap_rasters([src], master, tmp_path / "out.tif")
    with rasterio.open(out) as ds, rasterio.open(master) as m:
        assert (ds.width, ds.height) == (m.width, m.height)
        assert ds.transform == m.transform and ds.crs == m.crs
        assert ds.profile["dtype"] == "float32" and np.isnan(ds.nodata)
        a = ds.read(1)
    assert np.isnan(a[:, 28:]).all(), "masked strip must be NaN"
    assert np.allclose(a[:, :28], 5.0), "interior must carry the source value"


def test_homogeneous_tiles_take_the_single_pass_path(tmp_path, master):
    """Same CRS + resolution + lattice -> mosaic once, warp once."""
    t = Affine(60, 0, 400_000, 0, -60, 2_700_000)
    left = write_raster(tmp_path / "l.tif", np.ones((32, 16), np.float32), t, UTM37)
    right = write_raster(
        tmp_path / "r.tif", np.full((32, 16), 2.0, np.float32), t * Affine.translation(16, 0), UTM37
    )
    assert snap._homogeneous([left, right])[0] is True
    out = snap.snap_rasters([left, right], master, tmp_path / "o.tif")
    with rasterio.open(out) as ds:
        a = ds.read(1)
    assert np.allclose(a[:, :16], 1.0) and np.allclose(a[:, 16:28], 2.0)


def test_mixed_crs_tiles_take_the_per_tile_path(tmp_path, master):
    """Tiles in different CRSs cannot share a lattice, so each is warped separately."""
    a = write_raster(
        tmp_path / "a.tif",
        np.ones((32, 32), np.float32),
        Affine(60, 0, 400_000, 0, -60, 2_700_000),
        UTM37,
    )
    b = write_raster(
        tmp_path / "b.tif",
        np.full((32, 32), 3.0, np.float32),
        Affine(60, 0, 100_000, 0, -60, 2_700_000),
        UTM38,
    )
    ok, why = snap._homogeneous([a, b])
    assert ok is False and "CRS" in why
    out = snap.snap_rasters([a, b], master, tmp_path / "o.tif")
    with rasterio.open(out) as ds:
        assert ds.read(1).shape == (32, 32)


def test_overlap_first_wins_by_filename_not_folder(tmp_path, master):
    """`overlap="first"` resolves to the first *filename*, so the outcome does not
    depend on how the download happened to be foldered."""
    t = Affine(60, 0, 400_000, 0, -60, 2_700_000)
    (tmp_path / "zzz").mkdir()
    (tmp_path / "aaa").mkdir()
    # 'a.tif' sits in the later folder, 'b.tif' in the earlier one: sorting on the
    # full path would pick b, sorting on the filename picks a.
    a = write_raster(tmp_path / "zzz" / "a.tif", np.ones((32, 32), np.float32), t, UTM37)
    b = write_raster(tmp_path / "aaa" / "b.tif", np.full((32, 32), 9.0, np.float32), t, UTM37)
    out = snap.snap_rasters([b, a], master, tmp_path / "o.tif", overlap="first")
    with rasterio.open(out) as ds:
        assert np.allclose(ds.read(1)[:, :28], 1.0), "a.tif (first filename) must win"


def test_nonoverlapping_tiles_are_dropped(tmp_path, master):
    """A tile entirely off the master grid is skipped rather than warped."""
    on = write_raster(
        tmp_path / "on.tif",
        np.ones((32, 32), np.float32),
        Affine(60, 0, 400_000, 0, -60, 2_700_000),
        UTM37,
    )
    off = write_raster(
        tmp_path / "off.tif",
        np.ones((32, 32), np.float32),
        Affine(60, 0, 900_000, 0, -60, 2_000_000),
        UTM37,
    )
    with rasterio.open(master) as m:
        kept = snap.filter_overlapping([on, off], snap.read_master(master))
    assert [p.name for p in kept] == ["on.tif"]


def test_no_overlapping_inputs_raises(tmp_path, master):
    off = write_raster(
        tmp_path / "off.tif",
        np.ones((8, 8), np.float32),
        Affine(60, 0, 900_000, 0, -60, 2_000_000),
        UTM37,
    )
    with pytest.raises(RuntimeError, match="No input files overlap"):
        snap.snap_rasters([off], master, tmp_path / "o.tif")


def test_honor_nodata_controls_whether_source_nodata_is_masked(tmp_path, master):
    """The legacy LR sheets declare nodata=0 but carry meaningful zeros, so the
    single-pass path ignores declared nodata unless asked to honour it."""
    t = Affine(60, 0, 400_000, 0, -60, 2_700_000)
    a = np.full((32, 32), 4.0, dtype=np.float32)
    a[:8, :8] = 0.0
    src = write_raster(tmp_path / "z.tif", a, t, UTM37, nodata=0)

    kept = snap.snap_rasters([src], master, tmp_path / "keep.tif", honor_nodata=False)
    with rasterio.open(kept) as ds:
        assert np.allclose(ds.read(1)[:8, :8], 0.0), "zeros should pass through as data"

    masked = snap.snap_rasters([src], master, tmp_path / "mask.tif", honor_nodata=True)
    with rasterio.open(masked) as ds:
        assert np.isnan(ds.read(1)[:8, :8]).all(), "zeros should become NaN"


def test_master_grid_is_ones_with_zero_nodata(tmp_path):
    """02_base_grid writes a uint8 all-1s extent grid; 0 is reserved for outside-AOI."""
    params = {
        "utm_epsg": 32637,
        "pixel_size": 60,
        "nx": 512,
        "ny": 256,
        "minx": 400_000,
        "miny": 2_600_000,
        "maxx": 430_720,
        "maxy": 2_615_360,
    }
    out = grid.create_grid(params, {"paths": {"output": str(tmp_path / "g.tif")}})
    with rasterio.open(out) as ds:
        assert ds.profile["dtype"] == "uint8" and ds.nodata == 0
        assert (ds.width, ds.height) == (512, 256)
        assert ds.transform == Affine(60, 0, 400_000, 0, -60, 2_615_360)
        assert ds.crs == UTM37
        assert (ds.read(1) == 1).all()


def test_best_grid_dims_floors_to_whole_blocks():
    nx, ny, w, h = grid.best_grid_dims(60_000, 40_000, pixel_size=60, target_px=256)
    assert nx % 256 == 0 and ny % 256 == 0
    assert nx * 60 == w and ny * 60 == h
    assert nx <= 60_000 / 60 and ny <= 40_000 / 60

"""Smoke tests for magsr package."""

import numpy as np
import torch
from rasterio.transform import Affine

import magsr
from magsr.datasets import (
    KSAAlignedConfig,
    KSAShieldAlignedDataset,
    PatchGridSpec,
    PatchIndex,
    PatchWindow,
    WAConfig,
    sliding_window_patches,
)
from magsr.models import RDNpp, build_from_spec, ensure_soup


def test_import():
    assert magsr is not None


def test_mask_path_falls_back_to_committed_base_grids(tmp_path):
    """A from-scratch build (root with no 00_base_grid/) resolves the mask from the
    committed data/ksa_base_grids/ so no manual copy step is needed."""
    cfg = KSAAlignedConfig.from_yaml(root=tmp_path)  # empty root, no 00_base_grid/
    assert cfg.mask_path == magsr.DATA_DIR / "ksa_base_grids" / "magnetic_mask_grid60m.tif"
    assert cfg.mask_path.exists()

    in_root = tmp_path / "00_base_grid" / "magnetic_mask_grid60m.tif"
    in_root.parent.mkdir(parents=True)
    in_root.write_bytes(b"")  # a real dataset tree's own mask wins when present
    assert cfg.mask_path == in_root


def test_patch_grid_spec_stride_m():
    spec = PatchGridSpec(patch_px=128, stride_px=64, min_valid_frac=0.95)
    assert spec.stride_m(60.0) == 64 * 60.0
    assert spec.patch_m(60.0) == 128 * 60.0


def test_patch_window_from_pixel_math():
    # Simple north-up transform: 10 m pixels, origin at (1000, 5000).
    transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 5000.0)
    tw = PatchWindow.from_pixel(
        source_id="test/a",
        row_px=20,
        col_px=30,
        transform=transform,
        patch_px=128,
        valid_frac=1.0,
    )
    # left = 1000 + 30*10 = 1300; top = 5000 - 20*10 = 4800
    assert tw.left == 1300.0
    assert tw.top == 4800.0
    assert tw.right == 1300.0 + 128 * 10
    assert tw.bottom == 4800.0 - 128 * 10
    assert tw.width == 1280.0
    assert tw.height == 1280.0
    assert tw.name == "test_a_r00020_c00030"


def test_sliding_window_patches_synthetic():
    # 16x16 all-valid mask, patch 4, stride 4 -> 4x4 = 16 patches.
    mask = np.ones((16, 16), dtype=bool)
    meta = {
        "transform": Affine(1.0, 0.0, 0.0, 0.0, -1.0, 16.0),
        "crs": None,
        "nodata": None,
        "width": 16,
        "height": 16,
    }
    spec = PatchGridSpec(patch_px=4, stride_px=4, min_valid_frac=1.0)
    patches = sliding_window_patches(mask, meta, spec, source_id="synthetic")
    assert len(patches) == 16
    assert all(t.valid_frac == 1.0 for t in patches)

    # Stride 2 on 16x16 with 4px patches -> 7x7 = 49 patches (positions 0..12 step 2).
    spec2 = PatchGridSpec(patch_px=4, stride_px=2, min_valid_frac=1.0)
    patches2 = sliding_window_patches(mask, meta, spec2, source_id="synthetic")
    assert len(patches2) == 49


def test_cell_sweep_allow_partial_covers_raster_edge():
    from magsr.datasets.patching import CellGridSpec, bucket_patches_by_cell

    # patch 4 @ stride 2, 3x3-patch cells (12 px), 1-patch buffer -> pitch 16.
    # On a 26-px raster the second lattice row/col (origin 16) only fits
    # partially, so a full-fit-only sweep loses every patch beyond px 12.
    spec = CellGridSpec(
        patch=PatchGridSpec(patch_px=4, stride_px=2, min_valid_frac=1.0),
        cell_patches=3,
        buffer_patches=1,
    )
    mask = np.ones((26, 26), dtype=bool)
    meta = {
        "transform": Affine(1.0, 0.0, 0.0, 0.0, -1.0, 26.0),
        "crs": None,
        "nodata": None,
        "width": 26,
        "height": 26,
    }
    patches = sliding_window_patches(mask, meta, spec.patch, source_id="s")
    full_cells = sliding_window_patches(mask, meta, spec.as_cell_sweep(), source_id="s")
    all_cells = sliding_window_patches(mask, meta, spec.as_cell_sweep(), source_id="s", allow_partial=True)
    assert len(full_cells) == 1  # only the (0, 0) lattice cell fits flush
    assert len(all_cells) == 4  # partial trailing row/col cells kept

    # The bottom/right band (rows/cols >= pitch) is reachable with partials...
    covered = {t.row_px for ts in bucket_patches_by_cell(patches, all_cells, spec).values() for t in ts}
    assert max(covered) >= spec.pitch_px
    # ...and silently lost with full-fit-only cells (the original bug).
    lost = {t.row_px for ts in bucket_patches_by_cell(patches, full_cells, spec).values() for t in ts}
    assert max(lost) < spec.pitch_px


def test_sliding_window_patches_drops_below_threshold():
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4, :4] = True  # top-left quadrant valid
    meta = {
        "transform": Affine(1.0, 0.0, 0.0, 0.0, -1.0, 8.0),
        "crs": None,
        "nodata": None,
        "width": 8,
        "height": 8,
    }
    spec = PatchGridSpec(patch_px=4, stride_px=4, min_valid_frac=1.0)
    patches = sliding_window_patches(mask, meta, spec, source_id="synthetic")
    # Only the top-left 4x4 patch is fully valid.
    assert len(patches) == 1
    assert patches[0].row_px == 0 and patches[0].col_px == 0


def test_patch_index_save_load_roundtrip(tmp_path):
    spec = PatchGridSpec(patch_px=4, stride_px=4, min_valid_frac=0.9)
    patches = [
        PatchWindow(
            source_id="test/a",
            row_px=0,
            col_px=0,
            left=0.0,
            bottom=0.0,
            right=4.0,
            top=4.0,
            valid_frac=1.0,
            epsg="EPSG:32637",
        ),
        PatchWindow(
            source_id="test/a",
            row_px=0,
            col_px=4,
            left=4.0,
            bottom=0.0,
            right=8.0,
            top=4.0,
            valid_frac=0.95,
            epsg="EPSG:32637",
        ),
    ]
    idx = PatchIndex(spec=spec, product="AMF_RTP", patches=patches, extra={"block": 2})
    path = tmp_path / "idx.json"
    idx.save(path)
    loaded = PatchIndex.load(path)
    assert loaded.spec == spec
    assert loaded.product == "AMF_RTP"
    assert loaded.extra == {"block": 2}
    assert len(loaded) == 2
    assert loaded.patches == patches


def test_patch_index_split():
    spec = PatchGridSpec(patch_px=4, stride_px=4, min_valid_frac=0.9)
    patches = [PatchWindow("x", i, 0, 0, 0, 4, 4, 1.0) for i in range(10)]
    idx = PatchIndex(spec=spec, product="x", patches=patches)
    parts = idx.split({"train": 0.8, "val": 0.2}, seed=0)
    assert len(parts["train"]) == 8
    assert len(parts["val"]) == 2
    all_names = {t.name for t in parts["train"]} | {t.name for t in parts["val"]}
    assert len(all_names) == 10


def test_config_from_yaml_matches_yaml_values():
    """Guard against drift between `configs/datasets.yaml` and the dataclasses.

    Each `from_yaml()` must populate the fields the build scripts and notebook
    read downstream. This doesn't hit env vars (the aligned `root` field uses
    lazy default_factory and isn't touched unless a path property is called).
    """
    wa = WAConfig.from_yaml()
    assert wa.product == "MAG"
    assert wa.scale_factor == 4
    assert wa.patch_px == 128 and wa.stride_px == 128
    assert 0 < wa.val_fraction < 1

    try:
        ksaa = KSAAlignedConfig.from_yaml()
    except RuntimeError:
        from pathlib import Path

        ksaa = KSAAlignedConfig.from_yaml(root=Path("/tmp"))
    assert ksaa.patch_px == 132 and ksaa.stride_px == 66
    assert ksaa.cellgrid.cell_patches == 8 and ksaa.cellgrid.buffer_patches == 1
    assert ksaa.cellgrid.val_frac == 0.10 and ksaa.cellgrid.test_frac == 0.10
    assert ksaa.cell_grid_spec.pitch_px == (8 + 1) * 132
    assert ksaa.hr_products == ("AMF_RTP",)
    assert ksaa.load_dem is False


def test_rdnpp_x3_default_up_factors():
    assert RDNpp(upscale=3, nb=1, nf=8, gc=4).build_kwargs["up_factors"] == [3, 1]


def test_rdnpp_spec_roundtrip():
    """model_spec must fully describe the upsample graph: a model rebuilt from
    its build_kwargs loads the original state_dict for either x3 variant."""
    for up_factors in ([3], [3, 1]):
        model = RDNpp(upscale=3, nb=1, nf=8, gc=4, up_factors=up_factors)
        assert model.build_kwargs["up_factors"] == up_factors
        rebuilt = build_from_spec({"name": "RDNpp", "kwargs": dict(model.build_kwargs)})
        rebuilt.load_state_dict(model.state_dict())  # raises if graphs differ


def test_ensure_soup_averages_floats_and_keeps_metadata(tmp_path):
    """The soup is the midpoint of _best/_best_rmse for float tensors, copies int buffers from
    _best, inherits its model_spec, and is skipped (not rebuilt) once it exists."""
    spec = {"name": "RDNpp", "kwargs": {"upscale": 3, "nb": 1, "nf": 8, "gc": 4}}
    for var, w, ep in (("best", 1.0, 10), ("best_rmse", 3.0, 20)):
        torch.save(
            {
                "model": {"w": torch.full((2,), w), "n": torch.tensor(7)},
                "model_spec": spec,
                "epoch": ep,
                "optim": {"junk": 1},
            },
            tmp_path / f"run_{var}.pt",
        )

    out = ensure_soup(tmp_path, "run")
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert torch.equal(ck["model"]["w"], torch.full((2,), 2.0))
    assert ck["model"]["n"].item() == 7  # int buffer: copied from _best, not averaged
    assert ck["model_spec"] == spec and ck["epoch"] == "10+20" and "optim" not in ck

    out.write_bytes(b"sentinel")  # existing soup is reused...
    assert ensure_soup(tmp_path, "run") == out and out.read_bytes() == b"sentinel"
    ensure_soup(tmp_path, "run", force=True)  # ...unless forced
    assert torch.load(out, map_location="cpu", weights_only=False)["epoch"] == "10+20"

    assert ensure_soup(tmp_path, "missing_run") is None


def test_assemble_lr_input_channel_order_and_none_omission():
    """assemble_lr_input cats channels in training order (mag, DEM, LR-aux, MS) and omits any
    group whose tensor is None. The channel sub-methods are stubbed with distinct per-group marker
    values so the concatenation order is directly observable without touching rasters/norm_stats."""

    class _Stub:
        def normalize(self, lr, blocks=None, *, nan_fill=None):
            return torch.zeros(lr.shape[0], 1, 2, 2)  # mag -> 0.0

        def dem_features(self, dem):
            return torch.ones(dem.shape[0], 2, 2, 2)  # DEM -> 1.0 (2 channels)

        def lr_aux_to_channels(self, aux):
            return torch.full((aux.shape[0], 1, 2, 2), 2.0)  # LR-aux -> 2.0

        def ms_to_channels(self, ms):
            return torch.full((ms.shape[0], 1, 2, 2), 3.0)  # MS -> 3.0

        assemble_lr_input = KSAShieldAlignedDataset.assemble_lr_input

    ds = _Stub()
    lr = torch.zeros(2, 1, 2, 2)
    t = torch.zeros(2, 1, 2, 2)  # placeholder tensor for each optional group

    full = ds.assemble_lr_input(lr, [1, 1], dem=t, lr_aux=t, ms=t)
    assert [full[0, c, 0, 0].item() for c in range(full.shape[1])] == [0.0, 1.0, 1.0, 2.0, 3.0]

    no_dem = ds.assemble_lr_input(lr, [1, 1], lr_aux=t, ms=t)
    assert [no_dem[0, c, 0, 0].item() for c in range(no_dem.shape[1])] == [0.0, 2.0, 3.0]

    mag_only = ds.assemble_lr_input(lr, [1, 1])
    assert mag_only.shape[1] == 1 and mag_only[0, 0, 0, 0].item() == 0.0

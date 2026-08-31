"""Western Australia aeromagnetic survey dataset.

Eager-preprocessing pipeline: reads HR/LR GeoTIFFs, enumerates patches with the
shared `sliding_window_patches` helper, and saves normalized .npy pairs to disk.
`WAGoldfieldsPatchDataset` just loads those .npy files and wraps them in the
shared `{'hr', 'lr', 'meta'}` sample format.

All patch-build + split parameters flow through `WAConfig`, which loads from
`configs/datasets.yaml` by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from magsr import DATA_DIR
from magsr.datasets.config import load_yaml_section
from magsr.datasets.io import RasterSource
from magsr.datasets.patching import PatchGridSpec, PatchWindow, compute_mask, sliding_window_patches
from magsr.normalize import Normalizer, center_scale01


@dataclass(frozen=True)
class WAConfig:
    """All WA-specific paths + patch-build + split parameters.

    `data_dir` holds the source rasters; `patch_dir` holds the preprocessed
    .npy patches + manifest. Build/load params (`patch_px`, `val_fraction`, ...)
    come from `configs/datasets.yaml` via `from_yaml()`.
    """

    data_dir: Path = field(default_factory=lambda: DATA_DIR / "WA")
    patch_dir: Path = field(default_factory=lambda: DATA_DIR / "processed" / "wa" / "patches")
    product: str = "MAG"
    # Source files in data_dir: HR/LR rasters written by build_wa_dataset/01_clip; the
    # hand-drawn Train/Test split polygons.
    hr_filename: str = "Goldfields_20m_HR.tif"
    lr_filename: str = "Goldfields_80m_LR.tif"
    split_filename: str = "goldfields_split.geojson"
    scale_factor: int = 4
    patch_px: int = 128
    stride_px: int = 128
    min_valid_frac: float = 0.95
    q_low: int = 1
    q_high: int = 99
    val_fraction: float = 0.1
    seed: int = 42
    # --- extra LR input modalities (mirrors the KSA multi-channel setup) ---
    # lr_aux: derived LR products concatenated as input channels. "1VD" = first vertical
    #   derivative. Source precedence: LR raster band 2 (the GSWA official 1VD clipped in by
    #   01_clip) -> external snapped raster (vd_filename) -> FFT |k| derivative of the LR mag.
    lr_aux: tuple[str, ...] = ()
    # External fallback only: the GSWA 1VD-of-TMI is normally clipped straight into LR band 2 by
    # build_wa_dataset/01_clip_wa_goldfields.py, which the loader prefers. This standalone snapped
    # raster is the legacy source, used only if the LR grid is single-band. Set to "" to compute
    # 1VD via FFT instead.
    vd_filename: str | None = "snapped_cubicspline_1vd80m.tif"
    # use_dem + dem_mode="grad": add the 2-channel DEM slope gradient (dz/dx, dz/dy) from a DEM
    #   pre-snapped onto the LR magnetic grid (build_wa_dataset/03_fetch_snap_wa_dem.py).
    use_dem: bool = False
    dem_mode: str = "grad"
    # Bare-earth DTM (Geoscience Australia 1" SRTM-derived DEM-S) snapped to the LR grid by
    # scripts/build_wa_dataset/03_fetch_snap_wa_dem.py.
    dem_filename: str = "snapped_cubicspline_dtm80m.tif"

    @classmethod
    def default(cls) -> "WAConfig":
        return cls()

    @classmethod
    def from_yaml(cls, path: Path | None = None, **overrides: Any) -> "WAConfig":
        """Load the `wa` section of `configs/datasets.yaml` + apply overrides."""
        data = load_yaml_section("wa", path=path)
        data.update(overrides)
        return cls(**data)

    @property
    def patch_grid_spec(self) -> PatchGridSpec:
        return PatchGridSpec(
            patch_px=self.patch_px,
            stride_px=self.stride_px,
            min_valid_frac=self.min_valid_frac,
        )


def compute_global_percentiles(
    hr_path: Path,
    train_polygon,
    q_low: int = 1,
    q_high: int = 99,
    strip_height: int = 1024,
) -> dict[str, float]:
    """Robust percentile-based vmin/vmax inside the training polygon.

    Accumulates valid HR pixels strip-by-strip, then calls `np.percentile`
    once. Returned dict is written verbatim to `normalization.json` and
    consumed lazily by `WAGoldfieldsPatchDataset` at load time.

    Args:
        hr_path: Path to the high-resolution raster.
        train_polygon: Shapely geometry defining the training area.
        q_low, q_high: Percentiles (0-100) used as clip bounds.
        strip_height: Number of rows to read at a time for memory efficiency.
    """
    with rasterio.open(hr_path) as ds:
        bounds = train_polygon.bounds
        win = from_bounds(*bounds, transform=ds.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))

        row_off = int(win.row_off)
        col_off = int(win.col_off)
        total_rows = int(win.height)
        total_cols = int(win.width)

        valid_chunks: list[np.ndarray] = []
        for start in range(0, total_rows, strip_height):
            h = min(strip_height, total_rows - start)
            strip_win = rasterio.windows.Window(col_off, row_off + start, total_cols, h)
            data = ds.read(1, window=strip_win).astype(np.float32)

            valid = data[data != ds.nodata]
            if valid.size:
                valid_chunks.append(valid)

    if not valid_chunks:
        raise ValueError(f"No valid HR pixels inside the training polygon for {hr_path}")

    pixels = np.concatenate(valid_chunks)
    vmin, vmax = np.percentile(pixels, [q_low, q_high])
    return {"vmin": float(vmin), "vmax": float(vmax), "q_low": int(q_low), "q_high": int(q_high)}


# ---------------------------------------------------------------------------
# Extra LR modality helpers (1VD via FFT, DEM slope gradient).
# ---------------------------------------------------------------------------
def compute_1vd(mag: np.ndarray) -> np.ndarray:
    """First vertical derivative of an LR magnetic patch via the FFT |k| operator.

    Reuses the validated ``magsr.fourier.first_vertical_derivative`` (dx=dy=1, per-pixel
    wavenumber — absolute scale is irrelevant for a normalized input channel). NaN nodata is
    filled with the patch mean before the transform (the FFT cannot take NaN) and re-masked to
    NaN afterwards so the loader can fill it with the neutral 0.5. The WA grid is a TMI anomaly
    (no measured 1VD), so the derivative is taken on it directly.
    """
    import torch

    from magsr.fourier import first_vertical_derivative

    valid = np.isfinite(mag)
    fill = float(np.nanmean(mag)) if valid.any() else 0.0
    t = torch.from_numpy(np.where(valid, mag, fill).astype(np.float32))
    vd = first_vertical_derivative(t, 1.0, 1.0).numpy().astype(np.float32)
    vd[~valid] = np.nan
    return vd


def dem_gradient(dem: np.ndarray) -> np.ndarray:
    """Central-difference slope gradient of a DEM patch → (2, H, W) = [dz/dx, dz/dy].

    ``np.gradient`` is size-preserving (central diff interior, one-sided edges). Units are
    metres per LR pixel. NaN propagates and is filled with 0.5 by the loader after normalization.
    """
    gy, gx = np.gradient(dem.astype(np.float32))
    return np.stack([gx, gy], axis=0).astype(np.float32)


def enumerate_patch_windows(
    hr_path: Path,
    split_gdf,
    cfg: WAConfig,
    *,
    mask: np.ndarray | None = None,
    mask_meta: dict | None = None,
) -> list[PatchWindow]:
    """Per-polygon patching with HR + optional science-mask filtering.

    Computes the HR validity mask once, AND-s with any supplied science mask,
    then runs `sliding_window_patches` per train/test polygon and filters to
    (row, col) that are mod-scale_factor aligned so LR patches fall on integer
    pixel boundaries.
    """
    spec = cfg.patch_grid_spec
    src = RasterSource(hr_path)
    hr_mask, hr_meta = compute_mask(src)
    if mask is not None:
        row_off = mask_meta.get("row_off", 0) if mask_meta else 0
        col_off = mask_meta.get("col_off", 0) if mask_meta else 0
        if row_off == 0 and col_off == 0 and mask.shape == hr_mask.shape:
            hr_mask &= mask.astype(bool)
        else:
            h, w = mask.shape
            hr_mask[row_off : row_off + h, col_off : col_off + w] &= mask.astype(bool)

    windows: list[PatchWindow] = []
    for _, row in split_gdf.iterrows():
        set_type = row["set_type"]
        polygon = row.geometry
        patches = sliding_window_patches(
            hr_mask,
            hr_meta,
            spec,
            source_id=f"wa/{set_type}",
            polygon=polygon,
        )
        windows.extend(
            t for t in patches if t.row_px % cfg.scale_factor == 0 and t.col_px % cfg.scale_factor == 0
        )
    return windows


def _read_patch_pair(
    hr_ds,
    lr_ds,
    win: PatchWindow,
    *,
    cfg: WAConfig,
    dem_ds=None,
    vd_ds=None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read one HR/LR pair as raw nT with nodata → NaN.

    Returns ``(hr (H,W), lr_stack (C,h,w))`` where the LR channel order is
    ``[MAG, (1VD if "1VD" in cfg.lr_aux), (DEM elevation if cfg.use_dem)]`` — all raw. 1VD source
    precedence: LR raster band 2 (the GSWA 1VD baked into the 2-band LR grid) → the external
    pre-snapped 1VD raster (``vd_ds``) → FFT |k| of the LR mag patch. DEM read from the pre-snapped
    LR-grid raster. Normalization (incl. DEM→gradient) is deferred to the loader. Returns None if
    either magnetic raster's nodata coverage exceeds 1 - min_valid_frac.
    """
    hr_patch_px = cfg.patch_px
    lr_patch_px = hr_patch_px // cfg.scale_factor
    min_valid_frac = cfg.min_valid_frac

    hr_data = hr_ds.read(1, window=Window(win.col_px, win.row_px, hr_patch_px, hr_patch_px)).astype(
        np.float32
    )
    hr_nodata = hr_data == hr_ds.nodata
    if hr_nodata.any():
        if hr_nodata.mean() > (1.0 - min_valid_frac):
            return None
        hr_data = np.where(hr_nodata, np.nan, hr_data)

    lr_window = from_bounds(win.left, win.bottom, win.right, win.top, transform=lr_ds.transform)
    lr_data = lr_ds.read(1, window=lr_window, out_shape=(lr_patch_px, lr_patch_px)).astype(np.float32)
    lr_nodata = lr_data == lr_ds.nodata
    if lr_nodata.any():
        if lr_nodata.mean() > (1.0 - min_valid_frac):
            return None
        lr_data = np.where(lr_nodata, np.nan, lr_data)

    channels = [lr_data]
    if "1VD" in cfg.lr_aux:
        if lr_ds.count >= 2:  # 1VD baked into the 2-band LR grid (band 2) — preferred source
            vd = lr_ds.read(2, window=lr_window, out_shape=(lr_patch_px, lr_patch_px)).astype(np.float32)
            if lr_ds.nodata is not None:
                vd = np.where(vd == lr_ds.nodata, np.nan, vd)
            channels.append(vd)
        elif vd_ds is not None:
            vd = vd_ds.read(1, window=lr_window, out_shape=(lr_patch_px, lr_patch_px)).astype(np.float32)
            if vd_ds.nodata is not None:
                vd = np.where(vd == vd_ds.nodata, np.nan, vd)
            channels.append(vd)
        else:
            channels.append(compute_1vd(lr_data))
    if cfg.use_dem:
        if dem_ds is None:
            raise ValueError("cfg.use_dem is set but no dem_ds was opened")
        dem = dem_ds.read(1, window=lr_window, out_shape=(lr_patch_px, lr_patch_px)).astype(np.float32)
        if dem_ds.nodata is not None:
            dem = np.where(dem == dem_ds.nodata, np.nan, dem)
        channels.append(dem)

    lr_stack = np.stack(channels, axis=0)  # (C, h, w)
    return hr_data, lr_stack


def generate_wa_patch_pairs(
    hr_path: Path,
    lr_path: Path,
    split_gdf,
    output_dir: Path,
    stats: dict[str, float],
    cfg: WAConfig,
    *,
    mask: np.ndarray | None = None,
    mask_meta: dict | None = None,
) -> dict:
    """Generate raw HR/LR patch pairs (nT, nodata → NaN) plus normalization.json.

    Patches are written un-normalized; `stats` (from `compute_global_percentiles`)
    is serialized alongside them so `WAGoldfieldsPatchDataset` can clip + scale
    to [0, 1] at load time. Train patches land in ``output_dir/train/`` and test
    patches in ``output_dir/test/`` — the train/val split is re-seeded at dataset
    construction time.
    """
    manifest: dict = {"train": [], "test": [], "counts": {}}

    windows = enumerate_patch_windows(
        hr_path,
        split_gdf,
        cfg,
        mask=mask,
        mask_meta=mask_meta,
    )

    # LR channel order written into each {name}_lr.npy (used by the loader to slice/normalize).
    lr_channels = ["MAG"]
    if "1VD" in cfg.lr_aux:
        lr_channels.append("1VD")
    if cfg.use_dem:
        lr_channels.append("DEM")
    vd_idx = lr_channels.index("1VD") if "1VD" in lr_channels else None
    dem_idx = lr_channels.index("DEM") if "DEM" in lr_channels else None

    dem_path = cfg.data_dir / cfg.dem_filename if cfg.use_dem else None
    dem_ds = rasterio.open(dem_path) if dem_path is not None else None

    # Accumulate robust stats for the new channels over TRAIN patches only.
    vd_vals: list[np.ndarray] = []
    grad_abs: list[np.ndarray] = []

    with rasterio.open(hr_path) as hr_ds, rasterio.open(lr_path) as lr_ds:
        # 1VD source precedence: LR band 2 (baked-in GSWA 1VD) → external snapped raster → FFT.
        want_vd = "1VD" in cfg.lr_aux
        lr_has_vd_band = lr_ds.count >= 2
        vd_path = (
            cfg.data_dir / cfg.vd_filename if (want_vd and not lr_has_vd_band and cfg.vd_filename) else None
        )
        vd_ds = rasterio.open(vd_path) if vd_path is not None else None
        if want_vd:
            src = (
                "LR band 2"
                if lr_has_vd_band
                else (f"GSWA grid {cfg.vd_filename}" if vd_ds else "computed FFT |k|")
            )
            print(f"1VD source: {src}")
        for win in windows:
            pair = _read_patch_pair(hr_ds, lr_ds, win, cfg=cfg, dem_ds=dem_ds, vd_ds=vd_ds)
            if pair is None:
                continue
            hr_raw, lr_stack = pair
            split = "train" if win.source_id.endswith("/Train") else "test"
            _save_patch(output_dir / split, win.name, hr_raw, lr_stack)
            manifest[split].append(win.name)
            if split == "train":
                if vd_idx is not None:
                    v = lr_stack[vd_idx]
                    vd_vals.append(v[np.isfinite(v)])
                if dem_idx is not None:
                    g = dem_gradient(lr_stack[dem_idx])
                    grad_abs.append(np.abs(g[np.isfinite(g)]))

    if dem_ds is not None:
        dem_ds.close()
    if vd_ds is not None:
        vd_ds.close()

    manifest["counts"] = {k: len(v) for k, v in manifest.items() if k != "counts"}
    print(f"Patch counts: {manifest['counts']}")

    # Merge per-channel stats into the saved normalization.json so the loader is self-describing.
    stats = dict(stats)
    stats["lr_channels"] = lr_channels
    if vd_idx is not None and vd_vals:
        allv = np.concatenate(vd_vals)
        stats["vd_vmin"], stats["vd_vmax"] = float(np.percentile(allv, 0.5)), float(
            np.percentile(allv, 99.5)
        )
        print(f"1VD robust range: [{stats['vd_vmin']:.4f}, {stats['vd_vmax']:.4f}]")
    if dem_idx is not None and grad_abs:
        stats["dem_grad_clip"] = float(np.percentile(np.concatenate(grad_abs), 99.5))
        print(f"DEM grad clip (q99.5 |grad|): {stats['dem_grad_clip']:.3f} m/LRpx")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "normalization.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def _save_patch(split_dir: Path, name: str, hr: np.ndarray, lr: np.ndarray) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / f"{name}_hr.npy", hr)
    np.save(split_dir / f"{name}_lr.npy", lr)


class WAGoldfieldsPatchDataset:
    """Torch-compatible dataset loading WA patch pairs from .npy.

    HR is the magnetic patch ``(H, W)``. LR is a stacked ``(C, h, w)`` array whose channel
    order is recorded in ``normalization.json["lr_channels"]`` (``MAG`` first, then optional
    ``1VD`` and ``DEM``). On every ``__getitem__`` each channel is normalized to ``[0, 1]``
    and emitted as a separate LR product so ``pool_collate`` stacks them into the input tensor:

      MAG     percentile clip ``[vmin, vmax]`` → [0,1]   (same as HR)
      1VD     robust clip ``[vd_vmin, vd_vmax]`` → [0,1], NaN→0.5
      DEM     central-difference slope gradient → 2 channels ``DEMGX``/``DEMGY``,
              ``(g/(2·dem_grad_clip)+0.5)`` clamped to [0,1], NaN→0.5  (dem_mode="grad")

    Which optional channels are emitted is controlled by ``lr_aux`` / ``use_dem`` (a subset of
    what's stored), so the in=1 magnetic baseline can reuse the same patches. NaNs in HR/MAG
    propagate so callers can mask on them. Use `magsr.normalize.Normalizer.denormalize` to invert MAG.
    """

    def __init__(
        self,
        patch_dir: Path,
        names: list[str],
        *,
        product: str = "MAG",
        stats: dict[str, float] | None = None,
        lr_aux: tuple[str, ...] = (),
        use_dem: bool = False,
        dem_mode: str = "grad",
    ):
        self.patch_dir = Path(patch_dir)
        self.names = list(names)
        self.product = product
        self.lr_aux = tuple(lr_aux)
        self.use_dem = use_dem
        self.dem_mode = dem_mode

        if stats is None:
            stats = _load_stats(self.patch_dir)
        self.vmin = float(stats["vmin"])
        self.vmax = float(stats["vmax"])
        self.lr_channels = list(stats.get("lr_channels", ["MAG"]))
        self.vd_vmin = stats.get("vd_vmin")
        self.vd_vmax = stats.get("vd_vmax")
        self.dem_grad_clip = stats.get("dem_grad_clip")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        name = self.names[idx]
        hr = np.load(self.patch_dir / f"{name}_hr.npy")
        lr = np.load(self.patch_dir / f"{name}_lr.npy")
        if lr.ndim == 2:  # legacy single-channel patches
            lr = lr[None]
        ch = {n: i for i, n in enumerate(self.lr_channels)}

        lr_out: dict[str, "torch.Tensor"] = {self.product: torch.from_numpy(self._norm_mag(lr[ch["MAG"]]))}
        if "1VD" in self.lr_aux:
            lr_out["1VD"] = torch.from_numpy(self._norm_robust(lr[ch["1VD"]], self.vd_vmin, self.vd_vmax))
        if self.use_dem and self.dem_mode == "grad":
            g = self._norm_grad(dem_gradient(lr[ch["DEM"]]))  # (2, h, w)
            lr_out["DEMGX"] = torch.from_numpy(g[0])
            lr_out["DEMGY"] = torch.from_numpy(g[1])

        return {
            "hr": {self.product: torch.from_numpy(self._norm_mag(hr))},
            "lr": lr_out,
            "meta": {"name": name},
        }

    # eps=1e-8 is this loader's historical denominator guard — part of the numeric
    # fingerprint of every WA-trained checkpoint, so it stays.
    def _norm_mag(self, arr: np.ndarray) -> np.ndarray:
        return Normalizer(self.vmin, self.vmax, eps=1e-8).normalize(arr)

    @staticmethod
    def _norm_robust(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        return Normalizer(vmin, vmax, eps=1e-8).normalize(arr, nan_fill=0.5)

    def _norm_grad(self, g: np.ndarray) -> np.ndarray:
        out = center_scale01(g, float(self.dem_grad_clip))
        return np.nan_to_num(out, nan=0.5).astype(np.float32)


def _load_stats(patch_dir: Path) -> dict[str, float]:
    """Look for normalization.json next to patch_dir or inside its parent."""
    for candidate in (patch_dir / "normalization.json", patch_dir.parent / "normalization.json"):
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No normalization.json found near {patch_dir}. "
        "Re-run scripts/build_wa_dataset/04_patch_wa_dataset.py."
    )


def build_wa_datasets(
    cfg: WAConfig | None = None,
    **overrides: Any,
) -> dict[str, WAGoldfieldsPatchDataset]:
    """Load preprocessed WA patches and return train/val/test datasets.

    Splits the train pool randomly into train/val using `cfg.val_fraction` +
    `cfg.seed`. Must run `scripts/build_wa_dataset/04_patch_wa_dataset.py`
    first to produce the manifest.

    Args:
        cfg: `WAConfig` to use. Defaults to `WAConfig.from_yaml(**overrides)`
            when `None`; when a config is passed, `overrides` are applied on top.
    """
    if cfg is None:
        cfg = WAConfig.from_yaml(**overrides)
    elif overrides:
        cfg = replace(cfg, **overrides)

    manifest_path = cfg.patch_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run "
            "scripts/build_wa_dataset/04_patch_wa_dataset.py to preprocess patches."
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    train_pool = list(manifest["train"])
    test_names = list(manifest["test"])

    rng = np.random.default_rng(cfg.seed)
    n_val = max(1, int(len(train_pool) * cfg.val_fraction))
    val_idx = set(rng.permutation(len(train_pool))[:n_val].tolist())
    train_names = [n for i, n in enumerate(train_pool) if i not in val_idx]
    val_names = [n for i, n in enumerate(train_pool) if i in val_idx]

    stats = _load_stats(cfg.patch_dir)
    kwargs = {
        "product": cfg.product,
        "stats": stats,
        "lr_aux": cfg.lr_aux,
        "use_dem": cfg.use_dem,
        "dem_mode": cfg.dem_mode,
    }
    return {
        "train": WAGoldfieldsPatchDataset(cfg.patch_dir / "train", train_names, **kwargs),
        "val": WAGoldfieldsPatchDataset(cfg.patch_dir / "train", val_names, **kwargs),
        "test": WAGoldfieldsPatchDataset(cfg.patch_dir / "test", test_names, **kwargs),
    }

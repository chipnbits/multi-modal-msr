"""Rasterio-only raster I/O wrappers.

`RasterSource` wraps a rasterio dataset handle with lazy-open + cache and
windowed reads. Used by patch-mask construction and by the runtime sample
loaders in `western_australia` and `ksa_shield_aligned`.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


class RasterSource:
    """Lazy rasterio handle with windowed-read and warped-read helpers.

    One open dataset per instance; handles are reused across calls. Thread-safe
    via a per-instance lock — rasterio datasets aren't themselves thread-safe,
    but read-only access in a lock is fine for our patch enumeration / loading
    patterns.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        name: str | None = None,
    ):
        self._path = path
        self.name = name or str(path)
        self._ds: rasterio.DatasetReader | None = None
        self._lock = threading.Lock()

    # ---------- handle lifecycle ----------

    @property
    def dataset(self) -> rasterio.DatasetReader:
        """Return an open rasterio dataset, opening it on first access."""
        if self._ds is None:
            p = str(self._path)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Raster not found: {p}")
            self._ds = rasterio.open(p)
        return self._ds

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __enter__(self) -> "RasterSource":
        _ = self.dataset
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- geometry properties ----------

    @property
    def width(self) -> int:
        return self.dataset.width

    @property
    def height(self) -> int:
        return self.dataset.height

    @property
    def transform(self):
        return self.dataset.transform

    @property
    def crs(self):
        return self.dataset.crs

    @property
    def nodata(self) -> float | None:
        return self.dataset.nodata

    # ---------- reads ----------

    def read_window(
        self,
        row: int,
        col: int,
        height: int,
        width: int,
        *,
        out_dtype=np.float32,
    ) -> np.ndarray:
        """Read a rectangular window, NaN-filled outside the raster extent.

        In-raster nodata pixels are also converted to NaN so downstream code
        can mask unknown values uniformly with `np.isnan`.
        """
        ds = self.dataset
        read_row = max(0, row)
        read_col = max(0, col)
        read_h = min(row + height, ds.height) - read_row
        read_w = min(col + width, ds.width) - read_col

        out = np.full((height, width), np.nan, dtype=out_dtype)
        if read_h <= 0 or read_w <= 0:
            return out

        with self._lock:
            data = ds.read(
                1,
                window=Window(read_col, read_row, read_w, read_h),
            ).astype(out_dtype, copy=False)
        if ds.nodata is not None:
            data = np.where(data == ds.nodata, np.nan, data)
        paste_r = read_row - row
        paste_c = read_col - col
        out[paste_r : paste_r + read_h, paste_c : paste_c + read_w] = data
        return out


# ---------- module-level cache ----------

_source_cache: dict[str, RasterSource] = {}
_source_cache_lock = threading.Lock()


def open_raster(path: Path | str) -> RasterSource:
    """Open (or reuse) a RasterSource."""
    key = os.path.abspath(str(path))
    with _source_cache_lock:
        if key in _source_cache:
            return _source_cache[key]
        src = RasterSource(path)
        _source_cache[key] = src
        return src


def clear_source_cache() -> None:
    """Close all cached RasterSources (useful for tests + script teardown)."""
    with _source_cache_lock:
        for src in _source_cache.values():
            src.close()
        _source_cache.clear()

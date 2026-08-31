"""
magsr - Multi-modal MSR research project.

All per-machine filesystem paths flow through this module. A `.env` file at
the repo root (gitignored; see `.env.example`) is loaded at import time so
environment variables can be set once per checkout rather than per shell.

Public path handles:
  - `DATA_DIR`            — optional, defaults to `<repo>/data`.
  - `ksa_aligned_root()`  — required for the KSA aligned backend
                            (`ksa_shield_aligned`).

Call-site convention: read `DATA_DIR` as a module attribute; call
`ksa_aligned_root()` lazily inside dataset constructors so a plain
`import magsr` never fails on a machine that doesn't have the KSA data
mounted.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_FOLDER = Path(__file__).resolve().parents[2]

# Load once at import. `override=False` keeps any values already set in the
# real process environment (e.g. from CI or an explicit shell export).
load_dotenv(ROOT_FOLDER / ".env", override=False)


def _clear_leaked_proj_env() -> None:
    """Strip `PROJ_DATA` / `PROJ_LIB` from the process env.

    When the venv Python is launched from a shell with a conda env active
    (e.g. `ml`), that env's `PROJ_DATA` leaks into the process and points
    at a `proj.db` that's the wrong schema version for our wheels —
    symptoms include `DATABASE.LAYOUT.VERSION.MINOR = 5 whereas a number
    >= 6 is expected` on any EPSG lookup or `WarpedVRT` between UTM zones.

    The fix is simply to unset the env vars: both rasterio and pyproj
    ship their own bundled libproj + proj.db and, without `PROJ_DATA`
    set, each falls back to its own compile-time default. Pointing at
    only one of the two bundled dirs would always break the other
    library, since rasterio and pyproj generally link different PROJ
    versions.
    """
    for var in ("PROJ_DATA", "PROJ_LIB"):
        os.environ.pop(var, None)


_clear_leaked_proj_env()


def _env_path(name: str, default: Path | None = None) -> Path:
    val = os.environ.get(name)
    if val:
        return Path(val).expanduser()
    if default is not None:
        return default
    raise RuntimeError(
        f"Environment variable {name} is not set. "
        f"Copy .env.example to .env at the repo root and fill it in, "
        f"or export {name} in your shell."
    )


DATA_DIR: Path = _env_path("MAGSR_DATA_DIR", ROOT_FOLDER / "data")
WA_PATCH_DIR = ROOT_FOLDER / "data" / "processed" / "wa" / "patches"


def ksa_aligned_root() -> Path:
    """Root of the pre-snapped KSA aligned snapshot (single UTM37N grid).

    Required for the `ksa_shield_aligned` backend. Set
    `MAGSR_KSA_ALIGNED_ROOT` in `.env`.
    """
    return _env_path("MAGSR_KSA_ALIGNED_ROOT")

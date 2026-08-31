"""Overview figure of the snapped WA goldfields DEM, for artifact inspection.

Renders the LR-grid DEM (snapped_cubicspline_dtm80m.tif) as elevation plus the
central-difference slope gradients (dz/dx, dz/dy) exactly as the WA dataset
computes them (magsr.datasets.western_australia.dem_gradient), plus |grad|.
The gradient is divided by the 80 m LR pixel so it reads as a dimensionless
slope (m/80m = m/m).
The gradient panels expose snapping/resampling artifacts (seams, striping,
tile boundaries) that a smooth elevation ramp tends to hide.

Output: figures/wa_dem_overview.png

Usage: uv run python scripts/build_wa_dataset/03b_plot_wa_dem_overview.py
"""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from magsr.datasets import WAConfig
from magsr.datasets.western_australia import dem_gradient

cfg = WAConfig.from_yaml()
dem_path = cfg.data_dir / cfg.dem_filename
split_path = cfg.data_dir / cfg.split_filename

if not dem_path.exists():
    raise FileNotFoundError(
        f"Missing {dem_path}. Run scripts/build_wa_dataset/03_fetch_snap_wa_dem.py first."
    )

split_gdf = gpd.read_file(split_path) if split_path.exists() else None


def overlay_aoi(ax):
    """Draw the Train/Test split polygons (same style as 02_plot_wa_overview)."""
    if split_gdf is None:
        return
    colors = {"Test": "green", "Train": "blue"}
    for _, row in split_gdf.iterrows():
        geom = row.geometry
        xs, ys = geom.exterior.coords.xy if hasattr(geom, "exterior") else geom.geoms[0].exterior.coords.xy
        ax.plot(xs, ys, color=colors.get(row["set_type"], "red"), linewidth=2)


with rasterio.open(dem_path) as ds:
    dem = ds.read(1).astype(np.float32)
    if ds.nodata is not None:
        dem = np.where(dem == ds.nodata, np.nan, dem)
    b = ds.bounds
    extent = [b.left, b.right, b.bottom, b.top]
    print(f"DEM {dem_path.name}: shape={dem.shape} crs={ds.crs}")

valid = np.isfinite(dem)
print(
    f"  valid={valid.mean():.1%} elev[min/med/max]="
    f"{np.nanmin(dem):.1f}/{np.nanmedian(dem):.1f}/{np.nanmax(dem):.1f} m"
)

# Same gradient the dataset feeds the model: [dz/dx, dz/dy].
LR_PX_M = 80.0
g = dem_gradient(dem) / LR_PX_M
gx, gy = g[0], g[1]
gmag = np.hypot(gx, gy)

# Symmetric robust limits for the signed gradient panels; same scale for gx/gy.
gclip = float(np.nanpercentile(np.abs(np.concatenate([gx[valid], gy[valid]])), 99.5))
print(f"  slope clip (q99.5 |grad|): {gclip:.4f} (dimensionless, m/80m)")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

panels = [
    (axes[0, 0], dem, "Elevation (m)", "terrain", None, None),
    (axes[0, 1], gmag, "|slope| (m/80m, dimensionless)", "magma", 0.0, gclip),
    (axes[1, 0], gx, "dz/dx (m/80m, dimensionless)", "RdBu_r", -gclip, gclip),
    (axes[1, 1], gy, "dz/dy (m/80m, dimensionless)", "RdBu_r", -gclip, gclip),
]
for ax, arr, title, cmap, vmin, vmax in panels:
    im = ax.imshow(
        arr, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    overlay_aoi(ax)
    ax.set_title(title)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(f"WA Goldfields snapped DEM overview ({dem_path.name})", fontsize=14)
fig.tight_layout()

out_path = cfg.data_dir.parents[1] / "figures" / "wa_dem_overview.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)

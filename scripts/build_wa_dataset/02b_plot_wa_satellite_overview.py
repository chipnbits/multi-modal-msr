"""Satellite-imagery overview of the WA goldfields AOI with Train/Test boxes.

Fetches Esri World Imagery tiles over the study-area bounds and overlays the
Train/Test split polygons, for showing the actual ground surface in the report.

Requires contextily (web tile basemaps). Run transiently with:
    uv run --with contextily python scripts/build_wa_dataset/02b_plot_wa_satellite_overview.py

Output: figures/wa_satellite_overview.png
"""

from __future__ import annotations

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio

from magsr.datasets import WAConfig

# Larger text for report legibility.
plt.rcParams.update(
    {
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.size": 16,
    }
)

cfg = WAConfig.from_yaml()
wa_dir = cfg.data_dir
hr_path = wa_dir / cfg.hr_filename
split_path = wa_dir / cfg.split_filename

# AOI bounds from the HR raster; split polygons for the overlay.
with rasterio.open(hr_path) as hr:
    b = hr.bounds
    src_crs = hr.crs
split_gdf = gpd.read_file(split_path)

# Plot in the native lon/lat CRS (EPSG:7844)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(b.left, b.right)
ax.set_ylim(b.bottom, b.top)

print("Fetching Esri World Imagery tiles ...")
cx.add_basemap(ax, crs=src_crs, source=cx.providers.Esri.WorldImagery, zoom=10)

colors = {"Test": "lime", "Train": "deepskyblue"}
for _, row in split_gdf.iterrows():
    geom = row.geometry
    xs, ys = geom.exterior.coords.xy if hasattr(geom, "exterior") else geom.geoms[0].exterior.coords.xy
    ax.plot(xs, ys, color=colors.get(row["set_type"], "red"), linewidth=2.5, label=row["set_type"])

ax.set_aspect("equal")
ax.legend(loc="upper right")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("WA Goldfields surface (Esri World Imagery) with Train/Test regions")

out_path = wa_dir.parents[1] / "figures" / "wa_satellite_overview.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)

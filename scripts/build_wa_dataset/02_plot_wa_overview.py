"""
Stage 2: Overview plot of the clipped WA aeromagnetic rasters.

Renders the downsampled 20 m HR TMI with the Train/Test split polygons overlaid and
saves figures/wa_study_area_overview.png.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
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
geojson_path = wa_dir / cfg.split_filename

missing = [p.name for p in (hr_path, geojson_path) if not p.exists()]
if missing:
    raise FileNotFoundError(
        f"Missing {missing} in {wa_dir}. Run "
        "scripts/build_wa_dataset/01_clip_wa_goldfields.py first "
        "(see README -> Datasets -> Western Australia)."
    )

split_gdf = gpd.read_file(geojson_path)
print(f"GeoJSON CRS: {split_gdf.crs}")
print(f"Split regions:\n{split_gdf[['fid', 'set_type']].to_string(index=False)}")

# --- Overview plot ---

print("\nGenerating overview plot...")
with rasterio.open(hr_path) as hr_ds:
    downsample = max(1, hr_ds.width // 1000)
    out_shape = (hr_ds.height // downsample, hr_ds.width // downsample)
    data = hr_ds.read(1, out_shape=out_shape)
    data = np.where(data == hr_ds.nodata, np.nan, data)
    b = hr_ds.bounds
    extent = [b.left, b.right, b.bottom, b.top]

# Robust contrast
vmin, vmax = np.nanpercentile(data, [2, 98])
print(f"TMI display range (q2/q98): {vmin:.0f} / {vmax:.0f} nT")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
im = ax.imshow(
    data,
    extent=extent,
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
    aspect="equal",
    interpolation="nearest",
)
plt.colorbar(im, ax=ax, label="TMI (nT)", shrink=0.7)

colors = {"Test": "green", "Train": "blue"}
for _, row in split_gdf.iterrows():
    geom = row.geometry
    xs, ys = geom.exterior.coords.xy if hasattr(geom, "exterior") else geom.geoms[0].exterior.coords.xy
    ax.plot(xs, ys, color=colors.get(row["set_type"], "red"), linewidth=2, label=row["set_type"])

ax.legend(loc="upper right")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("WA Goldfields TMI (20m HR) with Train/Test Regions")

out_path = wa_dir.parents[1] / "figures" / "wa_study_area_overview.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)

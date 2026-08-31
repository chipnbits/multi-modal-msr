"""Two-panel figure of the WA survey footprints and patch tiling.

Shows the spatial distribution of the source surveys used to construct the HR set,
verifying which areas have sufficiently close line spacing (<= 300 m) to carry
high-frequency detail absent from the LR — the selection criterion of Smith et al.
(2022), "Magnetic grid resolution enhancement using machine learning: A case study
from the Eastern Goldfields Superterrane".

(a) Binary survey footprint map (<= 300 m line spacing green, > 300 m red).
(b) The 128x128 HR patch grid overlaid on a downsampled greyscale TMI raster.
"""

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import box as shapely_box
from shapely.validation import make_valid

from magsr import ROOT_FOLDER
from magsr.datasets import WAConfig
from magsr.datasets.western_australia import enumerate_patch_windows

# ── Config ────────────────────────────────────────────────────────────────────
DOWNSAMPLE = 8
cfg = WAConfig.from_yaml()
WA_FOLDER = cfg.data_dir
PATCH_PX = cfg.patch_px

COLORS = {
    "train": "#4488ff",
    "test": "#44ff44",
    "valid_survey": "#2ecc71",
    "valid_survey_edge": "#27ae60",
    "invalid_survey": "#e74c3c",
    "invalid_survey_edge": "#c0392b",
}

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data...")
study_regions = gpd.read_file(WA_FOLDER / "goldfields_split.geojson").to_crs("EPSG:7844")
surveys = gpd.read_file(
    WA_FOLDER / "WA_Mag_Merge_v1_2023_all_surveys" / "WA_Mag_Merge_all_surveys_p.shp"
).to_crs("EPSG:7844")
surveys["geometry"] = surveys["geometry"].apply(make_valid)

# Clip to study extent
sb = study_regions.total_bounds
buf = 0.1
surveys_clip = surveys[
    surveys.intersects(shapely_box(sb[0] - buf, sb[1] - buf, sb[2] + buf, sb[3] + buf))
].copy()

high_res = surveys_clip[surveys_clip["LINE_SPACI"] <= 300]
low_res = surveys_clip[surveys_clip["LINE_SPACI"] > 300]
print(f"Surveys: {len(high_res)} valid, {len(low_res)} invalid")

# Downsampled raster for display
plot_buf = 0.02
view = (sb[0] - plot_buf, sb[1] - plot_buf, sb[2] + plot_buf, sb[3] + plot_buf)

with rasterio.open(WA_FOLDER / "Goldfields_20m_HR.tif") as src:
    win = from_bounds(*view, src.transform)
    shape = (max(1, int(win.height / DOWNSAMPLE)), max(1, int(win.width / DOWNSAMPLE)))
    raster = src.read(1, window=win, out_shape=shape, resampling=Resampling.average)
    raster = np.where(raster == src.nodata, np.nan, raster)

extent = [view[0], view[2], view[1], view[3]]

# ── Patch mosaic: shared with patch_wa_dataset.py via enumerate_patch_windows ───
print("Computing patch mosaic...")
windows = enumerate_patch_windows(
    WA_FOLDER / "Goldfields_20m_HR.tif",
    study_regions,
    cfg,
)
patch_rects: dict[str, list[tuple[float, float, float, float]]] = {"Train": [], "Test": []}
for w in windows:
    set_type = w.source_id.split("/", 1)[1]  # "wa/Train" -> "Train"
    patch_rects[set_type].append((w.left, w.bottom, w.width, w.height))

n_train, n_test = len(patch_rects["Train"]), len(patch_rects["Test"])
print(f"Patches: {n_train} train, {n_test} test")


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_outlines(ax, style="solid"):
    for _, row in study_regions.iterrows():
        gpd.GeoDataFrame([row], crs=study_regions.crs).plot(
            ax=ax,
            facecolor="none",
            linewidth=2.5,
            linestyle=style,
            zorder=5,
            edgecolor=COLORS[row["set_type"].lower()],
        )


def setup_axes(ax, xlim=None, ylim=None):
    ax.set_xlabel("Longitude (\u00b0E)", fontsize=11)
    ax.set_ylabel("Latitude (\u00b0N)", fontsize=11)
    ax.set_xlim(xlim or (view[0], view[2]))
    ax.set_ylim(ylim or (view[1], view[3]))


# ── Figure ────────────────────────────────────────────────────────────────────
print("Generating figure...")
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
vmin, vmax = np.nanpercentile(raster, [2, 98])

# (a) Survey footprints
ax_a.set_facecolor("#1a1a2e")
if len(low_res) > 0:
    low_res.plot(
        ax=ax_a,
        facecolor=COLORS["invalid_survey"],
        edgecolor=COLORS["invalid_survey_edge"],
        alpha=0.5,
        linewidth=0.3,
        zorder=2,
    )
high_res.plot(
    ax=ax_a,
    facecolor=COLORS["valid_survey"],
    edgecolor=COLORS["valid_survey_edge"],
    alpha=0.4,
    linewidth=0.3,
    zorder=3,
)
draw_outlines(ax_a, style="--")
setup_axes(ax_a, xlim=(sb[0] - buf, sb[2] + buf), ylim=(sb[1] - buf, sb[3] + buf))
ax_a.grid(True, alpha=0.15, color="white", linestyle="--")
ax_a.set_title("(a) Survey Footprints by Line Spacing", fontsize=13, fontweight="bold")
ax_a.legend(
    handles=[
        mpatches.Patch(
            fc=COLORS["valid_survey"], ec=COLORS["valid_survey_edge"], alpha=0.5, label="Survey \u2264300 m"
        ),
        mpatches.Patch(
            fc=COLORS["invalid_survey"], ec=COLORS["invalid_survey_edge"], alpha=0.5, label="Survey >300 m"
        ),
        mpatches.Patch(fc="none", ec=COLORS["train"], lw=2, ls="--", label="Train extent"),
        mpatches.Patch(fc="none", ec=COLORS["test"], lw=2, ls="--", label="Test extent"),
    ],
    loc="upper right",
    fontsize=10,
    framealpha=0.9,
    edgecolor="gray",
)

# (b) Patch mosaic
ax_b.imshow(
    raster,
    extent=extent,
    origin="upper",
    cmap="gray",
    norm=Normalize(vmin=vmin, vmax=vmax),
    interpolation="bilinear",
    aspect="equal",
)
for set_type, rects in patch_rects.items():
    color = COLORS[set_type.lower()]
    patches = [Rectangle((x, y), w, h) for x, y, w, h in rects]
    ax_b.add_collection(
        PatchCollection(
            patches,
            facecolor=color,
            edgecolor=color,
            alpha=0.15,
            linewidth=0.3,
        )
    )
draw_outlines(ax_b)
setup_axes(ax_b)
ax_b.set_title(
    f"(b) Patch Mosaic ({PATCH_PX}\u00d7{PATCH_PX} px = {PATCH_PX * 20} m)\n"
    f"Train: {n_train}  \u00b7  Test: {n_test}",
    fontsize=13,
    fontweight="bold",
)
ax_b.legend(
    handles=[
        mpatches.Patch(
            fc=COLORS["train"], alpha=0.3, ec=COLORS["train"], label=f"Train patches ({n_train})"
        ),
        mpatches.Patch(fc=COLORS["test"], alpha=0.3, ec=COLORS["test"], label=f"Test patches ({n_test})"),
        mpatches.Patch(fc="none", ec=COLORS["train"], lw=2.5, label="Train region"),
        mpatches.Patch(fc="none", ec=COLORS["test"], lw=2.5, label="Test region"),
    ],
    loc="upper right",
    fontsize=10,
    framealpha=0.9,
    edgecolor="gray",
)

# Save
fig.suptitle(
    "Valid Training Zones (Survey Line Spacing \u2264 300 m)",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
out_path = ROOT_FOLDER / "figures" / "survey_footprints_and_patches.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()

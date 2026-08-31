"""3-D schematic of draped equivalent-layer upward continuation (KSA).

Renders the four surfaces of the drape-to-drape UC workflow, stacked in true
relative altitude over a real ~200 m-relief KSA DEM tile:

    LR level          z = DEM + 300 m   (upper target drape; continued field)
    HR survey drape   z = DEM +  60 m   (observed RTP anomaly, draped on terrain)
    ground (DEM)      z = DEM           (bare terrain, grey)
    equivalent layer  z = min(DEM) - 60 (flat sheet of fitted magnetisation sigma)

The HR anomaly is inverted (Tikhonov-regularised equivalent-layer fit) to a
buried magnetisation on the flat layer, then continued forward to the +300 m
drape — the LR surface shows the resulting smoothed/attenuated field on the
SAME colour scale as HR, so the low-pass action of continuation is visible.

Run (main-repo env; pyvista pulled ephemerally):

    uv run --with pyvista python experiments/plots/schematic_drape_uc_3d.py

Off-screen render -> PNG (VTK OSMesa). Tile / geometry are CLI-tunable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from magsr import ROOT_FOLDER
from magsr.datasets.ksa_shield_aligned import KSAAlignedConfig
from magsr.fourier import fit_equivalent_layer, upward_continue
from magsr.fourier._fft_utils import crop_centered

_KSA = KSAAlignedConfig.default()
DTM_TIF = _KSA.hr_dir / "snapped_cubicspline_DTM.tif"
MAG_TIF = _KSA.hr_normalized_product_path("AMF_RTP")


def load_tile(row: int, col: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dem, mag)`` float32 ``(n, n)`` windows on the 60 m grid."""
    with rasterio.open(DTM_TIF) as d, rasterio.open(MAG_TIF) as m:
        dem = d.read(1, window=Window(col, row, n, n)).astype("float32")
        mag = m.read(1, window=Window(col, row, n, n)).astype("float32")
    return dem, mag


def fit_and_continue(dem, mag, clr_hr, clr_lr, layer_below, eps, cg_iters):
    """EL fit on the HR drape -> sigma + field continued to the LR drape.

    Returns ``(sigma, lr_field)`` each ``(n, n)`` float32, in the tile grid.
    Uses the code's z-down convention (above-ground is negative): the sensor
    elevation ``DEM + clearance`` maps to ``z = -(DEM + clearance)``.
    """
    n = dem.shape[0]
    dem_t = torch.from_numpy(dem)
    mag_t = torch.from_numpy(mag)
    z_obs = -(clr_hr + dem_t)  # (n, n) z-down HR survey drape
    z_layer = float(layer_below - dem.min())  # flat layer, 60 m below lowest terrain
    layer = fit_equivalent_layer(
        mag_t,
        dx=60.0,
        dy=60.0,
        z_obs=z_obs,
        z_layer=z_layer,
        eps=eps,
        cg_iters=cg_iters,
    )
    z_lr = -(clr_lr + dem_t)  # LR target drape (z-down)
    lr_field = upward_continue(layer, z_target=z_lr, n_layers=16).numpy()

    pad = layer.k.shape[0]
    sigma = crop_centered(torch.fft.irfft2(layer.sigma_hat, s=(pad, pad)), n, n).numpy()
    return sigma, lr_field


def structured(x, y, z):
    """Build a pyvista StructuredGrid from 2-D x, y, z arrays."""
    import pyvista as pv

    grid = pv.StructuredGrid()
    grid.points = np.c_[x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")]
    grid.dimensions = (x.shape[1], x.shape[0], 1)
    return grid


def main(a: argparse.Namespace) -> None:
    import pyvista as pv

    pv.OFF_SCREEN = True

    dem, mag = load_tile(a.row, a.col, a.n)
    sigma, lr_field = fit_and_continue(dem, mag, a.clr_hr, a.clr_lr, a.layer_below, a.eps, a.cg_iters)
    n = a.n
    relief = float(dem.max() - dem.min())
    band = (a.clr_lr - a.clr_hr) - relief  # clear gap between HR and LR z-ranges
    print(f"relief={relief:.0f} m   HR-LR clear band={band:.0f} m (must be > 0)")

    # Horizontal grid in metres, north up (flip rows). z = elevation, with the
    # vertical exaggeration baked straight into z so meshes AND labels share it.
    xs = np.arange(n) * 60.0
    ys = np.arange(n) * 60.0
    X, Y = np.meshgrid(xs, ys)
    Y = Y.max() - Y  # north up
    z0 = dem.min() - a.layer_below  # scene datum = the equivalent-layer plane
    zs = a.zscale

    z_hr = (dem + a.clr_hr - z0) * zs
    z_lr = (dem + a.clr_lr - z0) * zs
    z_layer = np.zeros_like(dem)  # flat sheet at the datum

    # HR and LR share one nT scale so UC attenuation shows; sigma its own scale.
    vmag = float(np.percentile(np.abs(mag), 98))
    vsig = float(np.percentile(np.abs(sigma), 98))

    g_layer = structured(X, Y, z_layer)
    g_layer["sigma"] = sigma.ravel(order="F")
    g_hr = structured(X, Y, z_hr)
    g_hr["mag"] = mag.ravel(order="F")
    g_lr = structured(X, Y, z_lr)
    g_lr["field"] = lr_field.ravel(order="F")

    p = pv.Plotter(off_screen=True, window_size=(a.width, a.height))
    p.set_background("white")
    p.enable_anti_aliasing("msaa")
    p.enable_depth_peeling(12)  # correct ordering for the translucent slice stack

    # Three surfaces exactly as described: EL (below) -> HR drape -> LR level.
    # Moderate ambient keeps the scalar colours true while the drape still shades.
    lit = dict(smooth_shading=True, ambient=0.45, diffuse=0.6)
    # Dual horizontal scalar bar along the bottom (no in-scene labels).
    sb_mag = dict(
        title="dT  (nT)",
        vertical=False,
        position_x=0.07,
        position_y=0.05,
        width=0.40,
        height=0.05,
        title_font_size=52,
        label_font_size=42,
        color="black",
        n_labels=3,
    )
    sb_sig = dict(
        title="sigma",
        vertical=False,
        position_x=0.55,
        position_y=0.05,
        width=0.40,
        height=0.05,
        title_font_size=52,
        label_font_size=42,
        color="black",
        n_labels=3,
    )
    p.add_mesh(g_layer, scalars="sigma", cmap="PuOr", clim=(-vsig, vsig), scalar_bar_args=sb_sig, **lit)
    p.add_mesh(g_hr, scalars="mag", cmap="RdBu_r", clim=(-vmag, vmag), scalar_bar_args=sb_mag, **lit)
    p.add_mesh(g_lr, scalars="field", cmap="RdBu_r", clim=(-vmag, vmag), show_scalar_bar=False, **lit)

    # Interpolation indicator: the flat chessboard levels spanning the LR relief
    # that are per-pixel blended to form the draped product. Seen near edge-on at
    # this oblique angle, they read as a thin stacked "layer cake" at the LR band.
    xc, yc = float(X.mean()), float(Y.mean())
    isz, jsz = float(X.max() - X.min()), float(Y.max() - Y.min())
    for zeta in np.linspace(float(z_lr.min()), float(z_lr.max()), a.n_slices):
        quad = pv.Plane(
            center=(xc, yc, float(zeta)),
            direction=(0, 0, 1),
            i_size=isz,
            j_size=jsz,
            i_resolution=a.slice_grid,
            j_resolution=a.slice_grid,
        )
        # Faint translucent fill, but draw the wire grid as a separate opaque
        # actor so the lines stay crisp (per-actor opacity dims edges too).
        p.add_mesh(quad, color="#9a9a9a", opacity=0.05, show_edges=False, lighting=False)
        p.add_mesh(quad, style="wireframe", color="#333333", line_width=2.4, opacity=0.3, lighting=False)

    p.camera_position = "yz"
    p.camera.azimuth = a.azimuth
    p.camera.elevation = a.elevation
    p.reset_camera()
    p.camera.zoom(a.zoom)

    out = a.out or (ROOT_FOLDER / "figures" / "drape_uc_3d" / f"drape_uc_3d_r{a.row}_c{a.col}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out))

    # Trim the white border so the figure sits tight in LaTeX.
    try:
        from PIL import Image, ImageChops

        im = Image.open(out).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bbox = ImageChops.difference(im, bg).getbbox()
        if bbox:
            pad = 12
            l, t, r, b = bbox
            im.crop(
                (max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height))
            ).save(out)
    except Exception as e:  # pragma: no cover - cosmetic only
        print(f"(trim skipped: {e})")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", type=int, default=9744)
    ap.add_argument("--col", type=int, default=7680)
    ap.add_argument("--n", type=int, default=132)
    ap.add_argument("--clr-hr", type=float, default=60.0)
    ap.add_argument("--clr-lr", type=float, default=300.0)
    ap.add_argument("--layer-below", type=float, default=60.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--cg-iters", type=int, default=12)
    ap.add_argument(
        "--n-slices",
        type=int,
        default=6,
        help="Flat chessboard levels drawn at the LR band as the interp indicator.",
    )
    ap.add_argument(
        "--slice-grid",
        type=int,
        default=3,
        help="Wire-grid resolution (cells/side) on each interpolation slice.",
    )
    ap.add_argument("--zscale", type=float, default=7.0)
    ap.add_argument("--azimuth", type=float, default=-35.0)
    ap.add_argument("--elevation", type=float, default=18.0)
    ap.add_argument("--zoom", type=float, default=1.3)
    ap.add_argument("--width", type=int, default=1900)
    ap.add_argument("--height", type=int, default=1450)
    ap.add_argument("--out", type=Path, default=None)
    main(ap.parse_args())

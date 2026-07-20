"""
landscape_render.py

Polished 3D renders of the reweighted 2D FES using basic matplotlib.
- Interpolates the coarse histogram onto a fine grid (smooth, not blocky)
- Caps the unsampled 'roof' at a sane ceiling
- Forces invert=True so energy basins become landscape peaks
- Completely strips the data box framing, grids, and axes for an organic look
- Container-friendly: Zero extra installation steps needed.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless backend safe for Docker environments
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import config
from enhanced_fes2d import read_colvar, reweight, fes_2d


def _fine_grid(xc, yc, F, ceiling, upsample):
    """Upsample the (masked) FES onto a fine grid for a smooth shape."""
    sampled = np.isfinite(F)
    Ffill = np.where(sampled, np.minimum(F, ceiling), ceiling)
    fint = RegularGridInterpolator((yc, xc), Ffill, method="linear", bounds_error=False, fill_value=ceiling)
    mint = RegularGridInterpolator((yc, xc), sampled.astype(float), method="linear", bounds_error=False, fill_value=0.0)
    xf = np.linspace(xc.min(), xc.max(), len(xc) * upsample)
    yf = np.linspace(yc.min(), yc.max(), len(yc) * upsample)
    XX, YY = np.meshgrid(xf, yf)
    Z = np.clip(fint((YY, XX)), 0.0, ceiling)
    Z = np.where(mint((YY, XX)) >= 0.5, Z, np.nan)  # Clean gaps outside sampled region
    return XX, YY, Z


def landscape_mpl(xc, yc, F, out_png, elev=20, azim=-55, ceiling=25.0, upsample=8, invert=True):
    """
    Renders a clean, floating organic topological landscape using basic Matplotlib.
    Strips away data boxes, ticks, panel borders, and edge artifacts.
    """
    # 1. High-resolution grid upsampling to completely remove the blocky square look
    XX, YY, Z = _fine_grid(xc, yc, F, ceiling, upsample)

    # 3. Invert basins into peaks to match the presentation layout perspective
    Zplot = (ceiling - Z) if invert else Z

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize continuous array heights for a smooth color transition gradient
    norm = plt.Normalize(vmin=np.nanmin(Zplot), vmax=np.nanmax(Zplot))
    colors = plt.cm.turbo(norm(Zplot))

    # 2 & 4. Plot surface with ZERO linewidth edge boundaries and smooth shading
    ax.plot_surface(
        XX, YY, Zplot, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True, alpha=0.9
    )

    # 1. Drop the data frame boxes, axis grids, text ticks, and colorbars entirely
    ax.axis("off")
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    # Force transparency on internal default background pane plates
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # 5. Low panoramic viewpoint perspective mimicking a mountain range horizon
    ax.view_init(elev=elev, azim=azim)

    # Set proportional dimensional layout aspect scaling
    ax.set_box_aspect((1, 1, 0.45))

    # Export a presentation-ready high-DPI image with a transparent background canvas
    fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"✓ Saved polished organic landscape layout to {out_png.name}")


def run(cv_x="d_active", cv_y="wat", ceiling=25.0):
    out = config.OUTPUT_DIR
    cv = read_colvar(out / "COLVAR")

    for col in (cv_x, cv_y, "opes.bias"):
        if col not in cv.columns:
            raise KeyError(f"'{col}' not in COLVAR columns {list(cv.columns)}.")

    w = reweight(cv)
    x = cv[cv_x].to_numpy() * 10.0  # nm -> Angstrom
    y = cv[cv_y].to_numpy()
    xc, yc, F, _ = fes_2d(x, y, w)

    out_png = out / f"{config.PROTEIN_NAME}_topological_landscape.png"

    # Run the optimized layout setup directly
    landscape_mpl(xc, yc, F, out_png, ceiling=ceiling, upsample=8, invert=True)


if __name__ == "__main__":
    run()

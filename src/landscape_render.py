"""
landscape_render.py

Nicer 3D renders of the reweighted 2D FES than the default matplotlib surface:
- interpolates the coarse histogram onto a fine grid (smooth, not blocky),
- caps the unsampled 'roof' at a sane ceiling instead of the max,
- adds hillshade lighting, and lets you set the view angle or invert
  (basins -> peaks), and
- optionally writes an interactive Plotly HTML you can drag to any angle.

CPU-only, reads COLVAR alone. For interactivity, run it locally after pulling
the COLVAR from the pod, then open the .html.

    python landscape_render.py                # both renders, default view
    # in Python: landscape_mpl(..., invert=True) or view angles as you like
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.interpolate import RegularGridInterpolator

import config
from enhanced_fes2d import read_colvar, reweight, fes_2d


def _fine_grid(xc, yc, F, ceiling, upsample):
    """Upsample the (masked) FES onto a fine grid.

    Linear interpolation (no cubic overshoot -> no rim spikes). Cells outside
    the sampled region are returned as NaN so the surface has clean gaps rather
    than a flat 'roof' filled at the ceiling.
    """
    sampled = np.isfinite(F)
    Ffill = np.where(sampled, np.minimum(F, ceiling), ceiling)
    fint = RegularGridInterpolator((yc, xc), Ffill, method="linear",
                                   bounds_error=False, fill_value=ceiling)
    mint = RegularGridInterpolator((yc, xc), sampled.astype(float), method="linear",
                                   bounds_error=False, fill_value=0.0)
    xf = np.linspace(xc.min(), xc.max(), len(xc) * upsample)
    yf = np.linspace(yc.min(), yc.max(), len(yc) * upsample)
    XX, YY = np.meshgrid(xf, yf)
    Z = np.clip(fint((YY, XX)), 0.0, ceiling)
    Z = np.where(mint((YY, XX)) >= 0.5, Z, np.nan)   # gaps outside sampled region
    return XX, YY, Z


def landscape_mpl(xc, yc, F, out_png, xlabel, ylabel, elev=35, azim=-60,
                  ceiling=25.0, upsample=5, invert=False):
    """Smooth, hillshaded matplotlib surface. invert=True -> basins become peaks."""
    XX, YY, Z = _fine_grid(xc, yc, F, ceiling, upsample)
    Zplot = (ceiling - Z) if invert else Z
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Zplot, cmap=plt.cm.turbo, vert_exag=1.0, blend_mode="soft")

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, Zplot, facecolors=rgb, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_zlabel(("-" if invert else "") + "Free energy (kJ/mol)", labelpad=8)
    ax.set_title(f"OPES free-energy landscape — {config.PROTEIN_NAME}")
    ax.view_init(elev=elev, azim=azim)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def landscape_plotly(xc, yc, F, out_html, xlabel, ylabel, ceiling=25.0, upsample=5):
    """Interactive WebGL surface (drag to any angle). Requires plotly."""
    import plotly.graph_objects as go
    XX, YY, Z = _fine_grid(xc, yc, F, ceiling, upsample)
    fig = go.Figure(go.Surface(
        x=XX[0], y=YY[:, 0], z=Z, colorscale="Turbo",
        colorbar=dict(title="F (kJ/mol)"),
        lighting=dict(ambient=0.5, diffuse=0.85, roughness=0.4, specular=0.15),
        contours={"z": {"show": True, "usecolormap": True, "project_z": True,
                        "width": 1}},
    ))
    fig.update_layout(
        title=f"OPES free-energy landscape — {config.PROTEIN_NAME}",
        scene=dict(xaxis_title=xlabel, yaxis_title=ylabel,
                   zaxis_title="Free energy (kJ/mol)"),
        width=1000, height=800,
    )
    fig.write_html(out_html)


def run(cv_x="d_active", cv_y="wat", ceiling=25.0):
    out = config.OUTPUT_DIR
    cv = read_colvar(out / "COLVAR")
    for col in (cv_x, cv_y, "opes.bias"):
        if col not in cv.columns:
            raise KeyError(f"'{col}' not in COLVAR columns {list(cv.columns)}.")
    w = reweight(cv)
    x = cv[cv_x].to_numpy() * 10.0            # nm -> Angstrom
    y = cv[cv_y].to_numpy()
    xc, yc, F, _ = fes_2d(x, y, w)
    xlabel = f"{cv_x}  Glu35-Asp52 (Å)"
    ylabel = f"{cv_y}  water coordination"

    landscape_mpl(xc, yc, F, out / f"{config.PROTEIN_NAME}_landscape_smooth.png",
                  xlabel, ylabel, ceiling=ceiling)
    print(f"wrote {config.PROTEIN_NAME}_landscape_smooth.png")
    try:
        landscape_plotly(xc, yc, F, out / f"{config.PROTEIN_NAME}_landscape.html",
                         xlabel, ylabel, ceiling=ceiling)
        print(f"wrote {config.PROTEIN_NAME}_landscape.html  (open locally, drag to rotate)")
    except ImportError:
        print("plotly not installed -> run `pixi add plotly` for the interactive HTML")


if __name__ == "__main__":
    run()

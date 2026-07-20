"""
landscape_render.py

Smooth 3D renders of the reweighted 2D FES.

Why the default histogram FES looks jagged: it bins a finite number of samples,
and F = -kT ln P then amplifies the counting noise in low-population bins (the
rim especially) into spikes. A smooth analytic surface like MATLAB's `peaks`
never has this because it's an equation, not sampled data.

The fix here is to estimate the density *smoothly* with a weighted Gaussian KDE
of the reweighted samples, instead of a hard histogram. That is standard, honest
practice (still your data, just a smooth density estimate) and gives the smooth,
`peaks`-like surface. `bw` is the smoothness knob: larger = smoother.

CPU-only, reads COLVAR alone. Optional interactive Plotly HTML (needs plotly).
    python landscape_render.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import config
from enhanced_fes2d import read_colvar, reweight, KT


def kde_fes(x, y, weights, grid=150, bw=0.20, ceiling=22.0, pad=0.05):
    """Smooth reweighted FES via weighted Gaussian KDE.

    Returns the meshgrid GX, GY and the clipped free energy F (kJ/mol, min 0).
    """
    kde = gaussian_kde(np.vstack([x, y]), weights=weights, bw_method=bw)
    xr = (x.max() - x.min()) * pad
    yr = (y.max() - y.min()) * pad
    gx = np.linspace(x.min() - xr, x.max() + xr, grid)
    gy = np.linspace(y.min() - yr, y.max() + yr, grid)
    GX, GY = np.meshgrid(gx, gy)
    P = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    F = -KT * np.log(P + 1e-300)
    F -= F.min()
    return GX, GY, np.clip(F, 0.0, ceiling)


def _slice(F, show_max, invert):
    """Keep only F <= show_max (a smooth KDE isocontour), as plot heights."""
    Fm = np.where(F <= show_max, F, np.nan)
    return (show_max - Fm) if invert else Fm


def landscape_mpl(GX, GY, F, out_png, elev=32, azim=-58, show_max=18.0, invert=True):
    """Smooth surface, axes stripped, floating look. invert=True -> basins as peaks.

    show_max slices the surface at that free-energy contour (kJ/mol) so the low-
    density rim is cut along a smooth line, not the noisy fringe.
    """
    Z = _slice(F, show_max, invert)
    norm = plt.Normalize(np.nanmin(Z), np.nanmax(Z))
    colors = plt.cm.turbo(norm(np.nan_to_num(Z, nan=np.nanmin(Z))))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(GX, GY, Z, facecolors=colors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=True)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.5))
    fig.patch.set_alpha(0.0)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def landscape_plotly(GX, GY, F, out_html, xlabel, ylabel, show_max=18.0, invert=False):
    """Interactive WebGL surface (drag to any angle). Requires plotly."""
    import plotly.graph_objects as go
    Z = _slice(F, show_max, invert)
    fig = go.Figure(go.Surface(
        x=GX[0], y=GY[:, 0], z=Z, colorscale="Turbo",
        colorbar=dict(title="F (kJ/mol)"),
        lighting=dict(ambient=0.5, diffuse=0.85, roughness=0.4, specular=0.15),
    ))
    fig.update_layout(
        title=f"OPES free-energy landscape — {config.PROTEIN_NAME}",
        scene=dict(xaxis_title=xlabel, yaxis_title=ylabel,
                   zaxis_title="Free energy (kJ/mol)"),
        width=1000, height=800,
    )
    fig.write_html(out_html)


def run(cv_x="d_active", cv_y="wat", show_max=18.0, bw=0.20):
    out = config.OUTPUT_DIR
    cv = read_colvar(out / "COLVAR")
    for col in (cv_x, cv_y, "opes.bias"):
        if col not in cv.columns:
            raise KeyError(f"'{col}' not in COLVAR columns {list(cv.columns)}. "
                           f"This needs the 2D run (a COLVAR with a '{cv_y}' column).")
    w = reweight(cv)
    x = cv[cv_x].to_numpy() * 10.0            # nm -> Angstrom
    y = cv[cv_y].to_numpy()
    GX, GY, F = kde_fes(x, y, w, ceiling=show_max + 6.0, bw=bw)
    xlabel = f"{cv_x}  Glu35-Asp52 (Å)"
    ylabel = f"{cv_y}  water coordination"

    landscape_mpl(GX, GY, F, out / f"{config.PROTEIN_NAME}_landscape_smooth.png",
                  show_max=show_max, invert=True)
    print(f"wrote {config.PROTEIN_NAME}_landscape_smooth.png  (bw={bw}, show_max={show_max})")
    try:
        landscape_plotly(GX, GY, F, out / f"{config.PROTEIN_NAME}_landscape.html",
                         xlabel, ylabel, show_max=show_max)
        print(f"wrote {config.PROTEIN_NAME}_landscape.html  (open locally, drag to rotate)")
    except ImportError:
        print("plotly not installed -> run `pixi add plotly` for the interactive HTML")


if __name__ == "__main__":
    run()

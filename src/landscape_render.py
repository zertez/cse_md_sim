"""
landscape_topology_render.py

"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import config
from enhanced_fes2d import read_colvar, reweight, KT


def kde_fes(x, y, weights, grid=150, bw=0.20, ceiling=30.0, pad=0.05):
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


def landscape_mpl(GX, GY, F, out_png, xlabel, ylabel, elev=30, azim=-60, show_max=22.0, invert=False):
    """Smooth free-energy landscape.

    invert=False -> downward funnel
    invert=True  -> basin flipped up into a peak.
    """
    Fm = np.where(F <= show_max, F, np.nan)
    Z = (show_max - Fm) if invert else Fm
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        GX, GY, Z, cmap="turbo", vmin=zmin, vmax=zmax, rstride=1, cstride=1, linewidth=0, antialiased=True
    )
    # floor contour projection, like the reference figure
    ax.contour(GX, GY, Z, zdir="z", offset=zmin - 0.18 * (zmax - zmin), levels=14, cmap="turbo", alpha=0.6)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_zlabel("Free energy (kJ/mol)", labelpad=8)
    ax.set_title(f"OPES free-energy landscape — {config.PROTEIN_NAME}")
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.5, aspect=12, pad=0.10, label="Free energy (kJ/mol)")
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def run(cv_x="d_active", cv_y="wat", show_max=22.0, bw=0.20):
    out = config.OUTPUT_DIR
    cv = read_colvar(out / "COLVAR")
    for col in (cv_x, cv_y, "opes.bias"):
        if col not in cv.columns:
            raise KeyError(
                f"'{col}' not in COLVAR columns {list(cv.columns)}. "
                f"This needs the 2D run (a COLVAR with a '{cv_y}' column)."
            )
    w = reweight(cv)
    x = cv[cv_x].to_numpy() * 10.0  # nm -> Angstrom
    y = cv[cv_y].to_numpy()
    GX, GY, F = kde_fes(x, y, w, ceiling=show_max + 8.0, bw=bw)
    xlabel = f"{cv_x}  Glu35-Asp52 (Å)"
    ylabel = f"{cv_y}  water coordination"

    landscape_mpl(
        GX, GY, F, out / f"{config.PROTEIN_NAME}_landscape_smooth.png", xlabel, ylabel, show_max=show_max, invert=False
    )
    print(f"wrote {config.PROTEIN_NAME}_landscape_smooth.png  (bw={bw}, show_max={show_max})")


if __name__ == "__main__":
    run()

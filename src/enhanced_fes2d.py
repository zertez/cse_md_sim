"""
enhanced_fes2d.py

2D free-energy landscape from a 2D-OPES run, by reweighting COLVAR.

Reweighting weight per frame is w_i = exp(V_i / kT) with V_i = opes.bias; the
unbiased density P(s) is the weighted histogram and F(s) = -kT ln P(s). Both
CVs come from COLVAR (printed every 500 steps by plumed_water2d.dat),
so there is no trajectory alignment and no sparse-sampling penalty.

Run this after enhanced_sampling.py on the COLVAR from the 2D setup.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import config

# kT in kJ/mol (config.TEMPERATURE is an OpenMM Quantity; config re-exports unit).
KT = 0.0083144621 * config.TEMPERATURE.value_in_unit(config.unit.kelvin)


def read_colvar(path):
    """Parse a PLUMED COLVAR, taking column names from the '#! FIELDS' line."""
    with open(path) as f:
        # drop '#!' and 'FIELDS'
        names = f.readline().split()[2:]
    return pd.read_csv(path, sep=r"\s+", comment="#", names=names)


def reweight(cv):
    """OPES frame weights: w_i = exp(V_i / kT), max-shifted for stability."""
    bias = cv["opes.bias"].to_numpy()
    w = np.exp((bias - bias.max()) / KT)
    return w / w.sum()


def fes_1d(values, weights, bins=60):
    hist, edges = np.histogram(values, bins=bins, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fes = -KT * np.log(hist + 1e-12)
    return centers, fes - fes.min()


def fes_2d(x, y, weights, bins=40, min_count=5, smooth=1.0):
    """Reweighted 2D FES. Bins with too few raw samples are masked (NaN) so the
    landscape has no fake walls where nothing was sampled."""
    counts, xe, ye = np.histogram2d(x, y, bins=bins)
    Hw, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=weights, density=True)
    F = -KT * np.log(Hw.T + 1e-12)
    F -= np.nanmin(F)
    F[counts.T < min_count] = np.nan
    F_smooth = gaussian_filter(np.nan_to_num(F, nan=np.nanmax(F)), sigma=smooth)
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    return xc, yc, F, F_smooth


def run(cv_x="d_active", cv_y="wat"):
    out = config.OUTPUT_DIR
    colvar = out / "COLVAR"
    if not colvar.exists():
        print("COLVAR not found. Run enhanced_sampling.py first.")
        return

    cv = read_colvar(colvar)
    for col in (cv_x, cv_y, "opes.bias"):
        if col not in cv.columns:
            raise KeyError(
                f"'{col}' not in COLVAR columns {list(cv.columns)}. This expects the 2D run (plumed_water2d.dat)."
            )

    w = reweight(cv)
    neff = 1.0 / np.sum(w**2)
    # nm -> Angstrom
    x = cv[cv_x].to_numpy() * 10.0
    y = cv[cv_y].to_numpy()
    print(f"{len(cv)} frames; effective samples after reweighting: {neff:.0f}")
    if neff < 200:
        print("  WARNING: low effective sample size -> a 2D FES will be noisy. Run longer before trusting it.")

    xc, yc, F, F_smooth = fes_2d(x, y, w)
    XX, YY = np.meshgrid(xc, yc)
    xlabel = f"{cv_x}  Glu35-Asp52 (Å)"
    ylabel = f"{cv_y}  water coordination"

    # --- 3D landscape (the 'topology' view) ---
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(XX, YY, F_smooth, cmap="turbo", edgecolor="k", lw=0.15, rstride=1, cstride=1, antialiased=True)
    ax.contour(XX, YY, F_smooth, zdir="z", offset=0, levels=14, cmap="turbo", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Free energy (kJ/mol)")
    ax.set_title(f"OPES free-energy landscape — {config.PROTEIN_NAME}")
    ax.view_init(elev=38, azim=-60)
    fig.savefig(out / f"{config.PROTEIN_NAME}_fes_landscape.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 2D contour (publication-honest, à la the paper's Fig 5) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    pc = ax.contourf(xc, yc, F, levels=25, cmap="turbo")
    fig.colorbar(pc, label="Free energy (kJ/mol)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"2D FES — {config.PROTEIN_NAME}")
    fig.savefig(out / f"{config.PROTEIN_NAME}_fes_2d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- 1D reference profile along the catalytic distance ---
    cx, f1 = fes_1d(x, w)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cx, f1, lw=2.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Free energy (kJ/mol)")
    ax.grid(True)
    ax.set_title(f"1D FES along {cv_x} — {config.PROTEIN_NAME}")
    fig.savefig(out / f"{config.PROTEIN_NAME}_fes_1d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved landscape / 2D contour / 1D profile to {out}")


if __name__ == "__main__":
    run()

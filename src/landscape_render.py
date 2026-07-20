import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator

import config
from enhanced_fes2d import read_colvar, reweight, fes_2d


def _fine_grid(xc, yc, F, ceiling, upsample):
    """Upsample the coarse FES onto a fine grid for a smooth, organic mesh."""
    sampled = np.isfinite(F)
    Ffill = np.where(sampled, np.minimum(F, ceiling), ceiling)
    fint = RegularGridInterpolator((yc, xc), Ffill, method="linear", bounds_error=False, fill_value=ceiling)
    mint = RegularGridInterpolator((yc, xc), sampled.astype(float), method="linear", bounds_error=False, fill_value=0.0)

    xf = np.linspace(xc.min(), xc.max(), len(xc) * upsample)
    yf = np.linspace(yc.min(), yc.max(), len(yc) * upsample)
    XX, YY = np.meshgrid(xf, yf)
    Z = np.clip(fint((YY, XX)), 0.0, ceiling)
    Z = np.where(mint((YY, XX)) >= 0.5, Z, np.nan)
    return XX, YY, Z


def landscape_pyvista(xc, yc, F, out_png, ceiling=25.0, upsample=8, invert=True):
    """
    Renders an organic, smooth 3D free energy landscape using PyVista's VTK mesh engine.
    Strips away all grid lines, scientific bounding boxes, and applies cinematic lighting.
    """
    XX, YY, Z = _fine_grid(xc, yc, F, ceiling, upsample)

    # 3. Invert the surface so the energy basins look like the screenshot's mountains
    Zplot = (ceiling - Z) if invert else Z

    # Replace NaNs with a dummy value for the structured mesh geometry framework,
    # then we strip them using a threshold to get clean, organic edges.
    Z_fixed = np.where(np.isnan(Zplot), -999, Zplot)

    # Create a true 3D Structured Mesh
    grid = pv.StructuredGrid(XX, YY, Z_fixed)
    grid = grid.threshold(0.0, scalars=grid.active_scalars_name)  # Drops the unvisited "roof" completely

    # Initialize a headless high-res plotter instance
    plotter = pv.Plotter(off_screen=True, lighting="light_kit")

    # Set background completely transparent (or choose 'white')
    plotter.background_color = "white"

    # Add the mesh to the scene with cinematic settings
    plotter.add_mesh(
        grid,
        cmap="turbo",
        smooth_shading=True,  # 2. Employs Gouraud/Phong shading to remove grid lines entirely
        ambient=0.4,  # Fills harsh shadows with soft environmental light
        diffuse=0.8,  # Natural matte light scattering across peaks
        specular=0.1,  # Subtle highlights on the ridges
        show_scalar_bar=False,  # 1. Strips away the ugly scientific colorbar scale
    )

    # 5. Position the camera to match the low, sweeping view angle of the screenshot
    # PyVista camera position: [eye coordinates, focus center, up-vector]
    plotter.camera_position = [
        (xc.mean() + 15, yc.mean() - 15, ceiling * 1.8),  # Camera location
        (xc.mean(), yc.mean(), ceiling * 0.3),  # Look-at point
        (0.0, 0.0, 1.0),  # Up orientation
    ]

    # Save a clean, razor-sharp presentation image
    plotter.screenshot(out_png, window_size=[1200, 900], scale=3)
    plotter.close()
    print(f"✓ Saved organic PyVista landscape to {out_png.name}")


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

    out_png = out / f"{config.PROTEIN_NAME}_pyvista_landscape.png"

    # Run the cinematic mesh generation pipeline
    landscape_pyvista(xc, yc, F, out_png, ceiling=ceiling, upsample=8, invert=True)


if __name__ == "__main__":
    run()

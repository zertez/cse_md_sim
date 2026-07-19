"""
enhanced_analysis.py

Analysis module specifically for enhanced sampling trajectories (OPES).
Reusable and focused on collective variables + reweighting.
"""

import mdtraj as md
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config


def run_enhanced_analysis():
    output_dir = config.OUTPUT_DIR
    traj_file = output_dir / "enhanced.dcd"
    top_file = output_dir / "equilibrated.pdb"
    state_csv = output_dir / "enhanced_state.csv"
    colvar_file = output_dir / "COLVAR"  # from PLUMED

    print(f"Running enhanced sampling analysis for: {config.PROTEIN_NAME}")

    # Load trajectory
    if traj_file.exists():
        traj = md.load(str(traj_file), top=str(top_file))
        print(f"Enhanced Trajectory: {traj.n_frames} frames")
    else:
        print("Enhanced trajectory not found. Running basic analysis on production instead.")
        traj_file = output_dir / "production.dcd"
        traj = md.load(str(traj_file), top=str(top_file))

    # RMSD
    rmsd = md.rmsd(traj, traj[0], atom_indices=traj.topology.select("backbone"))
    print(f"Mean RMSD: {np.mean(rmsd) * 10:.2f} Å")

    # Radius of gyration
    rg = md.compute_rg(traj)

    # Load state data
    df = pd.read_csv(state_csv) if state_csv.exists() else None

    # Try to load PLUMED COLVAR if available
    if colvar_file.exists():
        try:
            colvar = pd.read_csv(colvar_file, delim_whitespace=True, comment="#")
            print(f"Loaded PLUMED COLVAR with {len(colvar)} entries")
        except:
            colvar = None
    else:
        colvar = None

    # === Plots ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(rmsd * 10)
    axes[0, 0].set_title(f"RMSD - Enhanced {config.PROTEIN_NAME}")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("RMSD (Å)")
    axes[0, 0].grid(True)

    axes[0, 1].plot(rg * 10)
    axes[0, 1].set_title("Radius of Gyration")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Rg (Å)")
    axes[0, 1].grid(True)

    if df is not None and "Temperature (K)" in df.columns:
        axes[1, 0].plot(df["Temperature (K)"])
        axes[1, 0].axhline(300, color="red", linestyle="--")
        axes[1, 0].set_title("Temperature")
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_ylabel("K")
        axes[1, 0].grid(True)

    # Plot CV from COLVAR if available
    if colvar is not None and len(colvar.columns) > 1:
        cv_col = colvar.columns[1]  # usually the first CV
        axes[1, 1].plot(colvar[cv_col])
        axes[1, 1].set_title(f"Collective Variable: {cv_col}")
        axes[1, 1].set_xlabel("Frame")
        axes[1, 1].set_ylabel(cv_col)
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, "No COLVAR CV found", ha="center")

    plt.tight_layout()
    plot_file = output_dir / f"{config.PROTEIN_NAME}_enhanced_analysis.png"
    plt.savefig(plot_file, dpi=300)
    print(f"Enhanced analysis plots saved: {plot_file}")

    # Save representative frame from enhanced traj
    if "traj" in locals():
        min_frame = np.argmin(rmsd)
        traj[min_frame].save_pdb(str(output_dir / f"{config.PROTEIN_NAME}_enhanced_representative.pdb"))
        print(f"Representative enhanced structure saved.")

    print("=== Enhanced Analysis completed ===")


if __name__ == "__main__":
    run_enhanced_analysis()

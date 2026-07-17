"""
analysis.py

Reusable post-MD analysis module.
Works for any enzyme as long as the output files follow the naming in config.py.
"""

import mdtraj as md
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config


def run_analysis():
    output_dir = config.OUTPUT_DIR
    traj_file = output_dir / "production.dcd"
    top_file = output_dir / "equilibrated.pdb"
    state_csv = output_dir / "production_state.csv"

    print(f"Running analysis for: {config.PROTEIN_NAME}")

    traj = md.load(str(traj_file), top=str(top_file))
    print(f"Trajectory: {traj.n_frames} frames")

    # RMSD
    rmsd = md.rmsd(traj, traj[0], atom_indices=traj.topology.select("backbone"))
    print(f"Mean RMSD: {np.mean(rmsd) * 10:.2f} Å")

    # RMSF
    rmsf = md.rmsf(traj, traj[0], atom_indices=traj.topology.select("backbone"))

    # Radius of gyration
    rg = md.compute_rg(traj)

    # State data
    df = pd.read_csv(state_csv)

    # === Plots ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(rmsd * 10)
    axes[0, 0].set_title(f"RMSD - {config.PROTEIN_NAME}")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("RMSD (Å)")
    axes[0, 0].grid(True)

    axes[0, 1].plot(rmsf * 10)
    axes[0, 1].set_title("RMSF (backbone)")
    axes[0, 1].set_xlabel("Residue")
    axes[0, 1].set_ylabel("RMSF (Å)")
    axes[0, 1].grid(True)

    axes[1, 0].plot(rg * 10)
    axes[1, 0].set_title("Radius of Gyration")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Rg (Å)")
    axes[1, 0].grid(True)

    if "Temperature (K)" in df.columns:
        axes[1, 1].plot(df["Temperature (K)"])
        axes[1, 1].axhline(300, color="red", linestyle="--")
        axes[1, 1].set_title("Temperature")
        axes[1, 1].set_xlabel("Frame")
        axes[1, 1].set_ylabel("K")
        axes[1, 1].grid(True)

    plt.tight_layout()
    plot_file = output_dir / f"{config.PROTEIN_NAME}_analysis.png"
    plt.savefig(plot_file, dpi=300)
    print(f"Plots saved: {plot_file}")

    # Save representative frame
    min_frame = np.argmin(rmsd)
    traj[min_frame].save_pdb(str(output_dir / f"{config.PROTEIN_NAME}_representative.pdb"))
    print(f"Representative structure saved.")


if __name__ == "__main__":
    run_analysis()

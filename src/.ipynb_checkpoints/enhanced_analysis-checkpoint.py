"""
enhanced_analysis.py

Robust OPES analysis.
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
    colvar_file = output_dir / "COLVAR"

    print(f"=== Enhanced OPES Analysis for {config.PROTEIN_NAME} ===")

    # Load trajectory
    if traj_file.exists():
        traj = md.load(str(traj_file), top=str(top_file))
        print(f"Loaded trajectory: {traj.n_frames} frames")
    else:
        print("Trajectory not found!")
        return

    # Load COLVAR - more robust parsing
    if colvar_file.exists():
        # Skip the header line and read properly
        with open(colvar_file, 'r') as f:
            lines = f.readlines()
        
        # Find the line with FIELDS
        header_line = next((i for i, line in enumerate(lines) if "FIELDS" in line), 0)
        colvar = pd.read_csv(colvar_file, sep=r'\s+', skiprows=header_line+1, comment='#')
        
        print(f"Loaded COLVAR with {len(colvar)} entries")
        print("Columns:", list(colvar.columns))
    else:
        colvar = None
        print("COLVAR not found!")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # RMSD
    rmsd = md.rmsd(traj, traj[0], atom_indices=traj.topology.select("backbone"))
    axes[0, 0].plot(rmsd * 10)
    axes[0, 0].set_title("RMSD")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("RMSD (Å)")
    axes[0, 0].grid(True)

    if colvar is not None and len(colvar.columns) > 1:
        # Use column index instead of name for safety
        axes[0, 1].plot(colvar.iloc[:, 1], label="RMSD (CV)")   # usually the second column
        if len(colvar.columns) > 2:
            axes[0, 1].plot(colvar.iloc[:, 2], label="d_active")
        axes[0, 1].set_title("Collective Variables")
        axes[0, 1].set_xlabel("Time (ps)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Bias (usually column with "bias" or 3rd/4th)
        bias_cols = [col for col in colvar.columns if "bias" in str(col).lower()]
        if bias_cols:
            axes[1, 0].plot(colvar[bias_cols[0]], color='red', label=bias_cols[0])
            axes[1, 0].set_title("OPES Bias")
            axes[1, 0].set_xlabel("Time (ps)")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

    # Rg
    rg = md.compute_rg(traj)
    axes[1, 1].plot(rg * 10)
    axes[1, 1].set_title("Radius of Gyration")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Rg (Å)")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_file = output_dir / f"{config.PROTEIN_NAME}_final_opes_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {plot_file}")

if __name__ == "__main__":
    run_enhanced_analysis()
"""
enhanced_analysis.py

Modular post-OPES + DeepTICA analysis.

Usage examples:
  python enhanced_analysis.py              # run everything
  python enhanced_analysis.py --opes       # only classic OPES plots
  python enhanced_analysis.py --deeptica   # only DeepTICA plots
"""

import argparse
from pathlib import Path

import mdtraj as md
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import config


def read_colvar(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    with open(path) as f:
        lines = f.readlines()
    header_line = next((i for i, line in enumerate(lines) if "FIELDS" in line), 0)
    return pd.read_csv(path, sep=r"\s+", skiprows=header_line + 1, comment="#")


def plot_opes(traj, colvar, out: Path):
    """Classic 2×2 OPES figure (RMSD, COLVAR CVs, bias, Rg)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # RMSD
    rmsd = md.rmsd(traj, traj[0], atom_indices=traj.topology.select("backbone"))
    axes[0, 0].plot(rmsd * 10)
    axes[0, 0].set_title("RMSD")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("RMSD (Å)")
    axes[0, 0].grid(True)

    # COLVAR CVs + bias
    if colvar is not None and len(colvar.columns) > 1:
        axes[0, 1].plot(colvar.iloc[:, 1], label=str(colvar.columns[1]))
        if len(colvar.columns) > 2:
            axes[0, 1].plot(colvar.iloc[:, 2], label=str(colvar.columns[2]))
        axes[0, 1].set_title("Collective Variables (COLVAR)")
        axes[0, 1].set_xlabel("Time (ps)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        bias_cols = [c for c in colvar.columns if "bias" in str(c).lower()]
        if bias_cols:
            axes[1, 0].plot(colvar[bias_cols[0]], color="red", label=bias_cols[0])
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
    plot_file = out / f"{config.PROTEIN_NAME}_final_opes_analysis.png"
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" OPES plot saved: {plot_file}")


def plot_deeptica(cvs: np.ndarray, colvar: pd.DataFrame | None, out: Path):
    """DeepTICA time series, 2D space, histograms, optional correlation with OPES CVs."""
    n = len(cvs)
    frames = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time series
    axes[0, 0].plot(frames, cvs[:, 0], label="DeepTICA CV1", lw=0.9, alpha=0.85)
    axes[0, 0].plot(frames, cvs[:, 1], label="DeepTICA CV2", lw=0.9, alpha=0.85)
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("CV value")
    axes[0, 0].set_title("DeepTICA CVs vs frame")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2D scatter colored by time
    sc = axes[0, 1].scatter(cvs[:, 0], cvs[:, 1], c=frames, s=8, cmap="viridis", alpha=0.7)
    axes[0, 1].set_xlabel("DeepTICA CV1")
    axes[0, 1].set_ylabel("DeepTICA CV2")
    axes[0, 1].set_title("CV space (color = time)")
    fig.colorbar(sc, ax=axes[0, 1], label="Frame")
    axes[0, 1].grid(True, alpha=0.3)

    # Histograms
    axes[1, 0].hist(cvs[:, 0], bins=50, alpha=0.7, label="CV1", density=True)
    axes[1, 0].hist(cvs[:, 1], bins=50, alpha=0.7, label="CV2", density=True)
    axes[1, 0].set_xlabel("CV value")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Distributions")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Correlation with original OPES CVs
    if colvar is not None:
        d_active = wat = None
        for col in colvar.columns:
            c = str(col).lower()
            if "d_active" in c:
                d_active = colvar[col].to_numpy()
            if c in ("wat", "water") or "coord" in c:
                wat = colvar[col].to_numpy()

        if d_active is not None and len(d_active) >= n:
            step = max(1, len(d_active) // n)
            d_sub = d_active[::step][:n]
            axes[1, 1].scatter(d_sub, cvs[:, 0], s=6, alpha=0.5, label="CV1 vs d_active")
            if wat is not None:
                w_sub = wat[::step][:n]
                axes[1, 1].scatter(w_sub, cvs[:, 1], s=6, alpha=0.5, label="CV2 vs wat")
            axes[1, 1].set_xlabel("OPES CV")
            axes[1, 1].set_ylabel("DeepTICA CV")
            axes[1, 1].set_title("DeepTICA vs original OPES CVs")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "COLVAR length mismatch\nor no d_active/wat", ha="center", va="center")
            axes[1, 1].set_axis_off()
    else:
        axes[1, 1].text(0.5, 0.5, "No COLVAR found", ha="center", va="center")
        axes[1, 1].set_axis_off()

    plt.tight_layout()
    plot_file = out / f"{config.PROTEIN_NAME}_deeptica_analysis.png"
    fig.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" DeepTICA plot saved: {plot_file}")


def run_enhanced_analysis(do_opes: bool = True, do_deeptica: bool = True):
    output_dir = config.OUTPUT_DIR
    traj_file = output_dir / "enhanced.dcd"
    top_file = output_dir / "equilibrated.pdb"
    colvar_file = output_dir / "COLVAR"
    deeptica_file = output_dir / "projected_cvs.npy"  # sits directly in lysozyme/

    print(f"=== Analysis for {config.PROTEIN_NAME} ===")

    # Load only what is needed
    traj = None
    colvar = None

    if do_opes:
        if not traj_file.exists():
            print("Trajectory not found — cannot make OPES plots.")
            do_opes = False
        else:
            traj = md.load(str(traj_file), top=str(top_file))
            print(f"Loaded trajectory: {traj.n_frames} frames")

    if do_opes or do_deeptica:
        colvar = read_colvar(colvar_file)
        if colvar is not None:
            print(f"Loaded COLVAR ({len(colvar)} entries)")
        else:
            print("COLVAR not found")

    # Run selected analyses
    if do_opes and traj is not None:
        plot_opes(traj, colvar, output_dir)

    if do_deeptica:
        if deeptica_file.exists():
            cvs = np.load(deeptica_file)
            print(f"Loaded DeepTICA CVs: shape {cvs.shape}")
            print(f"  CV1  min={cvs[:, 0].min():.3f}  max={cvs[:, 0].max():.3f}  std={cvs[:, 0].std():.3f}")
            print(f"  CV2  min={cvs[:, 1].min():.3f}  max={cvs[:, 1].max():.3f}  std={cvs[:, 1].std():.3f}")
            plot_deeptica(cvs, colvar, output_dir)
        else:
            print(f"No DeepTICA file at {deeptica_file} — skipping.")

    print("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPES + DeepTICA analysis")
    parser.add_argument("--opes", action="store_true", help="Only classic OPES plots")
    parser.add_argument("--deeptica", action="store_true", help="Only DeepTICA plots")
    args = parser.parse_args()

    # Default = run both
    if not args.opes and not args.deeptica:
        run_enhanced_analysis(do_opes=True, do_deeptica=True)
    else:
        run_enhanced_analysis(do_opes=args.opes, do_deeptica=args.deeptica)

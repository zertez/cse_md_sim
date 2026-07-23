"""
reweight_fes.py - OPES reweighting for the lysozyme run
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

kT = 0.0083145 * 300.0  # kJ/mol at 300 K


def read_colvar(path):
    """Parse a PLUMED COLVAR, taking column names from the '#! FIELDS' line."""
    with open(path) as f:
        names = f.readline().split()[2:]  # drop '#!' and 'FIELDS'
    return pd.read_csv(path, sep=r"\s+", comment="#", names=names)


def reweight(cv):
    """OPES frame weights: w_i = exp(V_i / kT), max-shifted, normalized."""
    # raises if column missing (good)
    bias = cv["opes.bias"].to_numpy()
    w = np.exp((bias - bias.max()) / kT)
    return w / w.sum()


def fes_1d(values, weights, bins=60):
    hist, edges = np.histogram(values, bins=bins, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fes = -kT * np.log(hist + 1e-12)
    return centers, fes - fes.min()


def delta_g(dact, weights, closed=0.70, open_=0.85):
    """ΔG(closed->open) along d_active, in kJ/mol, from reweighted populations."""
    p_closed = weights[dact < closed].sum()
    p_open = weights[dact > open_].sum()
    return -kT * np.log(p_open / p_closed)


def run_reweight_analysis():
    out = config.OUTPUT_DIR
    colvar = out / "COLVAR"
    if not colvar.exists():
        print("❌ COLVAR not found. Run enhanced_sampling.py first.")
        return

    print("=== OPES Reweighting ===")
    cv = read_colvar(colvar)
    w = reweight(cv)
    dact = cv["d_active"].to_numpy()
    rmsd = cv["rmsd"].to_numpy()
    print(f"Loaded {len(cv)} frames; ΔG(closed→open) = {delta_g(dact, w):.2f} kJ/mol")

    # === 1D FES along d_active (the CV that recrossed) ===
    x, fes = fes_1d(dact, w)
    plt.figure(figsize=(10, 6))
    plt.plot(x, fes, lw=2.5)
    plt.xlabel("d_active  (Glu35–Asp52, nm)")
    plt.ylabel("Free energy (kJ/mol)")
    plt.title("Reweighted 1D FES")
    plt.grid(True)
    plt.savefig(out / "fes_d_active_1d.png", dpi=300, bbox_inches="tight")

    # === 2D FES ===
    H, xe, ye = np.histogram2d(rmsd, dact, bins=40, weights=w, density=True)
    F = -kT * np.log(H.T + 1e-12)
    F -= F.min()
    plt.figure(figsize=(10, 8))
    plt.contourf(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), F, levels=30, cmap="viridis")
    plt.colorbar(label="Free energy (kJ/mol)")
    plt.xlabel("RMSD (nm)")
    plt.ylabel("d_active (nm)")
    plt.title("2D Free Energy Surface")
    plt.savefig(out / "fes_2d.png", dpi=300, bbox_inches="tight")

    # === Convergence: ΔG over cumulative time, + second run if present ===
    t = cv["time"].to_numpy()
    frac = np.linspace(0.2, 1.0, 20)
    dg_t = []
    for fr in frac:
        k = int(fr * len(cv))
        wk = w[:k] / w[:k].sum()
        dg_t.append(delta_g(dact[:k], wk))
    plt.figure(figsize=(10, 6))
    plt.plot(t[(frac * len(cv) - 1).astype(int)], dg_t, "o-", label="COLVAR")

    bck = out / "bck.0.COLVAR"
    if bck.exists():
        cv2 = read_colvar(bck)
        w2 = reweight(cv2)
        dg2 = delta_g(cv2["d_active"].to_numpy(), w2)
        plt.axhline(dg2, ls="--", color="gray", label=f"run 2 (bck.0): {dg2:.1f}")

    plt.xlabel("Time (ps)")
    plt.ylabel("ΔG closed→open (kJ/mol)")
    plt.title("Convergence of ΔG")
    plt.legend()
    plt.grid(True)
    plt.savefig(out / "fes_convergence.png", dpi=300, bbox_inches="tight")
    print("Saved: fes_d_active_1d.png, fes_2d.png, fes_convergence.png")


if __name__ == "__main__":
    run_reweight_analysis()

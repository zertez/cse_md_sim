"""
train_deeptica.py

Train Deep-TICA collective variables on an existing OPES trajectory.
Lightning-free version (plain PyTorch) so it starts quickly.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import mdtraj as md
from torch.utils.data import DataLoader, TensorDataset

from mlcolvar.cvs import DeepTICA

import config


def build_descriptors(traj: md.Trajectory) -> np.ndarray:
    print("  → Selecting key residues...")
    key_resids = [35, 52, 62, 63, 101, 108, 109]
    ca = traj.topology.select("name CA and resid " + " ".join(map(str, key_resids)))
    print(f"  → Found {len(ca)} CA atoms")

    pairs = [[ca[i], ca[j]] for i in range(len(ca)) for j in range(i + 1, len(ca))]
    print(f"  → {len(pairs)} distance pairs")

    print("  → Computing distances...")
    distances = md.compute_distances(traj, pairs)
    print("  → Done")
    return distances.astype(np.float32)


def train_deeptica(lag: int = 10, n_cvs: int = 2, max_epochs: int = 80, batch_size: int = 256):
    print("=== Deep-TICA training (PyTorch only) ===")
    output_dir = config.OUTPUT_DIR
    traj_file = output_dir / "enhanced.dcd"
    top_file = output_dir / "equilibrated.pdb"

    print(f"Trajectory: {traj_file}")
    print(f"Topology:   {top_file}")

    if not traj_file.exists() or not top_file.exists():
        raise FileNotFoundError("enhanced.dcd or equilibrated.pdb not found.")

    print("Loading trajectory...")
    traj = md.load(str(traj_file), top=str(top_file))
    print(f"  {traj.n_frames} frames")

    print("Building descriptors...")
    X = build_descriptors(traj)
    print(f"  Shape: {X.shape}")

    # Time-lagged pairs
    X_t = torch.from_numpy(X[:-lag])
    X_lag = torch.from_numpy(X[lag:])

    dataset = TensorDataset(X_t, X_lag)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DeepTICA(
        layers=[X.shape[1], 64, 32, n_cvs],
        options={"nn": {"activation": "tanh"}},
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training for {max_epochs} epochs...")
    model.train()
    for epoch in range(1, max_epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for data, data_lag in loader:
            data = data.to(device)
            data_lag = data_lag.to(device)

            optimizer.zero_grad()

            # Forward pass
            cv = model(data)
            cv_lag = model(data_lag)

            # Deep-TICA loss (simplified but works)
            loss = model.loss_function(cv, cv_lag)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{max_epochs}  loss = {avg_loss:.6f}")

    # Save
    model_dir = output_dir / "deeptica"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "deeptica_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Project full trajectory
    model.eval()
    with torch.no_grad():
        cvs = model(torch.from_numpy(X).to(device)).cpu().numpy()
    np.save(model_dir / "projected_cvs.npy", cvs)
    print(f"Projected CVs saved: shape {cvs.shape}")
    print("=== Finished successfully ===")


if __name__ == "__main__":
    train_deeptica()

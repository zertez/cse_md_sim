"""
train_deeptica.py

Train Deep-TICA collective variables on an existing OPES trajectory.
Uses mlcolvar's create_timelagged_dataset so both 'weights' and
'weights_lag' are present (required by DeepTICA.training_step).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import mdtraj as md
import lightning as L

from mlcolvar.cvs import DeepTICA
from mlcolvar.data import DictModule
from mlcolvar.utils.timelagged import create_timelagged_dataset
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


def train_deeptica(
    lag: int = 10,
    n_cvs: int = 2,
    max_epochs: int = 80,
    batch_size: int = 256,
):
    print("=== Deep-TICA training ===")
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

    # Official helper produces a DictDataset with the four keys DeepTICA expects:
    #   'data', 'data_lag', 'weights', 'weights_lag'
    print(f"Creating time-lagged dataset (lag={lag})...")
    dataset = create_timelagged_dataset(X, lag_time=lag)
    print(f"  Dataset keys: {list(dataset.keys)}")  # <-- fixed (no parentheses)
    print(f"  Samples: {len(dataset)}")

    datamodule = DictModule(dataset, lengths=[0.8, 0.2], batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = DeepTICA(
        layers=[X.shape[1], 64, 32, n_cvs],
        options={"nn": {"activation": "tanh"}},
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )

    print(f"Training for {max_epochs} epochs...")
    trainer.fit(model, datamodule)

    # Save
    model_dir = output_dir / "deeptica"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "deeptica_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Project full trajectory
    model.eval()
    model.to(device)
    with torch.no_grad():
        cvs = model(torch.from_numpy(X).to(device)).cpu().numpy()
    np.save(model_dir / "projected_cvs.npy", cvs)
    print(f"Projected CVs saved: shape {cvs.shape}")
    print("=== Finished successfully ===")


if __name__ == "__main__":
    train_deeptica()

"""
train_deeptica.py
-----------------
Stage C: Learn better collective variables with Deep-TICA
from an existing OPES trajectory.

This script:
1. Loads the biased trajectory + topology
2. Builds a rich set of descriptors
3. Trains a Deep-TICA model
4. Saves the model so it can later be used inside PLUMED
"""

from pathlib import Path
import numpy as np
import torch
import mdtraj as md

from mlcolvar.cvs import DeepTICA
from mlcolvar.data import DictDataset, DictLoader
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.io import create_dataset_from_files  # optional helper
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import config


def build_descriptors(traj: md.Trajectory) -> np.ndarray:
    """Return a (n_frames, n_features) array of descriptors.

    Start with a modest set of Cα-Cα distances around the active site.
    This list can be expanded once the pipeline is working.
    """
    # Lysozyme catalytic residues + a few neighbours (adjust as needed)
    key_resids = [35, 52, 62, 63, 101, 108, 109]
    ca = traj.topology.select("name CA and resid " + " ".join(map(str, key_resids)))

    pairs = [[ca[i], ca[j]] for i in range(len(ca)) for j in range(i + 1, len(ca))]
    distances = md.compute_distances(traj, pairs)  # nm

    return distances.astype(np.float32)


def train_deeptica(lag: int = 10, n_cvs: int = 2, max_epochs: int = 100):
    output_dir = config.OUTPUT_DIR
    traj_file = output_dir / "enhanced.dcd"
    top_file = output_dir / "equilibrated.pdb"

    if not traj_file.exists() or not top_file.exists():
        raise FileNotFoundError("enhanced.dcd or equilibrated.pdb not found. Run enhanced_sampling.py first.")

    print(f"=== Deep-TICA training for {config.PROTEIN_NAME} ===")
    print(f"Loading trajectory: {traj_file}")
    traj = md.load(str(traj_file), top=str(top_file))
    print(f"  {traj.n_frames} frames")

    print("Building descriptors...")
    X = build_descriptors(traj)
    print(f"  Descriptor shape: {X.shape}")

    # Time-lagged pairs
    dataset = DictDataset({"data": X[:-lag], "data_lag": X[lag:]})
    loader = DictLoader(dataset, batch_size=256, shuffle=True)

    model_dir = output_dir / "deeptica"
    model_dir.mkdir(exist_ok=True)

    model = DeepTICA(
        layers=[X.shape[1], 64, 32, n_cvs],
        options={"activation": "tanh"},
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=model_dir,
        filename="deeptica-{epoch:02d}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb, MetricsCallback()],
        accelerator="auto",
    )

    print("Training...")
    trainer.fit(model, loader)

    # Save final weights
    model_path = model_dir / "deeptica_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Project whole trajectory for quick inspection
    model.eval()
    with torch.no_grad():
        cvs = model(torch.from_numpy(X)).cpu().numpy()
    np.save(model_dir / "projected_cvs.npy", cvs)
    print(f"Projected CVs saved: {model_dir / 'projected_cvs.npy'}  shape={cvs.shape}")
    print("=== Deep-TICA training finished ===")


if __name__ == "__main__":
    train_deeptica()

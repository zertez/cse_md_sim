"""
train_deeptica.py

This script:
1. Loads the biased trajectory + topology
2. Builds a rich set of descriptors
3. Trains a Deep-TICA model
4. Saves the model so it can later be used inside PLUMED
"""

import sys
from pathlib import Path

# Make sure we can import config from the parent folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import mdtraj as md
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from mlcolvar.cvs import DeepTICA
from mlcolvar.data import DictDataset, DictLoader
from mlcolvar.utils.trainer import MetricsCallback

import config


def build_descriptors(traj: md.Trajectory) -> np.ndarray:
    print("  → Selecting key residues...")
    key_resids = [35, 52, 62, 63, 101, 108, 109]
    ca = traj.topology.select("name CA and resid " + " ".join(map(str, key_resids)))
    print(f"  → Found {len(ca)} CA atoms")

    print("  → Building pairwise distance list...")
    pairs = [[ca[i], ca[j]] for i in range(len(ca)) for j in range(i + 1, len(ca))]
    print(f"  → {len(pairs)} distance pairs")

    print("  → Computing distances with mdtraj...")
    distances = md.compute_distances(traj, pairs)  # nm
    print("  → Distance calculation finished")

    return distances.astype(np.float32)


def train_deeptica(lag: int = 10, n_cvs: int = 2, max_epochs: int = 100):
    print("=== Deep-TICA training starting ===")
    output_dir = config.OUTPUT_DIR
    print(f"Output directory: {output_dir}")

    traj_file = output_dir / "enhanced.dcd"
    top_file = output_dir / "equilibrated.pdb"
    print(f"Looking for trajectory: {traj_file}")
    print(f"Looking for topology:   {top_file}")

    if not traj_file.exists() or not top_file.exists():
        raise FileNotFoundError("enhanced.dcd or equilibrated.pdb not found. Run enhanced_sampling.py first.")

    print("Loading trajectory with mdtraj...")
    traj = md.load(str(traj_file), top=str(top_file))
    print(f"  Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")

    print("Building descriptors...")
    X = build_descriptors(traj)
    print(f"  Descriptor matrix shape: {X.shape}")

    print(f"Creating time-lagged dataset (lag = {lag} frames)...")
    dataset = DictDataset({"data": X[:-lag], "data_lag": X[lag:]})
    loader = DictLoader(dataset, batch_size=256, shuffle=True)
    print("  Dataset and loader ready")

    model_dir = output_dir / "deeptica"
    model_dir.mkdir(exist_ok=True)
    print(f"Model will be saved to: {model_dir}")

    print("Creating DeepTICA model...")
    model = DeepTICA(
        layers=[X.shape[1], 64, 32, n_cvs],
        options={"activation": "tanh"},
    )
    print("  Model created")

    checkpoint_cb = ModelCheckpoint(
        dirpath=model_dir,
        filename="deeptica-{epoch:02d}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    print("Creating Lightning Trainer...")
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb, MetricsCallback()],
        accelerator="auto",
    )
    print("  Trainer ready")

    print(f"Starting training for {max_epochs} epochs...")
    trainer.fit(model, loader)
    print("Training finished!")

    model_path = model_dir / "deeptica_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to: {model_path}")

    print("Projecting trajectory onto learned CVs...")
    model.eval()
    with torch.no_grad():
        cvs = model(torch.from_numpy(X)).cpu().numpy()
    np.save(model_dir / "projected_cvs.npy", cvs)
    print(f"Projected CVs saved: {model_dir / 'projected_cvs.npy'}  shape={cvs.shape}")

    print("=== Deep-TICA training finished successfully ===")


if __name__ == "__main__":
    train_deeptica()

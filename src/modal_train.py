"""
modal_train.py
"""

import modal
from pathlib import Path

app = modal.App("cse-md-sim")
volume = modal.Volume.from_name("cse-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "mlcolvar",
        "mdtraj",
        "numpy",
        "tqdm",
        "openmm",
    )
    .add_local_dir(
        local_path="/Users/marcusdalakerfigenschou/Documents/UiB/cse_drug_design_programming/cse_md_sim/src",
        remote_path="/root/cse_md_sim/src",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def run_deeptica_job():
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/cse_md_sim/src")

    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    import config

    config.OUTPUT_DIR = Path("/data/data/output/lysozyme")

    from ml.train_deeptica import train_deeptica

    train_deeptica()

    volume.commit()
    print("Volume committed.")


@app.local_entrypoint()
def main():
    run_deeptica_job.remote()

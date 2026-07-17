"""
enhanced_sampling.py

Modular enhanced sampling module using PLUMED + OPES.
Designed to be reusable across enzymes (just change the CV in plumed.dat).
"""

from openmm import *
from openmm.app import *
from openmm import unit
import config
from system_builder import build_system
from pathlib import Path
import subprocess


def run_enhanced_sampling(plumed_file: str = "plumed.dat"):
    print("=== Starting Enhanced Sampling (OPES) ===")

    output_dir = config.OUTPUT_DIR
    checkpoint_file = output_dir / "equilibrated.chk"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # Rebuild system
    modeller, system, topology, positions = build_system()

    # Add barostat
    system.addForce(MonteCarloBarostat(config.PRESSURE, config.TEMPERATURE))

    integrator = LangevinMiddleIntegrator(config.TEMPERATURE, 1.0 / unit.picosecond, config.TIMESTEP)

    simulation = Simulation(topology, system, integrator)
    simulation.loadCheckpoint(str(checkpoint_file))

    # === PLUMED Integration ===
    # We use PlumedForce (requires openmm-plumed or compatible setup)
    try:
        from openmmplumed import PlumedForce

        plumed_force = PlumedForce(str(Path("scripts") / plumed_file))
        system.addForce(plumed_force)
        print(f"PLUMED loaded with: {plumed_file}")
    except ImportError:
        print("Warning: openmmplumed not found. Make sure PLUMED is properly linked with OpenMM.")
        print("Falling back to running without bias for now.")

    # Reporters
    simulation.reporters.append(DCDReporter(str(output_dir / "enhanced.dcd"), 5000))
    simulation.reporters.append(
        StateDataReporter(
            str(output_dir / "enhanced_state.csv"),
            5000,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            separator=",",
        )
    )

    # Run biased simulation
    steps = 500000  # 1 ns for now (you can increase later)
    print(f"Running enhanced sampling for {steps * config.TIMESTEP}...")
    simulation.step(steps)
    print("Enhanced sampling finished.")

    # Save final checkpoint
    simulation.saveCheckpoint(str(output_dir / "enhanced.chk"))
    print(f"Enhanced checkpoint saved.")


if __name__ == "__main__":
    run_enhanced_sampling()

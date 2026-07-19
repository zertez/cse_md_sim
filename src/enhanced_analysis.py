"""
enhanced_sampling.py

Enhanced sampling using PLUMED + OPES.
Modular and reusable across enzymes.
"""

from openmm import *
from openmm.app import *
from openmm import unit
import config
from system_builder import build_system
from pathlib import Path


def run_enhanced_sampling(plumed_file: str = "plumed.dat"):
    print("=== Starting Enhanced Sampling with OPES ===")

    output_dir = config.OUTPUT_DIR
    checkpoint_file = output_dir / "equilibrated.chk"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # Build system
    modeller, system, topology, positions = build_system()
    system.addForce(MonteCarloBarostat(config.PRESSURE, config.TEMPERATURE))

    integrator = LangevinMiddleIntegrator(config.TEMPERATURE, 1.0 / unit.picosecond, config.TIMESTEP)

    simulation = Simulation(topology, system, integrator)
    simulation.loadCheckpoint(str(checkpoint_file))

    # === Load PLUMED with full path (fixed) ===
    plumed_path = Path("/workspace/cse_md_sim/scripts/plumed.dat")
    print(f"Using PLUMED file: {plumed_path}")
    print(f"File exists: {plumed_path.exists()}")

    try:
        from openmmplumed import PlumedForce

        plumed_force = PlumedForce(str(plumed_path))
        system.addForce(plumed_force)
        print(f"PLUMED successfully loaded: {plumed_path}")
    except ImportError:
        print("ERROR: openmm-plumed is not installed.")
        print("Please run: pixi add openmm-plumed")
        return
    except Exception as e:
        print(f"Failed to load PLUMED: {e}")
        return

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

    # Run
    steps = 500000  # 1 ns (increase later)
    print(f"Running {steps} steps with OPES bias...")
    simulation.step(steps)
    print("Enhanced sampling finished.")

    # Save checkpoint
    simulation.saveCheckpoint(str(output_dir / "enhanced.chk"))
    print(f"Enhanced checkpoint saved to {output_dir / 'enhanced.chk'}")


if __name__ == "__main__":
    run_enhanced_sampling()

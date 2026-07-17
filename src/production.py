"""
production.py

Loads the equilibrated checkpoint and runs production MD.
"""

from openmm import *
from openmm.app import *
from openmm import unit
import config
from system_builder import build_system
from pathlib import Path


def run_production():
    print("Starting Production")

    output_dir = config.OUTPUT_DIR
    checkpoint_file = output_dir / "equilibrated.chk"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}\nPlease run equilibration.py first.")

    # Rebuild the same system
    modeller, system, topology, positions = build_system()

    # Add barostat (must match what was used in equilibration)
    system.addForce(MonteCarloBarostat(config.PRESSURE, config.TEMPERATURE))

    integrator = LangevinMiddleIntegrator(config.TEMPERATURE, 1.0 / unit.picosecond, config.TIMESTEP)

    simulation = Simulation(topology, system, integrator)

    # Load the equilibrated state
    print(f"Loading checkpoint: {checkpoint_file}")
    simulation.loadCheckpoint(str(checkpoint_file))

    # Reporters
    simulation.reporters.append(
        DCDReporter(str(output_dir / "production.dcd"), 5000)  # every 10 ps
    )

    simulation.reporters.append(
        StateDataReporter(
            str(output_dir / "production_state.csv"),
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

    simulation.reporters.append(CheckpointReporter(str(output_dir / "production_checkpoint.chk"), 50000))

    # Production
    print(f"Running production ({config.PRODUCTION_STEPS * config.TIMESTEP})...")
    simulation.step(config.PRODUCTION_STEPS)
    print("Production finished.")

    # Save final state
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with open(output_dir / "final_state.xml", "w") as f:
        f.write(XmlSerializer.serialize(state))

    print(f"All production output written to: {output_dir}")
    print("=== Production completed successfully ===")


if __name__ == "__main__":
    run_production()

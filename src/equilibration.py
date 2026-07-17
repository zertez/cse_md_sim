"""
equilibration.py
----------------
Energy minimization + NVT + NPT equilibration.
Saves a checkpoint that production.py can restart from.
"""

from openmm import *
from openmm.app import *
from openmm import unit
import config
from system_builder import build_system
from pathlib import Path


def run_equilibration():
    print("Starting Equilibration")

    # Build system
    modeller, system, topology, positions = build_system()

    # Integrator
    integrator = LangevinMiddleIntegrator(config.TEMPERATURE, 1.0 / unit.picosecond, config.TIMESTEP)

    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # Minimization
    print("Minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)
    print("Minimization finished.")

    # Reporters
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation.reporters.append(
        StateDataReporter(
            str(output_dir / "equilibration_state.csv"),
            1000,
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

    # NVT
    print(f"Running NVT equilibration ({config.NVT_STEPS * config.TIMESTEP})...")
    simulation.context.setVelocitiesToTemperature(config.TEMPERATURE)
    simulation.step(config.NVT_STEPS)
    print("NVT finished.")

    # NPT
    print(f"Running NPT equilibration ({config.NPT_STEPS * config.TIMESTEP})...")
    system.addForce(MonteCarloBarostat(config.PRESSURE, config.TEMPERATURE))
    simulation.context.reinitialize(preserveState=True)

    simulation.step(config.NPT_STEPS)
    print("NPT finished.")

    # Save checkpoint for production
    checkpoint_file = output_dir / "equilibrated.chk"
    simulation.saveCheckpoint(str(checkpoint_file))
    print(f"Checkpoint saved to: {checkpoint_file}")

    # Also save final positions as PDB (nice for inspection)
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(topology, positions, open(output_dir / "equilibrated.pdb", "w"))
    print("Equilibrated structure written to equilibrated.pdb")

    print("=== Equilibration completed successfully ===\n")
    return simulation


if __name__ == "__main__":
    run_equilibration()

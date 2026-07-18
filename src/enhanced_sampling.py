"""
enhanced_sampling.py

Enhanced sampling with PLUMED + OPES on RunPod.
This version loads from equilibrated.pdb (much more reliable).
"""

from openmm import *
from openmm.app import *
from openmm import unit
import config
from pathlib import Path


def run_enhanced_sampling(plumed_file: str = "plumed.dat"):
    print("=== Starting Enhanced Sampling (OPES) ===")

    output_dir = config.OUTPUT_DIR
    equilibrated_pdb = output_dir / "equilibrated.pdb"

    if not equilibrated_pdb.exists():
        raise FileNotFoundError(f"{equilibrated_pdb} not found!")

    print(f"Loading equilibrated structure from: {equilibrated_pdb}")
    pdb = PDBFile(str(equilibrated_pdb))

    # Build system
    forcefield = ForceField(*config.FORCEFIELD_FILES)
    modeller = Modeller(pdb.topology, pdb.positions)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
    )
    system.addForce(MonteCarloBarostat(config.PRESSURE, config.TEMPERATURE))

    integrator = LangevinMiddleIntegrator(config.TEMPERATURE, 1.0 / unit.picosecond, config.TIMESTEP)

    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # === PLUMED ===
    plumed_path = Path("scripts") / plumed_file
    try:
        from openmmplumed import PlumedForce

        plumed_force = PlumedForce(str(plumed_path))
        system.addForce(plumed_force)
        print(f"PLUMED loaded: {plumed_path}")
    except Exception as e:
        print(f"PLUMED failed: {e}")
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
    steps = 500000  # 1 ns — increase this later (e.g. 5_000_000)
    print(f"Running {steps} steps with OPES bias...")
    simulation.step(steps)
    print("Enhanced sampling finished.")

    # Save results
    simulation.saveCheckpoint(str(output_dir / "enhanced.chk"))
    final_positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(modeller.topology, final_positions, open(output_dir / "enhanced_final.pdb", "w"))

    print(f"\n Done. Results saved in: {output_dir}")


if __name__ == "__main__":
    run_enhanced_sampling()

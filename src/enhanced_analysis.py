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

# Residues that make up the RMSD reference. Water and ions are excluded:
PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "HID",
    "HIE",
    "HIP",
    "CYX",
    "ASH",
    "GLH",
    "LYN",
}

BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def write_rmsd_reference(structure: Path, reference: Path) -> int:
    """Write a backbone-only PLUMED RMSD reference from a solvated PDB.

    Two details matter and neither is optional:

    - Serial numbers must be the atom's 1-based index in the *full* system,
      because that is how PLUMED addresses atoms. They are recounted here
      rather than copied from the serial column, which wraps past 99999.
    - PLUMED reads the occupancy column as alignment weights and the B-factor
      column as displacement weights. OpenMM writes B-factors of 0.00, which
      leaves the RMSD undefined, so both columns are forced to 1.00.
    """
    kept = []
    index = 0

    for line in structure.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue

        index += 1
        residue = line[17:20].strip()
        atom = line[12:16].strip()

        if residue not in PROTEIN_RESIDUES or atom not in BACKBONE_ATOMS:
            continue

        kept.append(f"ATOM  {index:5d}" + line[11:54] + "  1.00  1.00" + line[66:])

    if not kept:
        raise ValueError(f"No backbone atoms found in {structure}")

    reference.write_text("\n".join(kept) + "\nEND\n")
    return len(kept)


def resolve_plumed_script(template: Path, output_dir: Path, reference: Path) -> str:
    """Substitute @TOKEN@ placeholders in the PLUMED input with absolute paths.

    PLUMED resolves relative paths against the working directory, which differs
    between a local run and the RunPod container, so every path it touches is
    made absolute here instead.
    """
    substitutions = {
        "@REFERENCE@": reference,
        "@COLVAR@": output_dir / "COLVAR",
        "@KERNELS@": output_dir / "KERNELS",
        "@STATE@": output_dir / "opes_state.data",
    }

    script = template.read_text()
    for token, path in substitutions.items():
        script = script.replace(token, str(path.resolve()))

    unresolved = [line for line in script.splitlines() if "@" in line and not line.startswith("#")]
    if unresolved:
        raise ValueError(f"Unsubstituted PLUMED tokens: {unresolved}")

    return script


def run_enhanced_sampling(plumed_file: str = "plumed.dat"):
    print("=== Starting Enhanced Sampling with OPES ===")

    output_dir = config.OUTPUT_DIR
    checkpoint_file = output_dir / "equilibrated.chk"
    equilibrated_pdb = output_dir / "equilibrated.pdb"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if not equilibrated_pdb.exists():
        raise FileNotFoundError(f"Equilibrated structure not found: {equilibrated_pdb}")

    plumed_template = config.PROJECT_ROOT / "scripts" / plumed_file
    if not plumed_template.exists():
        raise FileNotFoundError(f"PLUMED input not found: {plumed_template}")

    # PLUMED reference and resolved input
    reference = output_dir / "rmsd_reference.pdb"
    n_atoms = write_rmsd_reference(equilibrated_pdb, reference)
    print(f"RMSD reference written: {reference} ({n_atoms} backbone atoms)")

    script = resolve_plumed_script(plumed_template, output_dir, reference)
    (output_dir / "plumed_resolved.dat").write_text(script)
    print(f"PLUMED input resolved from {plumed_template}")

    # Build system
    modeller, system, topology, positions = build_system()
    system.addForce(MonteCarloBarostat(config.PRESSURE, config.TEMPERATURE))

    integrator = LangevinMiddleIntegrator(config.TEMPERATURE, 1.0 / unit.picosecond, config.TIMESTEP)

    # The checkpoint is loaded against the same system equilibration saved it
    # from, before PLUMED enters the picture.
    simulation = Simulation(topology, system, integrator)
    simulation.loadCheckpoint(str(checkpoint_file))

    # === Load PLUMED ===
    # PlumedForce takes the script itself, not a path to it. The reinitialize
    # is what actually puts the force into the running context: the Context
    # copies the System when it is constructed, so a force added afterwards is
    # otherwise silently ignored and the run proceeds unbiased.
    from openmmplumed import PlumedForce

    system.addForce(PlumedForce(script))
    simulation.context.reinitialize(preserveState=True)
    print("PLUMED force active in context.")

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
    print(f"COLVAR written to {output_dir / 'COLVAR'}")


if __name__ == "__main__":
    run_enhanced_sampling()

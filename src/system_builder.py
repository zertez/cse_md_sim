"""
system_builder.py

Builds a fully solvated OpenMM System using designated force fields.
"""

from openmm import *
from openmm.app import *
from openmm import unit
from pathlib import Path
import config


def build_system(protein_file: Path = None):
    if protein_file is None:
        protein_file = config.PROTEIN_FILE

    print(f"Loading structure: {protein_file}")

    # Support both PDB and CIF
    if str(protein_file).lower().endswith(".cif"):
        pdb = PDBxFile(str(protein_file))
    else:
        pdb = PDBFile(str(protein_file))

    modeller = Modeller(pdb.topology, pdb.positions)

    # Load AMBER force field
    forcefield = ForceField(*config.FORCEFIELD_FILES)

    print("Checking / adding hydrogens...")
    modeller.addHydrogens(forcefield, pH=7.0)

    print("Adding solvent and ions...")
    modeller.addSolvent(forcefield, padding=config.PADDING, ionicStrength=config.IONIC_STRENGTH)

    print(f"Total atoms after solvation: {modeller.topology.getNumAtoms()}")

    print("Creating OpenMM System...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
    )

    return modeller, system, modeller.topology, modeller.positions


if __name__ == "__main__":
    modeller, system, topology, positions = build_system()
    print("System built successfully!")

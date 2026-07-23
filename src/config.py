from pathlib import Path
from openmm import unit

# === PATHS ===
PROJECT_ROOT = Path(__file__).parent.parent

# Changed to lysozyme
PROTEIN_NAME = "lysozyme"

PROTEIN_FILE = PROJECT_ROOT / "data/protein_files" / PROTEIN_NAME / "1AKI_prep.pdb"
OUTPUT_DIR = PROJECT_ROOT / "data/output" / PROTEIN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === SIMULATION PARAMETERS ===
TEMPERATURE = 300 * unit.kelvin
PRESSURE = 1.0 * unit.bar
TIMESTEP = 0.002 * unit.picoseconds

# Equilibration
NVT_STEPS = 50000  # 100 ps
NPT_STEPS = 50000  # 100 ps

# Production (you can keep 1 ns for now or increase)
PRODUCTION_STEPS = 500000  # 1 ns

# Enhanced sampling (OPES) - the real run, not the 1 ns proof of concept.
# A 2D FES needs many recrossings; 1 ns / 100 frames gave ~10 effective
# samples after reweighting, far too few. Increase ENHANCED_STEPS as needed.
ENHANCED_STEPS = 5000000  # 10 ns at 2 fs
ENHANCED_DCD_STRIDE = 2500  # save a frame every 5 ps (structure extraction)

# Force field – REMOVE GLYCAM (no sugar)
FORCEFIELD_FILES = [
    "amber14-all.xml",  # protein
    "amber14/tip3p.xml",  # water
]

# Solvation
PADDING = 1.0 * unit.nanometer
IONIC_STRENGTH = 0.15 * unit.molar

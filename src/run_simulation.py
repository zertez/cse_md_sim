"""
run_simulation.py

Runs the full workflow
"""

from equilibration import run_equilibration
from production import run_production


if __name__ == "__main__":
    run_equilibration()
    run_production()

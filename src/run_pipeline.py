"""
run_pipeline.py

Main orchestrator. Easy to extend with more stages later.
"""

from equilibration import run_equilibration
from production import run_production
from analysis import run_analysis


def main():
    print("=== Starting Full MD Pipeline ===")
    run_equilibration()
    run_production()
    run_analysis()
    print("=== Pipeline finished successfully ===")


if __name__ == "__main__":
    main()

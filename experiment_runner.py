#!/usr/bin/env python3
"""
experiment_runner.py

Orchestrates the full CMDP -> PyBullet experiment.

For each terrain:
    1. Generate terrain-specific start/goal scenarios once.
    2. For each scenario:
       - solve CMDP for D = 3, 5, 7, 10, 15
       - save the resulting CMDP path
       - run the SAME path for Monte Carlo seeds 1..25
    3. Each seed gets its own output directory.

IMPORTANT:
    This runner calls the existing run.py. It does NOT modify run.py.
    The CMDP implementation is also not modified by this file.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

D_VALUES = [3, 5, 7, 10, 15]
NUM_MONTE_CARLO = 25
SCENARIO_SEED = 42
SCENARIO_GENERATOR = "generate_scenarios.py"
RUN_SCRIPT = "run.py"
CMDP_SCRIPT = "cmdp_refactored.py"


def run_command(cmd, cwd):
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 80)

    result = subprocess.run(cmd, cwd=str(cwd), check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{' '.join(str(x) for x in cmd)}"
        )


def generate_scenarios(terrain_dir, python_exe):
    scenario_file = terrain_dir / "start_goal_scenarios.json"

    if scenario_file.exists():
        print(f"\nScenarios already exist: {scenario_file}")
        print("Keeping the existing terrain-specific scenarios.")
        return

    run_command(
        [
            python_exe,
            SCENARIO_GENERATOR,
            "--terrain-dir",
            str(terrain_dir),
            "--seed",
            str(SCENARIO_SEED),
        ],
        cwd=terrain_dir.parent,
    )


def load_scenarios(terrain_dir):
    path = terrain_dir / "start_goal_scenarios.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = []

    for scenario_type, items in data["scenarios"].items():
        for item in items:
            scenario = dict(item)
            scenario["scenario_type"] = scenario_type
            scenarios.append(scenario)

    return scenarios


def run_cmdp(
    terrain_dir,
    scenario,
    risk_budget,
    experiment_dir,
    python_exe,
):
    scenario_id = scenario["scenario_id"]
    start = scenario["start"]
    goal = scenario["goal"]

    condition_dir = terrain_dir / scenario_id / f"D{risk_budget:g}"
    condition_dir.mkdir(parents=True, exist_ok=True)

    cmdp_result = condition_dir / "cmdp_result.json"
    cmdp_path = condition_dir / "cmdp_path_world.csv"

    if cmdp_result.exists() and cmdp_path.exists():
        print(f"\nCMDP already exists: {condition_dir}")
        return cmdp_path

    run_command(
        [
            python_exe,
            str(experiment_dir / CMDP_SCRIPT),
            "--terrain-grid",
            str(terrain_dir / "terrain_grid.json"),
            "--start",
            f"{int(start[0])},{int(start[1])}",
            "--goal",
            f"{int(goal[0])},{int(goal[1])}",
            "--D",
            str(risk_budget),
            "--output",
            str(cmdp_result),
            "--path-output",
            str(cmdp_path),
            "--backend",
            "auto",
        ],
        cwd=experiment_dir,
    )

    if not cmdp_path.exists():
        raise RuntimeError(f"CMDP did not create path: {cmdp_path}")

    return cmdp_path


def run_monte_carlo(
    terrain_dir,
    scenario_id,
    risk_budget,
    cmdp_path,
    experiment_dir,
    python_exe,
):
    condition_dir = terrain_dir / scenario_id / f"D{risk_budget:g}"

    for seed in range(1, NUM_MONTE_CARLO + 1):
        seed_dir = condition_dir / f"seed_{seed:03d}"

        result_files = list(seed_dir.glob("result_*.json"))
        if result_files:
            print(
                f"Seed {seed:03d}: already completed -> "
                f"{result_files[0]}"
            )
            continue

        seed_dir.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                python_exe,
                str(experiment_dir / RUN_SCRIPT),
                "--terrain-dir",
                str(terrain_dir),
                "--cmdp-path",
                str(cmdp_path),
                "--scenario-id",
                scenario_id,
                "--risk-budget",
                str(risk_budget),
                "--seed",
                str(seed),
                "--output-dir",
                str(seed_dir),
                "--headless",
            ],
            cwd=experiment_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete CMDP + PyBullet Monte Carlo experiment."
    )

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        required=True,
        help="Directory containing terrain_01, terrain_02, ...",
    )

    parser.add_argument(
        "--terrain",
        type=str,
        default=None,
        help="Run only one terrain, e.g. terrain_01.",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child scripts.",
    )

    args = parser.parse_args()

    experiments_dir = args.experiments_dir.resolve()
    experiment_dir = Path(__file__).resolve().parent

    if args.terrain:
        terrain_dirs = [experiments_dir / args.terrain]
    else:
        terrain_dirs = sorted(
            p for p in experiments_dir.iterdir()
            if p.is_dir() and p.name.startswith("terrain_")
        )

    if not terrain_dirs:
        raise RuntimeError(
            f"No terrain_* directories found in {experiments_dir}"
        )

    print("=" * 80)
    print("FULL EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Experiments directory : {experiments_dir}")
    print(f"Terrains              : {len(terrain_dirs)}")
    print(f"D values              : {D_VALUES}")
    print(f"Monte Carlo seeds     : {NUM_MONTE_CARLO}")
    print(f"Scenario seed         : {SCENARIO_SEED}")

    total_conditions = 0
    total_trials = 0

    for terrain_dir in terrain_dirs:
        print("\n" + "#" * 80)
        print(f"TERRAIN: {terrain_dir.name}")
        print("#" * 80)

        terrain_grid = terrain_dir / "terrain_grid.json"
        if not terrain_grid.exists():
            print(f"Skipping: missing {terrain_grid}")
            continue

        generate_scenarios(terrain_dir, args.python)
        scenarios = load_scenarios(terrain_dir)

        print(f"Loaded {len(scenarios)} scenarios.")

        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]

            print("\n" + "-" * 80)
            print(
                f"SCENARIO: {scenario_id} | "
                f"{scenario['start']} -> {scenario['goal']}"
            )
            print("-" * 80)

            for D in D_VALUES:
                total_conditions += 1

                print(
                    f"\n>>> {terrain_dir.name} | "
                    f"{scenario_id} | D={D}"
                )

                cmdp_path = run_cmdp(
                    terrain_dir=terrain_dir,
                    scenario=scenario,
                    risk_budget=D,
                    experiment_dir=experiment_dir,
                    python_exe=args.python,
                )

                run_monte_carlo(
                    terrain_dir=terrain_dir,
                    scenario_id=scenario_id,
                    risk_budget=D,
                    cmdp_path=cmdp_path,
                    experiment_dir=experiment_dir,
                    python_exe=args.python,
                )

                total_trials += NUM_MONTE_CARLO

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"CMDP conditions processed : {total_conditions}")
    print(f"PyBullet trials processed : {total_trials}")
    print("=" * 80)


if __name__ == "__main__":
    main()

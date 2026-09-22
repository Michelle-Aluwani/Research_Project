#!/usr/bin/env python3
"""
baseline_runner.py

Runs the supplied Standard RRT* and Risk-weighted RRT* baselines using:

    - the same terrain folders as CMDP,
    - the same terrain-specific start/goal scenarios,
    - the same 25 Monte Carlo seeds,
    - the same existing run.py physical execution,
    - and no modification to run.py.

Important:
    RRT* paths are deliberately saved as:

        cmdp_path_world.csv

because run.py already accepts that filename/format through --cmdp-path.
The file name does NOT mean the path came from CMDP.

For every planner seed:
    1. RRT* generates one stochastic path.
    2. That path is saved as cmdp_path_world.csv.
    3. The untouched run.py is called with that path.
    4. The same seed is passed to run.py.

Risk-weighted RRT* uses:
    alpha = 0, 0.25, 0.50, 0.75, 1.0

Beta is calculated once per terrain/scenario and stored in the existing
start_goal_scenarios.json. Existing start/goal coordinates are never changed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ALPHA_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]
NUM_MONTE_CARLO = 25
SCENARIO_SEED = 42

RRT_SCRIPT = "rrt_baseline.py"
RUN_SCRIPT = "run.py"

# run.py requires --risk-budget even though the physical simulation does not
# use it to alter the physics. Zero is used only as a compatibility field for
# the baseline execution record.
BASELINE_RISK_BUDGET = 0.0


def run_command(cmd, cwd):
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 80)

    result = subprocess.run(cmd, cwd=str(cwd), check=False)

    if result.returncode not in (0, 2):
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}:\n"
            f"{' '.join(str(x) for x in cmd)}"
        )

    return result.returncode


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_scenarios(path):
    data = load_json(path)

    scenarios = []
    for scenario_type, items in data["scenarios"].items():
        for item in items:
            scenario = dict(item)
            scenario["scenario_type"] = scenario_type
            scenarios.append(scenario)

    return data, scenarios


def calculate_beta_once(
    terrain_dir,
    scenario_id,
    start,
    goal,
    experiment_dir,
    python_exe,
):
    """
    Ask the integrated RRT* implementation to calculate the approved beta
    from the exact scenario start/goal.

    The value is calculated once by this runner and then passed explicitly to
    every alpha and seed for that terrain/scenario.
    """
    # Importing here avoids requiring the planner during simple CLI inspection.
    sys.path.insert(0, str(experiment_dir))
    import rrt_baseline

    rrt_baseline.configure_terrain(terrain_dir)
    return rrt_baseline.calculate_beta(
        tuple(start),
        tuple(goal),
    )


def ensure_betas(terrain_dir, scenario_file, experiment_dir, python_exe):
    """
    Backfill beta into an existing scenario file without regenerating or
    changing any start/goal pair.
    """
    data, scenarios = load_scenarios(scenario_file)

    changed = False

    for scenario in scenarios:
        if "beta" in scenario and scenario["beta"] is not None:
            continue

        beta = calculate_beta_once(
            terrain_dir=terrain_dir,
            scenario_id=scenario["scenario_id"],
            start=scenario["start"],
            goal=scenario["goal"],
            experiment_dir=experiment_dir,
            python_exe=python_exe,
        )

        for item in data["scenarios"][scenario["scenario_type"]]:
            if item["scenario_id"] == scenario["scenario_id"]:
                item["beta"] = float(beta)
                changed = True
                print(
                    f"Stored beta: {terrain_dir.name} / "
                    f"{scenario['scenario_id']} = {beta:.8f}"
                )
                break

    if changed:
        data.setdefault("metadata", {})
        data["metadata"]["beta_definition"] = (
            "beta = representative scenario edge distance / "
            "representative scenario edge hazard probability. "
            "The representative edge is the fixed start-to-goal scenario "
            "edge evaluated using the RRT* edge-distance and "
            "edge-hazard definitions. Beta is fixed for all alpha values "
            "and Monte Carlo seeds of that terrain/scenario."
        )
        save_json(data, scenario_file)
        print(f"Updated beta values: {scenario_file}")
    else:
        print(f"All beta values already present: {scenario_file}")

    return load_scenarios(scenario_file)[1]


def run_planner(
    terrain_dir,
    scenario,
    algorithm,
    seed,
    experiment_dir,
    python_exe,
    alpha=None,
    beta=None,
):
    scenario_id = scenario["scenario_id"]

    if algorithm == "rrt_star":
        condition_dir = (
            terrain_dir
            / "baseline"
            / scenario_id
            / "rrt_star"
            / f"seed_{seed:03d}"
        )
    else:
        condition_dir = (
            terrain_dir
            / "baseline"
            / scenario_id
            / "risk_weighted_rrt_star"
            / f"alpha_{alpha:.2f}"
            / f"seed_{seed:03d}"
        )

    condition_dir.mkdir(parents=True, exist_ok=True)

    path_output = condition_dir / "cmdp_path_world.csv"
    planner_result = condition_dir / "planner_result.json"

    if path_output.exists() and planner_result.exists():
        print(
            f"Planner already completed: "
            f"{terrain_dir.name}/{scenario_id}/{algorithm}/seed={seed}"
        )
        return path_output, planner_result

    cmd = [
        python_exe,
        str(experiment_dir / RRT_SCRIPT),
        "--terrain-dir",
        str(terrain_dir),
        "--scenario-id",
        scenario_id,
        "--algorithm",
        algorithm,
        "--seed",
        str(seed),
        "--path-output",
        str(path_output),
        "--result-output",
        str(planner_result),
    ]

    if alpha is not None:
        cmd += ["--alpha", str(alpha)]

    if beta is not None:
        cmd += ["--beta", str(beta)]

    return_code = run_command(cmd, cwd=experiment_dir)

    if return_code == 2:
        print(
            f"No path found: {terrain_dir.name} / {scenario_id} / "
            f"{algorithm} / seed={seed}"
        )
        return None, planner_result

    if not path_output.exists():
        raise RuntimeError(f"Planner did not create path: {path_output}")

    return path_output, planner_result


def run_pybullet(
    terrain_dir,
    scenario,
    path_output,
    planner_result,
    algorithm,
    seed,
    experiment_dir,
    python_exe,
    alpha=None,
):
    scenario_id = scenario["scenario_id"]

    if algorithm == "rrt_star":
        execution_dir = (
            terrain_dir
            / "baseline"
            / scenario_id
            / "rrt_star"
            / f"seed_{seed:03d}"
        )
    else:
        execution_dir = (
            terrain_dir
            / "baseline"
            / scenario_id
            / "risk_weighted_rrt_star"
            / f"alpha_{alpha:.2f}"
            / f"seed_{seed:03d}"
        )

    result_files = list(execution_dir.glob("result_*.json"))
    if result_files:
        print(f"PyBullet already completed: {result_files[0]}")
        return

    # IMPORTANT:
    # This is the existing run.py interface. run.py itself is untouched.
    run_command(
        [
            python_exe,
            str(experiment_dir / RUN_SCRIPT),
            "--terrain-dir",
            str(terrain_dir),
            "--cmdp-path",
            str(path_output),
            "--scenario-id",
            scenario_id,
            "--risk-budget",
            str(BASELINE_RISK_BUDGET),
            "--seed",
            str(seed),
            "--output-dir",
            str(execution_dir),
            "--headless",
        ],
        cwd=experiment_dir,
    )


def run_terrain(terrain_dir, experiment_dir, python_exe):
    scenario_file = terrain_dir / "start_goal_scenarios.json"

    if not scenario_file.exists():
        raise FileNotFoundError(
            f"Missing fixed scenario file: {scenario_file}\n"
            "Generate it first with generate_scenarios.py."
        )

    scenarios = ensure_betas(
        terrain_dir,
        scenario_file,
        experiment_dir,
        python_exe,
    )

    for scenario in scenarios:
        beta = float(scenario["beta"])

        print("\n" + "#" * 80)
        print(
            f"TERRAIN={terrain_dir.name} | "
            f"SCENARIO={scenario['scenario_id']} | "
            f"START={scenario['start']} | GOAL={scenario['goal']} | "
            f"BETA={beta:.8f}"
        )
        print("#" * 80)

        # --------------------------------------------------------
        # Standard RRT*
        # --------------------------------------------------------
        for seed in range(1, NUM_MONTE_CARLO + 1):
            path_output, planner_result = run_planner(
                terrain_dir=terrain_dir,
                scenario=scenario,
                algorithm="rrt_star",
                seed=seed,
                experiment_dir=experiment_dir,
                python_exe=python_exe,
            )

            if path_output is not None:
                run_pybullet(
                    terrain_dir=terrain_dir,
                    scenario=scenario,
                    path_output=path_output,
                    planner_result=planner_result,
                    algorithm="rrt_star",
                    seed=seed,
                    experiment_dir=experiment_dir,
                    python_exe=python_exe,
                )

        # --------------------------------------------------------
        # Risk-weighted RRT*
        # --------------------------------------------------------
        for alpha in ALPHA_VALUES:
            for seed in range(1, NUM_MONTE_CARLO + 1):
                path_output, planner_result = run_planner(
                    terrain_dir=terrain_dir,
                    scenario=scenario,
                    algorithm="risk_weighted_rrt_star",
                    seed=seed,
                    alpha=alpha,
                    beta=beta,
                    experiment_dir=experiment_dir,
                    python_exe=python_exe,
                )

                if path_output is not None:
                    run_pybullet(
                        terrain_dir=terrain_dir,
                        scenario=scenario,
                        path_output=path_output,
                        planner_result=planner_result,
                        algorithm="risk_weighted_rrt_star",
                        seed=seed,
                        experiment_dir=experiment_dir,
                        python_exe=python_exe,
                        alpha=alpha,
                    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--terrain",
        default=None,
        help="Run one terrain, e.g. terrain_01. Omit to run all terrain_* folders.",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
    )

    args = parser.parse_args()

    experiment_dir = Path(__file__).resolve().parent
    experiments_dir = args.experiments_dir.resolve()

    if args.terrain:
        terrains = [experiments_dir / args.terrain]
    else:
        terrains = sorted(
            p for p in experiments_dir.glob("terrain_*")
            if p.is_dir()
        )

    if not terrains:
        raise RuntimeError(f"No terrain folders found in {experiments_dir}")

    for terrain_dir in terrains:
        if not (terrain_dir / "terrain_grid.json").exists():
            print(f"Skipping {terrain_dir}: no terrain_grid.json")
            continue

        run_terrain(
            terrain_dir=terrain_dir,
            experiment_dir=experiment_dir,
            python_exe=args.python,
        )


if __name__ == "__main__":
    main()

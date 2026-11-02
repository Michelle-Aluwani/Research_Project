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

Failure handling:
    One failed trial/condition never kills the terrain job. Failures are
    appended to <terrain_dir>/failures.jsonl and the runner carries on.
    A robot falling over is NOT a failure: run.py records it and exits 0.
    "CMDP cannot reach the goal" is also not a failure: cmdp_result.json
    records reached_goal=false and the Monte Carlo trials are skipped.

Stale data:
    If terrain_grid.json is missing or was produced by an older version of
    the pipeline (no/old metadata.schema_version), it is regenerated and the
    derived files (scenarios, betas, CMDP and baseline results) are MOVED
    into <terrain_dir>/_stale_<timestamp>/ rather than silently reused.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

D_VALUES = [3,4,5,6,7]
NUM_MONTE_CARLO = 5
SCENARIO_SEED = 42
SCENARIO_GENERATOR = "generate_scenarios.py"
RUN_SCRIPT = "run.py"
CMDP_SCRIPT = "cmdp_refactored.py"
BASELINE_RUNNER_SCRIPT = "baseline_runner.py"


def run_command(cmd, cwd):
    """Run a command and return its exit code. Never raises on failure."""
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 80)

    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    return result.returncode


def log_failure(terrain_dir, what, cmd=None, returncode=None):
    """Append one failure record and keep going."""
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": what,
        "returncode": returncode,
        "cmd": " ".join(str(x) for x in cmd) if cmd else None,
    }
    print(f"\n!!! FAILURE (continuing): {what} (exit code {returncode})")
    try:
        with (Path(terrain_dir) / "failures.jsonl").open(
            "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def find_terrain_obj(terrain_dir):
    """Find the single source OBJ in a terrain folder.

    Accepts newterrain.obj, newterrain_1.obj, ... Generated collision files
    (pybullet_*.obj, *_zup.obj) are ignored.
    """
    candidates = [
        p for p in sorted(terrain_dir.glob("*.obj"))
        if not p.name.startswith("pybullet_")
        and not p.stem.endswith("_zup")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .obj file found in {terrain_dir}")
    preferred = [p for p in candidates if p.name.startswith("newterrain")]
    if len(preferred) == 1:
        return preferred[0]
    raise RuntimeError(
        f"Several .obj files in {terrain_dir}: "
        f"{[p.name for p in candidates]}. Keep exactly one."
    )


def grid_is_current(json_path):
    """True when terrain_grid.json exists and has the current schema."""
    if not json_path.is_file():
        return False
    try:
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)["metadata"]
        from cmdp_refactored import TERRAIN_GRID_SCHEMA_VERSION
        return int(meta.get("schema_version", 0)) >= TERRAIN_GRID_SCHEMA_VERSION
    except Exception:
        return False


def quarantine_stale(terrain_dir):
    """Move files derived from an old terrain grid out of the way."""
    stale_names = [
        "terrain_grid.json",
        "start_goal_scenarios.json",
        "baseline",
        "failures.jsonl",
    ]
    # scenario folders: mixed_01, mud_02, water_01, hazard_01, long_02 ...
    for p in terrain_dir.iterdir():
        if p.is_dir() and p.name.rsplit("_", 1)[0] in (
            "mixed", "mud", "water", "hazard", "long"
        ):
            stale_names.append(p.name)

    existing = [n for n in stale_names if (terrain_dir / n).exists()]
    if not existing:
        return

    stale_dir = terrain_dir / f"_stale_{time.strftime('%Y%m%d_%H%M%S')}"
    stale_dir.mkdir(parents=True, exist_ok=True)
    for name in existing:
        shutil.move(str(terrain_dir / name), str(stale_dir / name))
    print(f"Moved stale files {existing} -> {stale_dir}")


def ensure_terrain_grid(terrain_dir, experiment_dir):
    """Generate terrain_grid.json from the OBJ when missing or stale."""
    json_path = terrain_dir / "terrain_grid.json"

    if str(experiment_dir) not in sys.path:
        sys.path.insert(0, str(experiment_dir))

    if grid_is_current(json_path):
        print(f"\nTerrain grid already exists: {json_path}")
        return

    obj_path = find_terrain_obj(terrain_dir)

    if json_path.exists():
        print(f"\nTerrain grid is from an older pipeline version: {json_path}")
    quarantine_stale(terrain_dir)

    from cmdp_refactored import generate_terrain_grid

    print(f"\nGenerating terrain grid from {obj_path.name}: {json_path}")
    generate_terrain_grid(
        obj_path=obj_path,
        json_path=json_path,
        resolution=0.5,
        samples_per_side=5,
        chunk_size=100_000,
    )
    if not json_path.is_file():
        raise RuntimeError(f"Terrain generator did not create {json_path}")
    print(f"Terrain grid generated: {json_path}")


def generate_scenarios(terrain_dir, python_exe, experiment_dir):
    scenario_file = terrain_dir / "start_goal_scenarios.json"

    if scenario_file.exists():
        print(f"\nScenarios already exist: {scenario_file}")
        print("Keeping the existing terrain-specific scenarios.")
        return

    # The generator lives next to this script, not next to the terrain
    # folders, so use its absolute path and run from the script's folder.
    cmd = [
        python_exe,
        str(experiment_dir / SCENARIO_GENERATOR),
        "--terrain-dir",
        str(terrain_dir),
        "--seed",
        str(SCENARIO_SEED),
    ]
    code = run_command(cmd, cwd=experiment_dir)
    if code != 0 or not scenario_file.exists():
        log_failure(terrain_dir, "scenario generation", cmd, code)


def load_scenarios(terrain_dir):
    path = terrain_dir / "start_goal_scenarios.json"

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = []

    # A category may contain 0, 1 or 2 scenarios; empty ones are skipped.
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

    if cmdp_result.exists():
        if cmdp_path.exists():
            print(f"\nCMDP already exists: {condition_dir}")
            return cmdp_path
        try:
            with cmdp_result.open("r", encoding="utf-8") as f:
                reached = json.load(f).get("reached_goal", False)
        except Exception:
            reached = None
        if reached is False:
            print(f"\nCMDP already solved (goal not reached): {condition_dir}")
            return None

    cmd = [
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
    ]
    code = run_command(cmd, cwd=experiment_dir)

    if code == 2:
        # CMDP finished but could not reach the goal: a valid outcome.
        print(f"CMDP found no path to the goal: {scenario_id} D={risk_budget:g}")
        return None

    if code != 0 or not cmdp_path.exists():
        log_failure(
            terrain_dir,
            f"CMDP {scenario_id} D={risk_budget:g}",
            cmd,
            code,
        )
        return None

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

        cmd = (
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
            ]
        )
        code = run_command(cmd, cwd=experiment_dir)
        if code != 0:
            log_failure(
                terrain_dir,
                f"PyBullet {scenario_id} D={risk_budget:g} seed={seed}",
                cmd,
                code,
            )


def run_baseline_runner(
    terrain_dir,
    experiment_dir,
    python_exe,
):
    """
    Run the complete RRT*/RW-RRT* baseline suite for this terrain
    immediately after all CMDP conditions for the terrain are complete.
    """
    cmd = [
        python_exe,
        str(experiment_dir / BASELINE_RUNNER_SCRIPT),
        "--experiments-dir",
        str(terrain_dir.parent),
        "--terrain",
        terrain_dir.name,
    ]
    code = run_command(cmd, cwd=experiment_dir)
    if code != 0:
        log_failure(terrain_dir, "baseline runner", cmd, code)


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete CMDP + PyBullet Monte Carlo experiment."
    )

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        required=True,
        help="Directory containing terrain_1, terrain_2, ...",
    )

    parser.add_argument(
        "--terrain",
        type=str,
        default=None,
        help="Run only one terrain, e.g. terrain_1.",
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

        try:
            ensure_terrain_grid(terrain_dir, experiment_dir)
        except Exception as exc:  # keep going with the next terrain
            log_failure(terrain_dir, f"terrain grid: {exc}")
            continue
        generate_scenarios(terrain_dir, args.python, experiment_dir)
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

                if cmdp_path is None:
                    print("No CMDP path for this condition; "
                          "skipping its Monte Carlo trials.")
                    continue

                run_monte_carlo(
                    terrain_dir=terrain_dir,
                    scenario_id=scenario_id,
                    risk_budget=D,
                    cmdp_path=cmdp_path,
                    experiment_dir=experiment_dir,
                    python_exe=args.python,
                )

                total_trials += NUM_MONTE_CARLO

        # CMDP for this terrain is now complete.
        # Only after ALL scenarios and D values have finished,
        # run the RRT*/RW-RRT* baseline suite for this terrain.
        print("\n" + "#" * 80)
        print(f"CMDP COMPLETE FOR TERRAIN: {terrain_dir.name}")
        print(f"STARTING BASELINE RUNNER: {BASELINE_RUNNER_SCRIPT}")
        print("#" * 80)

        run_baseline_runner(
            terrain_dir=terrain_dir,
            experiment_dir=experiment_dir,
            python_exe=args.python,
        )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"CMDP conditions processed : {total_conditions}")
    print(f"PyBullet trials processed : {total_trials}")
    print("=" * 80)


if __name__ == "__main__":
    main()

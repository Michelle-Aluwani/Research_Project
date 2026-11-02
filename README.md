# CMDP vs. RRT* Path Planning on Terrain, with PyBullet Laikago Execution

This repository plans paths over terrain meshes with a **Constrained MDP (CMDP)** planner at several risk budgets (`D`) and compares them against **Standard RRT\*** and **Risk-weighted RRT\*** baselines. Every planned path is then executed physically by a Laikago quadruped in **PyBullet** over multiple Monte Carlo seeds.

> **Status:** this README covers the experiment runners and orchestration scripts. Sections marked **TODO** depend on files (`run.py`, `cmdp_refactored.py`, `rrt_baseline.py`, `generate_scenarios.py`) that will be documented in detail in a later update. The results-collection and plotting scripts are documented in [Collecting and analysing results](#collecting-and-analysing-results).

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Project layout](#project-layout)
3. [Script reference](#script-reference)
4. [Quick start](#quick-start)
5. [Recipes: common ways to run](#recipes-common-ways-to-run)
6. [Output folder structure](#output-folder-structure)
7. [Configuration constants](#configuration-constants)
8. [Failure handling and re-running](#failure-handling-and-re-running)
9. [Running a single PyBullet trial by hand](#running-a-single-pybullet-trial-by-hand)
10. [Collecting and analysing results](#collecting-and-analysing-results)
11. [Calibration / debug runs](#calibration--debug-runs)
12. [Troubleshooting](#troubleshooting)

---

## Pipeline overview

For each terrain folder (`terrain_1`, `terrain_2`, ...):

```text
terrain .obj
   │
   ▼
1. terrain_grid.json            (cmdp_refactored.generate_terrain_grid, 0.5 resolution, 5 samples/side)
   │
   ▼
2. start_goal_scenarios.json    (generate_scenarios.py, seed 42; created once and then kept)
   │
   ├──────────────► 3a. CMDP planning   for each scenario × each D
   │                     → <scenario_id>/D<D>/cmdp_path_world.csv + cmdp_result.json
   │
   └──────────────► 3b. Baseline planning (rrt_baseline.py)
                         • Standard RRT*
                         • Risk-weighted RRT* for each alpha  (beta computed once per scenario)
                         → baseline/<scenario_id>/.../seed_XXX/cmdp_path_world.csv + planner_result.json
   │
   ▼
4. PyBullet execution (run.py, headless) of every saved path, for each Monte Carlo seed
     → result_*.json, pybullet_execution.csv, ... in each seed folder
   │
   ▼
5. Result collection (collect_*.py): per-terrain CSVs, accumulated-risk CSVs, diagnostics JSON, ZIP archives
```

Key design points:

- **Same scenarios for everyone.** CMDP and both baselines use the same terrain-specific start/goal pairs from `start_goal_scenarios.json`. Existing start/goal pairs are never overwritten.
- **Baseline paths are saved as `cmdp_path_world.csv`.** This is deliberate so `run.py` can accept any planner's path through `--cmdp-path`. The filename does **not** mean the path came from CMDP.
- **Same seed for planner and simulator** (baselines): the seed that generated an RRT\* path is also passed to `run.py`.
- **Beta** (for Risk-weighted RRT\*) is calculated once per terrain/scenario from the fixed start→goal edge and written into `start_goal_scenarios.json`; it is shared by all alphas and seeds.
- **A robot falling is not a failure.** `run.py` records the fall in its result JSON and exits `0`. "Planner found no path" is also not a failure (exit code `2`).

---

## Project layout

Run everything from this directory. The terrain folders live **next to the scripts**, so `--experiments-dir .` points at them.

```text
project/
├── README.md
├── run.py                              # single PyBullet execution of one path (TODO: document flags)
├── cmdp_refactored.py                  # terrain grid generation + CMDP solver
├── rrt_baseline.py                     # Standard RRT* / Risk-weighted RRT* planner
├── generate_scenarios.py               # start/goal scenario generator
│
├── experiment_runner.py                # full sequential pipeline (planning + PyBullet)
├── baseline_runner.py                  # sequential RRT* planning + PyBullet
│
├── experiment_runner_no_pybullet.py    # all planning (CMDP + RRT*), parallel, no PyBullet
├── cmdp_parallel_runner.py             # CMDP planning only, parallel
├── experiment_runner_no_pybullet_CMDP_only.py  # CMDP planning only, strict (no generation)
├── baseline_parallel_runner.py         # RRT* planning only, parallel
├── pybullet_parallel_runner.py         # PyBullet only, CMDP + baselines, parallel
├── pybullet_parallel_runner_CMDP_only.py       # PyBullet only, CMDP paths, parallel
│
├── collect_cmdp_results.py             # -> terrain_X/cmdp_results.csv
├── collect_baseline_planner_results.py # -> terrain_X/baseline_planner_results.csv
├── collect_pybullet_results.py         # -> terrain_X/pybullet_results.csv
├── collect_pybullet_risk.py            # -> terrain_X/pybullet_accumulated_risk.csv
├── collect_all_terrain_csvs.py         # gathers per-terrain CSVs -> COLLECTED_TERRAIN_CSVS/
├── collect_results_per_terrain.py      # per-terrain ZIPs of JSON-converted results -> cmdp/, baseline/
├── collect_diagnostics.py              # tracking diagnostics -> diagnostics/*.json
├── plot_tracking.py                    # route-vs-actual plot for one run
│
├── terrain_1/                          # one folder per terrain
│   └── newterrain*.obj                 # exactly one source .obj per terrain
├── terrain_2/
├── ...
└── PYBULLET_CALIBRATION/               # controller debug/calibration logs (see its own README)
```

Each terrain folder must contain **exactly one** source `.obj` (names like `newterrain.obj` or `newterrain_1.obj`). Generated files (`pybullet_*.obj`, `*_zup.obj`) are ignored when locating it.

Several scripts import each other, so **keep all scripts in the same folder**:

- `experiment_runner.py` is imported by `cmdp_parallel_runner.py`, `pybullet_parallel_runner.py`, and both `*_CMDP_only.py` runners.
- `baseline_runner.py` is imported by `baseline_parallel_runner.py` and `pybullet_parallel_runner.py`.
- `experiment_runner_no_pybullet.py` launches `cmdp_parallel_runner.py` and `baseline_parallel_runner.py`.

---

## Script reference

### Planning + simulation (sequential, original runners)

| Script | What it does |
|---|---|
| `experiment_runner.py` | Full pipeline per terrain: ensure grid → ensure scenarios → CMDP for each `D` → PyBullet for each seed → then launches `baseline_runner.py` for that terrain. |
| `baseline_runner.py` | For one/all terrains: computes missing betas, runs RRT\* and Risk-weighted RRT\* planners, then PyBullet for every resulting path. |

### Planning only (no PyBullet)

| Script | What it does | Generates grid/scenarios? |
|---|---|---|
| `experiment_runner_no_pybullet.py` | Per terrain runs `cmdp_parallel_runner.py`, then `baseline_parallel_runner.py`. Stops that terrain's remaining stages if a stage fails. | Yes (via CMDP stage) |
| `cmdp_parallel_runner.py` | CMDP planning for every scenario × `D`, in parallel. Skips conditions already solved. | **Yes**, if missing |
| `experiment_runner_no_pybullet_CMDP_only.py` | CMDP planning only. Preserves existing results, including previously failed plans. | **No.** Skips terrains missing grid/scenarios |
| `baseline_parallel_runner.py` | RRT\* + Risk-weighted RRT\* planning only, in parallel. Writes betas before launching jobs. | No. Requires grid + scenarios |

### PyBullet only (no planning)

| Script | What it does |
|---|---|
| `pybullet_parallel_runner.py` | Simulates saved CMDP **and** baseline paths. Seeds, `D`, alpha, and scenarios selectable per run. |
| `pybullet_parallel_runner_CMDP_only.py` | Simulates saved CMDP paths only, for `experiment_runner.D_VALUES`. Never touches baselines. |

### Common flags

| Flag | Applies to | Meaning |
|---|---|---|
| `--experiments-dir DIR` | all runners | Folder containing `terrain_*` folders. Use `.` when terrains sit beside the scripts. |
| `--terrain NAME` | all runners | Select a terrain, e.g. `terrain_1`. Repeatable on the parallel runners; single value on `experiment_runner.py` and `baseline_runner.py`. Omit to run all `terrain_*`. |
| `--workers N` | parallel runners | Parallel jobs (1–32, default 2). |
| `--python EXE` | all runners | Python executable for child processes (defaults to the current one). |

### Extra flags for `pybullet_parallel_runner.py`

| Flag | Meaning |
|---|---|
| `--planner {all,cmdp,baseline}` | Which planner's paths to simulate (default `all`). |
| `--cmdp-seeds N` | CMDP seeds `1..N` (default: `experiment_runner.NUM_MONTE_CARLO`). |
| `--baseline-seeds N` | RRT\* seeds `1..N` (default: `baseline_runner.NUM_MONTE_CARLO`). |
| `--d-values D [D ...]` | CMDP risk budgets to simulate, e.g. `4 6 7`. |
| `--alphas A [A ...]` | Risk-weighted RRT\* alphas to simulate, e.g. `0.75 0.5`. Standard RRT\* is always included. |
| `--scenario-ids ID [ID ...]` | Only these scenarios, e.g. `mud_06 mud_07 hazard_06`. |
| `--extra "FLAGS"` | Extra flags appended to every `run.py` call, e.g. `"--reverse-deg 120"`. |
| `--rerun` | Re-simulate even if a `result_*.json` already exists. |

### Extra flags for `pybullet_parallel_runner_CMDP_only.py`

| Flag | Meaning |
|---|---|
| `--seeds N` | Execution seeds `1..N` (default 3). |
| `--extra "FLAGS"` | Extra flags passed to `run.py`. |
| `--rerun` | Repeat finished runs (overwrites results in those seed folders). |

---

## Quick start

```bash
# from the project directory (terrain_* folders are right here)
cd project

# 1. Plan everything (CMDP + RRT* baselines), no simulation
python experiment_runner_no_pybullet.py --experiments-dir . --workers 4

# 2. Simulate all saved paths in PyBullet
python pybullet_parallel_runner.py --experiments-dir . --workers 4

# 3. Inspect results
ls terrain_1/*/D*/seed_001/result_*.json
```

Or run one terrain end-to-end in one go (sequential, slower):

```bash
python experiment_runner.py --experiments-dir . --terrain terrain_1
```

---

## Recipes: common ways to run

### Full pipeline

```bash
# Everything, sequential: grid -> scenarios -> CMDP -> PyBullet -> baselines (all terrains)
python experiment_runner.py --experiments-dir .

# One terrain only
python experiment_runner.py --experiments-dir . --terrain terrain_1
```

### Planning only (parallel)

```bash
# CMDP + RRT* baselines for all terrains
python experiment_runner_no_pybullet.py --experiments-dir . --workers 4

# Specific terrains
python experiment_runner_no_pybullet.py --experiments-dir . --terrain terrain_1 --terrain terrain_2 --workers 4

# CMDP only (generates grid/scenarios if missing)
python cmdp_parallel_runner.py --experiments-dir . --terrain terrain_1 --workers 4

# CMDP only, strict: reuse existing grid/scenarios, generate nothing
python experiment_runner_no_pybullet_CMDP_only.py --experiments-dir . --terrain terrain_1 --workers 4

# RRT* + Risk-weighted RRT* only (needs terrain_grid.json and start_goal_scenarios.json)
python baseline_parallel_runner.py --experiments-dir . --terrain terrain_1 --workers 4
```

### PyBullet only (parallel)

```bash
# Everything saved: CMDP + all baselines
python pybullet_parallel_runner.py --experiments-dir . --workers 4

# CMDP paths only
python pybullet_parallel_runner.py --experiments-dir . --planner cmdp --workers 4

# Baselines only
python pybullet_parallel_runner.py --experiments-dir . --planner baseline --workers 4

# Fine-grained selection: 5 CMDP seeds, 3 baseline seeds, chosen D/alpha/scenarios
python pybullet_parallel_runner.py --experiments-dir . --terrain terrain_1 \
    --cmdp-seeds 5 --baseline-seeds 3 \
    --d-values 4 6 7 --alphas 0.75 0.5 \
    --scenario-ids mud_06 mud_07 hazard_06

# Pass an extra flag to every run.py call
python pybullet_parallel_runner.py --experiments-dir . --extra "--reverse-deg 120"

# Re-simulate runs that already have a result_*.json
python pybullet_parallel_runner.py --experiments-dir . --terrain terrain_1 --rerun

# CMDP-only simulation with 10 seeds
python pybullet_parallel_runner_CMDP_only.py --experiments-dir . --seeds 10 --workers 4
```

### Baselines, sequential (plan + simulate together)

```bash
python baseline_runner.py --experiments-dir . --terrain terrain_1
```

> **Worker count tip:** each PyBullet job is single-threaded (the parallel runners set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, etc. to `1`). Set `--workers` to roughly the number of physical CPU cores you can spare.

---

## Output folder structure

```text
terrain_1/
├── newterrain.obj
├── terrain_grid.json                    # terrain grid / hazard data (schema-versioned)
├── start_goal_scenarios.json            # fixed start/goal pairs (+ beta per scenario)
├── failures.jsonl                       # one JSON line per real failure (see below)
├── _stale_<timestamp>/                  # quarantined files from an older pipeline version
│
├── mud_01/                              # <scenario_id>: mixed_*, mud_*, water_*, hazard_*, long_*
│   └── D4/                              # one folder per CMDP risk budget
│       ├── cmdp_result.json             # records reached_goal true/false
│       ├── cmdp_path_world.csv          # the planned path (absent if goal unreachable)
│       └── seed_001/ ... seed_00N/      # one PyBullet run per seed
│           ├── result_*.json
│           ├── pybullet_execution.csv
│           ├── path_tracking_diagnostics.csv   (depending on controller version)
│           └── stumble_log.csv
│
└── baseline/
    └── mud_01/
        ├── rrt_star/
        │   └── seed_001/
        │       ├── cmdp_path_world.csv  # RRT* path (name kept for run.py compatibility)
        │       ├── planner_result.json
        │       └── result_*.json, pybullet_*.csv, ...
        └── risk_weighted_rrt_star/
            └── alpha_0.50/
                └── seed_001/ ...
```

Per-run summary files (`result_*.json`) record the scenario, risk budget, seed, reached-goal / fall / timeout status, distance to goal, waypoints reached, and related fields. See the [calibration README](PYBULLET_CALIBRATION/README.md) for descriptions of the other CSVs.

---

## Configuration constants

Defaults live at the top of the runner scripts. **Edit them there** (the parallel runners import them), and note that some module docstrings still mention older values.

| Constant | File | Current value | Meaning |
|---|---|---|---|
| `D_VALUES` | `experiment_runner.py` | `[4, 6, 7]` | CMDP risk budgets planned and simulated. |
| `NUM_MONTE_CARLO` | `experiment_runner.py` | `5` | Default CMDP execution seeds. |
| `SCENARIO_SEED` | `experiment_runner.py` | `42` | Seed for scenario generation. |
| `ALPHA_VALUES` | `baseline_runner.py` | `[0.50, 0.75]` | Risk-weighted RRT\* alphas (Standard RRT\* is always run too). |
| `NUM_MONTE_CARLO` | `baseline_runner.py` | `3` | Default baseline seeds. |
| `BASELINE_RISK_BUDGET` | `baseline_runner.py` | `0.0` | Compatibility field passed to `run.py` for baseline runs; does not alter physics. |
| `--seeds` default | `pybullet_parallel_runner_CMDP_only.py` | `3` | CMDP-only simulator default seeds. |
| Terrain grid | `experiment_runner.py` | resolution `0.5`, `5` samples/side | Used by `generate_terrain_grid`. |

---

## Failure handling and re-running

- **Real failures** (crashed process, missing output) are appended to `<terrain>/failures.jsonl` and the runner moves on. Look there first if something is missing.
- **Exit code `2`** from a planner means *no path found*. This is a valid outcome, logged as `NO_PATH`, not a failure. CMDP records `reached_goal: false` in `cmdp_result.json`, and its Monte Carlo trials are skipped.
- **Skipping finished work:** the runners skip any condition that already has results:
  - CMDP: `cmdp_result.json` exists (and the path file exists if `reached_goal` is true).
  - Baselines: `planner_result.json` exists with a path (or a recorded `path_found: false`).
  - PyBullet: a `result_*.json` exists in the seed folder.
- **Forcing a re-run:** use `--rerun` on the PyBullet runners, or delete the relevant result files/folders. There is no `--rerun` for planning, so delete `cmdp_result.json` / `planner_result.json` to replan.
- **Stale data protection:** if `terrain_grid.json` is missing or from an older schema, `experiment_runner.py` / `cmdp_parallel_runner.py` regenerate it and **move** the derived files (scenarios, betas, CMDP and baseline results, scenario folders, `failures.jsonl`) into `_stale_<timestamp>/` instead of reusing them. `baseline_runner.py` instead skips such terrains and tells you to run `experiment_runner.py` first.

---

## Running a single PyBullet trial by hand

`run.py` executes one path for one seed:

```bash
python run.py \
  --terrain-dir terrain_1 \
  --cmdp-path terrain_1/mud_01/D4/cmdp_path_world.csv \
  --scenario-id mud_01 \
  --risk-budget 4 \
  --seed 1 \
  --output-dir terrain_1/mud_01/D4/seed_001 \
  [--headless]
```

- `--headless` uses PyBullet **DIRECT** mode (no GUI, no waiting at the end). Omit it to watch the run in a window.
- For a baseline path, pass its `cmdp_path_world.csv` and `--risk-budget 0`.
- Any extra flags supported by `run.py` can be passed through the batch runners with `--extra "..."`.

**TODO:** full `run.py` flag list and controller parameters (to be added after `run.py` is reviewed).

---

## Collecting and analysing results

These scripts read the experiment folders and **never modify the original run files**. Except for `collect_diagnostics.py` and `plot_tracking.py`, they take **no command-line arguments**: place them in the project directory (beside `terrain_*`) and run them with `python <script>.py`. They scan every folder named `terrain_<number>`.

### Recommended order

```bash
python collect_cmdp_results.py               # CMDP planning results
python collect_baseline_planner_results.py   # RRT* / RW-RRT* planning results
python collect_pybullet_results.py           # PyBullet execution results (CMDP + baselines)
python collect_pybullet_risk.py              # risk actually accumulated during execution
python collect_all_terrain_csvs.py           # copy the per-terrain CSVs into one folder
```

Optional extras:

```bash
python collect_results_per_terrain.py        # ZIPs of everything, converted to JSON
python collect_diagnostics.py                # per-run tracking-diagnostics statistics
python plot_tracking.py --csv <diag.csv> --route <cmdp_path_world.csv> --out tracking.png
```

### Script reference

| Script | Reads | Writes |
|---|---|---|
| `collect_cmdp_results.py` | every `cmdp_result.json` (excluding `baseline/`) | `terrain_X/cmdp_results.csv` (one row per scenario × D; JSON fields flattened) |
| `collect_baseline_planner_results.py` | `planner_result.json` / `planner_results.json` under `terrain_X/baseline/` | `terrain_X/baseline_planner_results.csv` (planner, scenario, scenario_type, D, alpha, seed + all flattened JSON fields) |
| `collect_pybullet_results.py` | `result_*.json` (and similar) anywhere in a terrain, CMDP and baseline | `terrain_X/pybullet_results.csv` (planner, D, scenario, scenario_type, seed + flattened fields, `source`) |
| `collect_pybullet_risk.py` | every `pybullet_execution.csv` (needs `actual_x`, `actual_y`, `actual_z`) + `terrain_grid.json` | `terrain_X/pybullet_accumulated_risk.csv`, plus copies in `COLLECTED_PYBULLET_RISK/` |
| `collect_all_terrain_csvs.py` | `cmdp_results.csv`, `baseline_planner_results.csv`, `pybullet_results.csv` in each terrain | `COLLECTED_TERRAIN_CSVS/terrain_X_<name>.csv` (copies; reruns overwrite) |
| `collect_results_per_terrain.py` | all `.csv/.tsv/.json/.jsonl` in each scenario folder and in `baseline/` | `cmdp/terrain_X_cmdp_results.zip`, `baseline/terrain_X_baseline_results.zip` (each file converted to JSON) |
| `collect_diagnostics.py` | every `path_tracking_diagnostics.csv` | `diagnostics/<run path>_diagnostics.json`, `processing_report.json`, `diagnostics_summary.csv` |
| `plot_tracking.py` | one diagnostics CSV + the route CSV | a PNG (default `tracking.png`) |

Planner, scenario, D, alpha, and seed are inferred from the **folder names** (e.g. `mud_01/D4/seed_001`, `risk_weighted_rrt_star/alpha_0.50/seed_001`), so don't rename those folders.

### Accumulated risk of the executed path (`collect_pybullet_risk.py`)

Uses the **same risk model as CMDP** (`load_terrain_grid` / `prepare_risk_model` from `cmdp_refactored.py`), so `cmdp_refactored.py` must be in the project directory. For each `pybullet_execution.csv` it walks the robot's *actual* trajectory (interpolated at a quarter of the grid resolution so no cell is skipped) and adds a cell's risk **once each time the robot enters it**. Folders starting with `_stale_` are skipped.

Main output columns: `terrain, planner, scenario, D, alpha, seed, run_id`, sample counts, start/end position, `actual_distance_2d_m`, `actual_distance_3d_m`, `risk_cell_visits`, `unique_grid_cells`, `unknown_risk_visits`, `risk_complete`, `accumulated_risk`, `partial_accumulated_risk`, `risk_remaining`, `source`.

Settings at the top of the script:

| Setting | Default | Meaning |
|---|---|---|
| `RISK_MODEL_FOLDER` | project dir | Where `cmdp_refactored.py` lives. |
| `OUTPUT_FOLDER` | `COLLECTED_PYBULLET_RISK/` | Collected copies. |
| `WRITE_POINT_DETAILS` | `False` | `True` also writes `pybullet_risk_trace.csv` (risk at every logged position). **Can be very large.** |

### Tracking diagnostics (`collect_diagnostics.py`)

```bash
python collect_diagnostics.py                                   # scans the script's folder
python collect_diagnostics.py --root "C:/Research" --output "C:/Research/diagnostics"
python collect_diagnostics.py --include-rows                    # also embed every sample (larger files)
```

Per run it computes coverage (rows, duration, sampling interval, modes), sampled 2D travel distance, and statistics (mean, median, min/max, std, p90/p95/p99, RMSE; signed and absolute) for route error, local route error, heading error, pitch, roll, yaw rate, steering, and target change, plus waypoint/progress/recovery summaries.

Caveats built into the output:
- A short recording is **not proof of a fall**. `classification.fell` and `mission_success` are left `null`; confirm from `result_*.json`.
- `route_error_m` is the controller-reported deviation, not an independent geometric error.
- Distances sampled from diagnostic timestamps may underestimate the full-resolution trajectory.
- The script **refuses to overwrite** an existing output file (it reports a filename conflict). Delete the `diagnostics/` folder or pass a new `--output` before re-running.

### Plotting a run (`plot_tracking.py`)

```bash
python plot_tracking.py \
  --csv   terrain_1/mud_01/D4/seed_001/path_tracking_diagnostics.csv \
  --route terrain_1/mud_01/D4/cmdp_path_world.csv \
  --out   tracking.png
```

Left panel: planned route vs actual path, with heading arrows, look-ahead targets, and the first point where local route error exceeds 0.85 m. Right panel: local error, heading error, steering, and yaw rate over time. Requires `matplotlib` and the diagnostics columns `time_s, robot_x, robot_y, look_x, look_y, local_route_error_m, heading_error_deg, steering, yaw_rate_deg_s` (and optionally `body_yaw_deg`).

### Things to watch

- `collect_pybullet_results.py` searches the whole terrain folder recursively, so **old `_stale_*` folders may be picked up**. Move or delete them first if you don't want them in the CSV.
- `collect_all_terrain_csvs.py` copies only `cmdp_results.csv`, `baseline_planner_results.csv`, and `pybullet_results.csv`. The accumulated-risk CSV is collected separately by `collect_pybullet_risk.py` into `COLLECTED_PYBULLET_RISK/`.
- Run the collectors again after new planning or simulation runs; each rerun overwrites the previous CSVs.
- Baseline PyBullet rows have D recorded as `0` because `run.py` is called with `--risk-budget 0`; use `planner` and `alpha` to identify them.

---

## Calibration / debug runs

The `PYBULLET_CALIBRATION/` folder holds exploratory logs from tuning the Laikago controller (gait, steering, waypoint/segment tracking, pivoting, corner handling, long routes). These are **debug evidence, not final results**: mostly one seed and one scenario per folder, and many runs fall or time out. Keep them separate from the final experiment outputs used in any report.

See [`PYBULLET_CALIBRATION/README.md`](PYBULLET_CALIBRATION/README.md) for the folder-by-folder map and interpretation notes.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Missing fixed scenario file ... Generate it first` | Run `experiment_runner.py` or `cmdp_parallel_runner.py` first (they create scenarios), or run `generate_scenarios.py --terrain-dir <terrain> --seed 42`. |
| `Missing terrain_grid.json or start_goal_scenarios.json; run CMDP planning first.` | `baseline_parallel_runner.py` does not generate these. Run CMDP planning first. |
| `Skipping ...: terrain_grid.json is from an older pipeline version` | Run `experiment_runner.py` (or `cmdp_parallel_runner.py`) for that terrain to regenerate it. |
| `Several .obj files in <terrain>` | Keep exactly one source `.obj` per terrain folder (generated `pybullet_*.obj` / `*_zup.obj` are fine). |
| `WARNING: D=... is not in experiment_runner.D_VALUES` | The requested `D` isn't in the configured list. It still runs if a path exists, but check for a typo. |
| `WARNING: scenario ... not found` | The `--scenario-ids` value doesn't exist in that terrain's `start_goal_scenarios.json`. |
| `No module named experiment_runner` / `baseline_runner` | Run from the project directory with all scripts in one folder. |
| Simulations seem skipped | A `result_*.json` already exists; use `--rerun`. |
| Parallel run is slow or thrashing | Lower `--workers`; each job already uses a single thread. |
| Something silently missing | Check `<terrain>/failures.jsonl`. |
| `collect_pybullet_risk.py`: `No module named cmdp_refactored` | Keep it beside `cmdp_refactored.py`, or set `RISK_MODEL_FOLDER`. |
| `collect_diagnostics.py`: `Output filename conflict` | The output folder already has files; delete `diagnostics/` or use `--output`. |
| Empty collected CSVs / "No ... results found" | The matching runs haven't finished, or the folder names were changed. |

---

## TODO (next README update)

- [ ] Document `run.py` flags and controller parameters
- [ ] Document `cmdp_refactored.py` (state/risk model, solver backends, `--backend auto`)
- [ ] Document `rrt_baseline.py` (RRT\* edge-distance/hazard definitions, alpha/beta)
- [ ] Document `generate_scenarios.py` (scenario categories and selection rules)
- [ ] Add dependencies / `requirements.txt` and tested Python version

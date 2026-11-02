# Laikago / PyBullet Debug and Calibration Runs

This archive contains exploratory physical-execution tests for the planned CMDP paths in PyBullet. It is best treated as **controller calibration and debugging evidence**, not as the complete set of final evaluation experiments.

## What the experiments were investigating

The folder names and the saved execution/diagnostic logs indicate the following sequence of questions:

1. **Can the Laikago gait move reliably in the intended travel direction?**  
   Early runs (`rotate_test`, `steering_test`, `steer_f0_positive`, `hip_test`) investigate orientation, steering response, and gait/actuator behavior. Several runs terminate early from loss of footing or pitch/roll threshold violations.

2. **Can the robot follow the planner's waypoint path?**  
   `original_tracking`, `wp_test`, and `segment_tracking` compare/diagnose waypoint-following versus segment/projection-style route tracking. The diagnostic CSVs record route error, heading error, steering, yaw rate, body attitude, and progress.

3. **Can turns and corners be handled without losing route progress?**  
   `pivot_test`, `pivot_test2`, `corner_safe_test`, `fixed_test`, and `fixed_test2` are corner/turn-related experiments. The available diagnostics identify waypoint-following/corner-fix modes; the `corner_safe_test` folder contains terrain assets only, so its actual run outcome is not preserved in this archive.

4. **Does changing the turning strategy improve heading control?**  
   `differential_test` is explicitly labelled as a differential-turn test. The diagnostics show an `original` tracking mode, while the saved result records a fall after wall contact. This is evidence of a tested variant, not evidence that it solved the issue.

5. **Can a longer or different terrain route be completed?**  
   `terrain4_6stuck` is a Terrain 4 long-route run. Its result records a fall before reaching the goal. `mud_01_D5_reverse` is a separate Mud scenario run; it did not fall, but it also did not record goal completion (126/127 waypoints reached).

6. **Which configuration should be retained as a candidate?**  
   `final_a` and `final_b` are labelled as candidate/final iterations. Their result files show similar outcomes: both fell before reaching the goal, after contacting a wall. The folder names alone do not establish which configuration was intended to be the final selected one, nor what exact parameter changed between them.

## Folder-by-folder map

| Folder | Likely test / purpose | What is actually saved |
|---|---|---|
| `hazard_01_D3_seed1` | Initial/reference Hazard scenario execution | Execution CSV, stumble log, result JSON; no tracking-diagnostics CSV |
| `original_tracking` | Baseline/original waypoint tracking behavior | Execution, tracking diagnostics, stumble log, result JSON |
| `wp_test` | Waypoint-following test/diagnostic run | Execution, tracking diagnostics, stumble log, result JSON |
| `segment_tracking` | Segment/projection-based tracking variant | Diagnostics label mode `segment`; execution and result logs |
| `rotate_test` | Rotation/pivot behavior during tracking | Diagnostics include pivot-active and pivot-entry fields; execution and result logs |
| `steering_test` | Steering actuator/command response test | Execution and result logs; no tracking-diagnostics CSV |
| `steer_f0_positive` | Fixed positive steering-command test | Diagnostics show steering fixed at `0.5`; execution and result logs |
| `hip_test` | Gait/hip-actuation or gait stability experiment | Corner-fix tracking diagnostics plus execution/result logs; exact parameter change is not documented in the folder |
| `pivot_test` | Pivot/turn strategy iteration | Execution and tracking diagnostics; fall recorded |
| `pivot_test2` | Follow-up pivot/turn iteration | Execution and tracking diagnostics; timed out without falling, but did not reach goal |
| `corner_safe_test` | Corner-safe controller/route iteration | Terrain meshes/material only; no execution result or diagnostics in this folder |
| `fixed_test` | Fixed/corner-fix tracking variant | Execution and diagnostics; fall recorded |
| `fixed_test2` | Follow-up fixed/corner-fix variant | Execution and diagnostics; fall recorded early |
| `differential_test` | Differential turning variant | Execution and diagnostics; result records wall contact and fall |
| `final_a` | Candidate configuration A | Full logs; fall recorded after wall contact |
| `final_b` | Candidate configuration B | Full logs; fall recorded after wall contact |
| `mud_01_D5_reverse` | Mud scenario, reverse-orientation/travel-direction test | Full logs; no fall, 126/127 waypoints reached, but goal not recorded as reached |
| `terrain4_6stuck` | Long route on Terrain 4; investigate progress/stuck/fall behavior | Full logs; fall recorded before goal |

## How to read the output files

- `pybullet_execution.csv` — time-series physical execution data (robot state and execution trajectory).
- `path_tracking_diagnostics.csv` — route-following diagnostics such as tracking mode, waypoint/segment progress, route error, heading error, steering, yaw rate, and body attitude. Column sets vary between controller versions.
- `stumble_log.csv` — logged stumble/slip events; an empty or absent log does not mean the controller was never tested.
- `result_*.json` — per-run summary, including scenario, risk budget, seed, reached-goal/fall/timeout status, fall time/reason, distance to goal, waypoints reached, body attitude, and wall-contact information.
- `pybullet_*_collision.obj` — generated collision meshes used by the PyBullet terrain setup.
- `terrain_visual_zup.obj` and `newterrain_1.mtl` — visual terrain export/material assets.
- `tracking.png` — saved visualization in `fixed_test`.

## Important interpretation notes

- These are **debug/calibration runs**, often one seed and one scenario per folder. Do not treat them as statistically comparable experimental results.
- The saved `result_*.json` files show that most Hazard-path iterations did **not** reach the goal. Some terminate from falls, and `pivot_test2` timed out after wall contact. The Mud run avoided a fall but did not formally reach the goal.
- `final_a` and `final_b` are names, not proof of a final validated configuration. Keep the exact source script/configuration and command line alongside whichever run is selected.
- Several folders contain generated terrain meshes copied alongside the logs. These are useful for reproducibility but can make the repository unnecessarily large if duplicated.
- The archive contains logs for multiple controller iterations, but not all corresponding source scripts/config snapshots. Where a folder's exact parameter change is not documented, this README describes the apparent test purpose without claiming a specific changed value.
- The available `run_debug_corner_safe.py` script documents a PyBullet execution setup with waypoint/segment/corner-safe tracking options, differential-turn options, and gait/controller parameters. Its settings should only be attributed to a run if that script version is confirmed to have produced that run's files.

## Suggested GitHub organization

```text
laikago-pybullet/
├── README.md
├── scripts/
│   ├── run_debug_corner_safe.py
│   └── [other versioned controller scripts]
├── calibration/
│   ├── original_tracking/
│   ├── segment_tracking/
│   ├── pivot_test/
│   └── ...
├── selected_candidate/
│   ├── source_script.py
│   ├── command.txt
│   └── logs/
└── terrain_assets/
    └── [shared meshes/materials, stored once where possible]
```

Recommended practice:
- Keep the scripts, README, run commands, result JSONs, and key CSV logs in GitHub.
- Store each run's exact script version or commit hash and command line.
- Avoid duplicating identical terrain OBJ/MTL files in every run folder; keep shared assets once and document the reference.
- If mesh/log files make the repository too large, use Git LFS or a GitHub Release for bulky artifacts.
- Keep calibration/debugging runs separate from the final experiment outputs used in the report.

## Short summary

The archive documents iterative attempts to make Laikago execute CMDP-generated paths in PyBullet, focusing on gait/orientation, steering, waypoint/segment tracking, turning/pivoting, corner handling, and longer-route execution. It is useful as implementation and calibration evidence. It does **not**, by itself, establish successful physical execution across the report's full experimental set.


## Re-running a calibration/debug test

If someone wants to reproduce one of these tests, they should adapt the command below to the selected run's terrain, CMDP path, scenario ID, risk budget, seed, and output directory:

```bash
python run.py --terrain-dir <terrain_dir> --cmdp-path <path.csv> \
  --scenario-id <id> --risk-budget <D> --seed <n> \
  --output-dir <dir> [--headless]
```

**Run it from the repository root when possible.** If the terminal is currently inside `PYBULLET_CALIBRATION/` but the `run.py` entry point and supporting run files are in the parent directory, use `../run.py` (and adjust the other paths so they resolve from the current working directory), for example:

```bash
python ../run.py --terrain-dir ../<terrain_dir> --cmdp-path ../<path.csv> \
  --scenario-id <id> --risk-budget <D> --seed <n> \
  --output-dir ../<dir> --headless
```

The `../` prefix means “go up one directory.” If the entry point or data files are located elsewhere, change the relative paths accordingly; the command is a template, not a guaranteed copy-paste command for every folder. Use the matching scenario/path/configuration for the run being reproduced.

`--headless` uses PyBullet **DIRECT** mode (no GUI window and no waiting at the end). Omit `--headless` when a visible PyBullet window is desired, if the runner supports GUI mode.

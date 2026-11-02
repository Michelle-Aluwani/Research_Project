#!/usr/bin/env python3
"""Run ONLY missing CMDP planning cases for the configured D values.

Requires existing terrain_grid.json and start_goal_scenarios.json.
Does not generate grids/scenarios, run baselines, or launch PyBullet.
Reads D_VALUES from experiment_runner.py.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import experiment_runner as er


def existing_result_complete(result: Path, path: Path) -> bool:
    if not result.is_file():
        return False
    try:
        data = json.loads(result.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return False
    if data.get('reached_goal') is True:
        return path.is_file()
    # Intentionally preserve an existing failed-plan result as well.
    if data.get('reached_goal') is False:
        return True
    return False


def run_job(job):
    label, cmd, cwd = job
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    try:
        rc = subprocess.run(cmd, cwd=str(cwd), env=env, check=False).returncode
    except OSError as exc:
        return label, 1, cmd, str(exc)
    return label, rc, cmd, ''


def main():
    p = argparse.ArgumentParser(description='CMDP-only planning; existing terrain and scenario files are reused')
    p.add_argument('--experiments-dir', type=Path, required=True)
    p.add_argument('--terrain', action='append', help='Repeat for specific terrains, e.g. --terrain terrain_1')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--python', default=sys.executable)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        p.error('--workers must be between 1 and 32')

    root = args.experiments_dir.resolve()
    terrains = ([root / t for t in args.terrain] if args.terrain else
                sorted(t for t in root.glob('terrain_*') if t.is_dir()))
    if not terrains or any(not t.is_dir() for t in terrains):
        p.error('No matching terrain folders found.')

    project = Path(__file__).resolve().parent
    solver = project / er.CMDP_SCRIPT
    if not solver.is_file():
        p.error(f'Missing CMDP solver: {solver}')
    print(f'CMDP ONLY: D_VALUES = {er.D_VALUES}', flush=True)

    failed = 0
    for terrain in terrains:
        print(f'\n===== {terrain.name}: CMDP planning (NO PYBULLET) =====', flush=True)
        grid = terrain / 'terrain_grid.json'
        scenarios_file = terrain / 'start_goal_scenarios.json'
        if not grid.is_file() or not scenarios_file.is_file():
            print(f'SKIP: missing existing grid or scenarios for {terrain.name}. No files generated.', flush=True)
            failed += 1
            continue
        scenarios = er.load_scenarios(terrain)
        if not scenarios:
            print('No generated start-goal pairs for this terrain; skipping.', flush=True)
            continue

        jobs = []
        skipped = 0
        for sc in scenarios:
            sid = sc['scenario_id']
            for D in er.D_VALUES:
                condition = terrain / sid / f'D{D:g}'
                result = condition / 'cmdp_result.json'
                path = condition / 'cmdp_path_world.csv'
                if existing_result_complete(result, path):
                    skipped += 1
                    continue
                condition.mkdir(parents=True, exist_ok=True)
                cmd = [args.python, str(solver), '--terrain-grid', str(grid),
                       '--start', f"{int(sc['start'][0])},{int(sc['start'][1])}",
                       '--goal', f"{int(sc['goal'][0])},{int(sc['goal'][1])}",
                       '--D', str(D), '--output', str(result),
                       '--path-output', str(path), '--backend', 'auto']
                jobs.append((f'{sid} D={D}', cmd, project))
        print(f'Pending: {len(jobs)} | existing complete results skipped: {skipped}', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, j) for j in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                label, rc, cmd, error = future.result()
                status = 'OK' if rc == 0 else 'NO_PATH' if rc == 2 else 'FAIL'
                print(f'[{i}/{len(jobs)}] {status}: {label} {error}', flush=True)
                if status == 'FAIL':
                    failed += 1
                    er.log_failure(terrain, label, cmd, rc)
    print('\nCMDP planning finished. Baselines and PyBullet were NOT run.', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

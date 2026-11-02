#!/usr/bin/env python3
"""CMDP planning only. Never launches PyBullet or baseline planning."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import experiment_runner as er


def run_job(job):
    label, command, cwd = job
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    try:
        code = subprocess.run(command, cwd=str(cwd), env=env, check=False).returncode
        return label, code, command
    except Exception as exc:
        return label, str(exc), command


def batch(jobs, workers, terrain):
    print(f'[{terrain.name}] CMDP planning: {len(jobs)} pending jobs; workers={workers}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            label, code, cmd = future.result()
            status = 'OK' if code == 0 else 'NO_PATH' if code == 2 else 'FAIL'
            print(f'[{i}/{len(jobs)}] {status}: {label}', flush=True)
            if status == 'FAIL':
                er.log_failure(terrain, label, cmd, code)


def main():
    p = argparse.ArgumentParser(description='Parallel CMDP planning only')
    p.add_argument('--experiments-dir', type=Path, required=True)
    p.add_argument('--terrain', action='append', help='Repeat to select terrains')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--python', default=sys.executable)
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        p.error('--workers must be between 1 and 32')
    root = args.experiments_dir.resolve()
    terrains = ([root / t for t in args.terrain] if args.terrain else
                sorted(t for t in root.glob('terrain_*') if t.is_dir()))
    if not terrains or not all(t.is_dir() for t in terrains):
        p.error('One or more terrain folders do not exist')
    project = Path(__file__).resolve().parent
    print('CMDP risk budgets:', er.D_VALUES, flush=True)
    for terrain in terrains:
        print(f'\n===== {terrain.name}: CMDP PLANNING ONLY =====', flush=True)
        er.ensure_terrain_grid(terrain, project)
        er.generate_scenarios(terrain, args.python, project)
        scenarios = er.load_scenarios(terrain)
        if not scenarios:
            print('No scenarios available; skipping', flush=True)
            continue
        jobs = []
        for sc in scenarios:
            sid = sc['scenario_id']
            for D in er.D_VALUES:
                out = terrain / sid / f'D{D:g}'
                out.mkdir(parents=True, exist_ok=True)
                result = out / 'cmdp_result.json'
                path = out / 'cmdp_path_world.csv'
                if result.is_file():
                    try:
                        saved = json.loads(result.read_text(encoding='utf-8'))
                        if (saved.get('reached_goal') is False or
                                (saved.get('reached_goal') is True and path.is_file())):
                            continue
                    except (ValueError, OSError):
                        pass
                cmd = [args.python, str(project / er.CMDP_SCRIPT),
                       '--terrain-grid', str(terrain / 'terrain_grid.json'),
                       '--start', f"{int(sc['start'][0])},{int(sc['start'][1])}",
                       '--goal', f"{int(sc['goal'][0])},{int(sc['goal'][1])}",
                       '--D', str(D), '--output', str(result),
                       '--path-output', str(path), '--backend', 'auto']
                jobs.append((f'{sid} D={D}', cmd, project))
        batch(jobs, args.workers, terrain)
    print('\nCMDP planning finished. No PyBullet simulations were launched.', flush=True)


if __name__ == '__main__':
    main()

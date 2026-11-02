#!/usr/bin/env python3
"""RRT* and risk-weighted RRT* planning only; no PyBullet simulations."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import baseline_runner as br


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
    print(f'[{terrain.name}] Baseline planning: {len(jobs)} pending jobs; workers={workers}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            label, code, cmd = future.result()
            status = 'OK' if code == 0 else 'NO_PATH' if code == 2 else 'FAIL'
            print(f'[{i}/{len(jobs)}] {status}: {label}', flush=True)
            if status == 'FAIL':
                br.log_failure(terrain, label, cmd, code)


def main():
    p = argparse.ArgumentParser(description='Parallel RRT* baseline planning only')
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
    print('Alpha values:', br.ALPHA_VALUES, 'Baseline seeds:', br.NUM_MONTE_CARLO, flush=True)
    for terrain in terrains:
        print(f'\n===== {terrain.name}: BASELINE PLANNING ONLY =====', flush=True)
        scenario_file = terrain / 'start_goal_scenarios.json'
        grid_file = terrain / 'terrain_grid.json'
        if not scenario_file.is_file() or not grid_file.is_file():
            print('Missing terrain_grid.json or start_goal_scenarios.json; run CMDP planning first.', flush=True)
            continue
        # Beta is written once, before concurrent planner jobs start.
        scenarios = br.ensure_betas(terrain, scenario_file, project, args.python)
        jobs = []
        for sc in scenarios:
            if sc.get('beta') is None:
                print(f"Skipping {sc['scenario_id']}: beta unavailable", flush=True)
                continue
            sid = sc['scenario_id']
            cases = [('rrt_star', None)] + [('risk_weighted_rrt_star', a) for a in br.ALPHA_VALUES]
            for algorithm, alpha in cases:
                for seed in range(1, br.NUM_MONTE_CARLO + 1):
                    out = terrain / 'baseline' / sid / algorithm
                    if alpha is not None:
                        out = out / f'alpha_{alpha:.2f}'
                    out = out / f'seed_{seed:03d}'
                    out.mkdir(parents=True, exist_ok=True)
                    path = out / 'cmdp_path_world.csv'
                    result = out / 'planner_result.json'
                    if result.is_file():
                        try:
                            saved = json.loads(result.read_text(encoding='utf-8'))
                            if (saved.get('path_found') is False or
                                    (saved.get('path_found') is True and path.is_file())):
                                continue
                        except (ValueError, OSError):
                            pass
                    cmd = [args.python, str(project / br.RRT_SCRIPT),
                           '--terrain-dir', str(terrain), '--scenario-id', sid,
                           '--algorithm', algorithm, '--seed', str(seed),
                           '--path-output', str(path), '--result-output', str(result)]
                    if alpha is not None:
                        cmd += ['--alpha', str(alpha), '--beta', str(sc['beta'])]
                    jobs.append((f'{sid} {algorithm} alpha={alpha} seed={seed}', cmd, project))
        batch(jobs, args.workers, terrain)
    print('\nBaseline planning finished. No PyBullet simulations were launched.', flush=True)


if __name__ == '__main__':
    main()

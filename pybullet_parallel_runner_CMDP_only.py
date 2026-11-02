#!/usr/bin/env python3
"""Run PyBullet ONLY for existing CMDP paths at experiment_runner.D_VALUES.

Never launches the planners or baseline simulations. Uses 10 seeds by default,
configurable with --seeds. Existing completed results are skipped by default.
"""
import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import experiment_runner as er


def finished(folder: Path) -> bool:
    return any(folder.glob('result_*.json'))


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
    p = argparse.ArgumentParser(description='CMDP-only PyBullet runs from already saved paths')
    p.add_argument('--experiments-dir', type=Path, required=True)
    p.add_argument('--terrain', action='append', help='Repeat to select terrains')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--seeds', type=int, default=3, help='Number of execution seeds (default: 3)')
    p.add_argument('--extra', default='', help='Additional flags passed to run.py')
    p.add_argument('--rerun', action='store_true', help='Repeat finished runs; overwrites results in those seed folders')
    args = p.parse_args()
    if not 1 <= args.workers <= 32:
        p.error('--workers must be between 1 and 32')
    if args.seeds < 1:
        p.error('--seeds must be at least 1')
    extra = shlex.split(args.extra)
    root = args.experiments_dir.resolve()
    terrains = ([root / t for t in args.terrain] if args.terrain else
                sorted(t for t in root.glob('terrain_*') if t.is_dir()))
    if not terrains or any(not t.is_dir() for t in terrains):
        p.error('No matching terrain folders found.')
    project = Path(__file__).resolve().parent
    simulator = project / er.RUN_SCRIPT
    if not simulator.is_file():
        p.error(f'No simulation script: {simulator}')

    failed = 0
    print(f'CMDP PyBullet ONLY: D_VALUES={er.D_VALUES}, seeds=1..{args.seeds}', flush=True)
    for terrain in terrains:
        print(f'\n===== {terrain.name}: CMDP PYBULLET ONLY =====', flush=True)
        if not (terrain / 'start_goal_scenarios.json').is_file():
            print('Missing scenarios; skipping', flush=True)
            failed += 1
            continue
        jobs = []
        skipped = 0
        missing_paths = 0
        for sc in er.load_scenarios(terrain):
            sid = sc['scenario_id']
            for D in er.D_VALUES:
                condition = terrain / sid / f'D{D:g}'
                path = condition / 'cmdp_path_world.csv'
                if not path.is_file():
                    missing_paths += 1
                    continue
                for seed in range(1, args.seeds + 1):
                    out = condition / f'seed_{seed:03d}'
                    if finished(out) and not args.rerun:
                        skipped += 1
                        continue
                    out.mkdir(parents=True, exist_ok=True)
                    cmd = [args.python, str(simulator), '--terrain-dir', str(terrain),
                           '--cmdp-path', str(path), '--scenario-id', sid,
                           '--risk-budget', str(D), '--seed', str(seed),
                           '--output-dir', str(out), '--headless'] + extra
                    jobs.append((f'{sid} D={D} seed={seed}', cmd, project))
        print(f'Pending: {len(jobs)} | completed skipped: {skipped} | without path: {missing_paths}', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, j) for j in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                label, rc, cmd, error = future.result()
                print(f'[{i}/{len(jobs)}] {"OK" if rc == 0 else "FAIL"}: {label} {error}', flush=True)
                if rc != 0:
                    failed += 1
                    er.log_failure(terrain, label, cmd, rc)
    print('\nCMDP PyBullet finished. No baseline runs or planning jobs launched.', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

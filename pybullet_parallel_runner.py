#!/usr/bin/env python3
"""Run PyBullet for saved CMDP and RRT* paths, without rerunning planners.

Seeds, D values, alpha values and scenarios can now be chosen per run, e.g.

    python pybullet_runner.py --experiments-dir experiments --terrain terrain_01 \
        --cmdp-seeds 5 --baseline-seeds 3 \
        --d-values 4 6 7 --alphas 0.75 0.5 \
        --scenario-ids mud_06 mud_07 hazard_06
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import experiment_runner as er
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


def finished(out):
    return any(out.glob('result_*.json'))


def batch(jobs, workers, terrain):
    print(f'[{terrain.name}] PyBullet: {len(jobs)} pending simulations; workers={workers}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            label, code, cmd = future.result()
            status = 'OK' if code == 0 else 'FAIL'
            print(f'[{i}/{len(jobs)}] {status}: {label}', flush=True)
            if status == 'FAIL':
                er.log_failure(terrain, label, cmd, code)


def main():
    p = argparse.ArgumentParser(description='Parallel PyBullet simulations from saved planner paths')
    p.add_argument('--experiments-dir', type=Path, required=True)
    p.add_argument('--terrain', action='append', help='Repeat to select terrains')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--planner', choices=('all', 'cmdp', 'baseline'), default='all')
    p.add_argument('--python', default=sys.executable)
    # NEW (optional): extra flags appended to EVERY job, and re-simulate finished folders.
    p.add_argument('--extra', default='', help='extra flags for every job, e.g. "--reverse-deg 120"')
    p.add_argument('--rerun', action='store_true', help='simulate again even if result_*.json exists')
    # NEW (optional): choose seeds / conditions separately for each planner.
    p.add_argument('--cmdp-seeds', type=int, default=None,
                   help='CMDP seeds 1..N (default: experiment_runner.NUM_MONTE_CARLO)')
    p.add_argument('--baseline-seeds', type=int, default=None,
                   help='RRT* seeds 1..N (default: baseline_runner.NUM_MONTE_CARLO)')
    p.add_argument('--d-values', type=float, nargs='+', default=None,
                   help='CMDP risk limits to simulate, e.g. 4 6 7 (default: all)')
    p.add_argument('--alphas', type=float, nargs='+', default=None,
                   help='risk-weighted RRT* alpha values, e.g. 0.75 0.5 (default: all). '
                        'Standard RRT* is always included.')
    p.add_argument('--scenario-ids', nargs='+', default=None,
                   help='only simulate these scenario ids, e.g. mud_06 mud_07 (default: all)')
    args = p.parse_args()
    extra = shlex.split(args.extra)
    if not 1 <= args.workers <= 32:
        p.error('--workers must be between 1 and 32')

    cmdp_seeds = args.cmdp_seeds if args.cmdp_seeds is not None else er.NUM_MONTE_CARLO
    baseline_seeds = args.baseline_seeds if args.baseline_seeds is not None else br.NUM_MONTE_CARLO
    if cmdp_seeds < 1 or baseline_seeds < 1:
        p.error('seed counts must be at least 1')
    d_values = args.d_values if args.d_values else list(er.D_VALUES)
    alphas = args.alphas if args.alphas else list(br.ALPHA_VALUES)
    for D in d_values:
        if not any(abs(D - known) < 1e-9 for known in er.D_VALUES):
            print(f'WARNING: D={D:g} is not in experiment_runner.D_VALUES {list(er.D_VALUES)}', flush=True)
    for a in alphas:
        if not any(abs(a - known) < 1e-9 for known in br.ALPHA_VALUES):
            print(f'WARNING: alpha={a:g} is not in baseline_runner.ALPHA_VALUES {list(br.ALPHA_VALUES)}', flush=True)

    root = args.experiments_dir.resolve()
    terrains = ([root / t for t in args.terrain] if args.terrain else
                sorted(t for t in root.glob('terrain_*') if t.is_dir()))
    if not terrains or not all(t.is_dir() for t in terrains):
        p.error('One or more terrain folders do not exist')
    project = Path(__file__).resolve().parent
    for terrain in terrains:
        print(f'\n===== {terrain.name}: PYBULLET ONLY =====', flush=True)
        print(f'CMDP seeds 1-{cmdp_seeds}, D={[f"{D:g}" for D in d_values]}; '
              f'baseline seeds 1-{baseline_seeds}, alpha={alphas}', flush=True)
        jobs = []
        scenario_file = terrain / 'start_goal_scenarios.json'
        if not scenario_file.is_file():
            print('Missing scenarios; skipping terrain', flush=True)
            continue
        scenarios = er.load_scenarios(terrain)
        if args.scenario_ids:
            wanted = set(args.scenario_ids)
            known = {sc['scenario_id'] for sc in scenarios}
            for missing in sorted(wanted - known):
                print(f'WARNING: scenario {missing} not found in {terrain.name}', flush=True)
            scenarios = [sc for sc in scenarios if sc['scenario_id'] in wanted]
        if args.planner in ('all', 'cmdp'):
            for sc in scenarios:
                sid = sc['scenario_id']
                for D in d_values:
                    condition = terrain / sid / f'D{D:g}'
                    path = condition / 'cmdp_path_world.csv'
                    if not path.is_file():
                        continue
                    for seed in range(1, cmdp_seeds + 1):
                        out = condition / f'seed_{seed:03d}'
                        if finished(out) and not args.rerun:
                            continue
                        out.mkdir(parents=True, exist_ok=True)
                        cmd = [args.python, str(project / er.RUN_SCRIPT),
                               '--terrain-dir', str(terrain), '--cmdp-path', str(path),
                               '--scenario-id', sid, '--risk-budget', str(D),
                               '--seed', str(seed), '--output-dir', str(out), '--headless'] + extra
                        jobs.append((f'CMDP {sid} D={D} seed={seed}', cmd, project))
        if args.planner in ('all', 'baseline'):
            for sc in scenarios:
                sid = sc['scenario_id']
                cases = [('rrt_star', None)] + [('risk_weighted_rrt_star', a) for a in alphas]
                for algorithm, alpha in cases:
                    for seed in range(1, baseline_seeds + 1):
                        out = terrain / 'baseline' / sid / algorithm
                        if alpha is not None:
                            out = out / f'alpha_{alpha:.2f}'
                        out = out / f'seed_{seed:03d}'
                        path = out / 'cmdp_path_world.csv'
                        if not path.is_file() or (finished(out) and not args.rerun):
                            continue
                        out.mkdir(parents=True, exist_ok=True)
                        cmd = [args.python, str(project / br.RUN_SCRIPT),
                               '--terrain-dir', str(terrain), '--cmdp-path', str(path),
                               '--scenario-id', sid, '--risk-budget', str(br.BASELINE_RISK_BUDGET),
                               '--seed', str(seed), '--output-dir', str(out), '--headless'] + extra
                        jobs.append((f'{algorithm} {sid} alpha={alpha} seed={seed}', cmd, project))
        batch(jobs, args.workers, terrain)
    print('\nPyBullet stage finished. No planning jobs were launched.', flush=True)


if __name__ == '__main__':
    main()
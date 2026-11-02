#!/usr/bin/env python3
"""Run all CMDP and RRT* planning jobs without launching PyBullet.

Uses the separate planning-only runners. Original experiment runners are untouched.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='All planning experiments, no PyBullet')
    parser.add_argument('--experiments-dir', type=Path, required=True)
    parser.add_argument('--terrain', action='append', help='Repeat for specific terrains; default: all terrain_* folders')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be between 1 and 32')
    root = args.experiments_dir.resolve()
    terrains = ([root / name for name in args.terrain] if args.terrain else
                sorted(p for p in root.glob('terrain_*') if p.is_dir()))
    if not terrains or any(not p.is_dir() for p in terrains):
        parser.error('No matching terrain folders found (or a selected terrain is missing).')
    project = Path(__file__).resolve().parent
    failures = []
    for terrain in terrains:
        for stage, script in [('CMDP', 'cmdp_parallel_runner.py'),
                              ('RRT* baselines', 'baseline_parallel_runner.py')]:
            script_path = project / script
            if not script_path.is_file():
                parser.error(f'Missing {script_path}. Put all three planning runners together.')
            cmd = [args.python, str(script_path), '--experiments-dir', str(root),
                   '--terrain', terrain.name, '--workers', str(args.workers),
                   '--python', args.python]
            print(f'\n===== {terrain.name}: {stage} (NO PYBULLET) =====', flush=True)
            try:
                rc = subprocess.run(cmd, cwd=str(project), check=False).returncode
            except OSError as exc:
                print(f'Cannot start {stage}: {exc}', flush=True)
                rc = 1
            if rc != 0:
                failures.append((terrain.name, stage, rc))
                print(f'ERROR: {terrain.name} {stage} failed; skipping remaining stages for this terrain.', flush=True)
                break
    if failures:
        print('\nFailed stages:', failures, flush=True)
        return 1
    print('\nAll requested planning stages finished. PyBullet was NOT launched.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

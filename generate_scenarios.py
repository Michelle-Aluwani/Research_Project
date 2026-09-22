#!/usr/bin/env python3
"""
Generate fixed start/goal scenarios for ONE terrain.

Important experiment rule:
    - Each terrain gets its OWN coordinates.
    - Scenario categories are the same across terrains.
    - The generated scenarios are fixed and reused for every D and MC seed.
    - Scenario generation seed is separate from the PyBullet Monte Carlo seed.

Example:
    python generate_scenarios.py --terrain-dir experiments/terrain_01
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

NUM_SCENARIOS = 5
MIN_DISTANCE = 40.0
MAX_DISTANCE = 150.0
BORDER_MARGIN = 5
NUM_PAIRS = 10000
MIN_PAIR_DIFFERENCE = 15.0

SCENARIO_SEED = 42

TERRAIN_ROCK = 0
TERRAIN_MUD = 1
TERRAIN_WATER = 2

SCENARIO_TYPES = (
    "mixed",
    "mud",
    "water",
    "hazard",
    "long",
)


# ============================================================
# LOAD TERRAIN
# ============================================================

def load_terrain(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    grid = np.asarray(data["grid"], dtype=np.int8)
    metadata = data["metadata"]

    return grid, metadata


# ============================================================
# DIRECT ROUTE
# ============================================================

def direct_route_cells(grid, start, goal):
    nx, ny = grid.shape

    x1, y1 = start
    x2, y2 = goal

    distance = math.hypot(x2 - x1, y2 - y1)
    samples = int(math.ceil(distance)) + 1

    xs = np.rint(np.linspace(x1, x2, samples)).astype(int)
    ys = np.rint(np.linspace(y1, y2, samples)).astype(int)

    cells = []
    seen = set()

    for x, y in zip(xs, ys):
        if 0 <= x < nx and 0 <= y < ny:
            cell = (int(x), int(y))
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)

    return cells


def route_terrain_counts(grid, route):
    values = np.asarray([grid[x, y] for x, y in route])

    return {
        "rock": int(np.sum(values == TERRAIN_ROCK)),
        "mud": int(np.sum(values == TERRAIN_MUD)),
        "water": int(np.sum(values == TERRAIN_WATER)),
        "total": int(len(values)),
    }


# ============================================================
# CANDIDATE CELLS
# ============================================================

def candidate_cells(grid):
    nx, ny = grid.shape

    cells = []

    for x in range(BORDER_MARGIN, nx - BORDER_MARGIN):
        for y in range(BORDER_MARGIN, ny - BORDER_MARGIN):

            # Start/goal must not be water.
            if grid[x, y] == TERRAIN_WATER:
                continue

            cells.append((x, y))

    return np.asarray(cells, dtype=np.int32)


def choose_start_goal_candidates(grid, rng):
    """
    Generate random valid start/goal pairs for this terrain.

    The coordinates are deliberately generated from the current
    terrain grid, so a different terrain gets different valid pairs.
    """

    cells = candidate_cells(grid)

    if len(cells) < 2:
        raise RuntimeError("Not enough valid start/goal cells.")

    # Prefer opposite sides of this particular terrain, while
    # retaining the terrain-specific geometry.
    nx = grid.shape[0]

    left = cells[cells[:, 0] < nx * 0.25]
    right = cells[cells[:, 0] > nx * 0.75]

    if len(left) == 0 or len(right) == 0:
        left = cells
        right = cells

    candidates = []

    for _ in range(NUM_PAIRS):
        start = tuple(left[rng.integers(len(left))])
        goal = tuple(right[rng.integers(len(right))])

        distance = math.hypot(
            goal[0] - start[0],
            goal[1] - start[1],
        )

        if not MIN_DISTANCE <= distance <= MAX_DISTANCE:
            continue

        route = direct_route_cells(grid, start, goal)
        counts = route_terrain_counts(grid, route)

        candidates.append({
            "start": start,
            "goal": goal,
            "distance_cells": distance,
            "route": route,
            "counts": counts,
        })

    return candidates


# ============================================================
# SCENARIO SCORING
# ============================================================

def score_scenario(candidate, scenario_type):
    counts = candidate["counts"]
    distance = candidate["distance_cells"]

    mud = counts["mud"]
    water = counts["water"]
    non_rock = mud + water

    if scenario_type == "mixed":
        # Require both mud and water.
        if mud == 0 or water == 0:
            return None
        return (
            1000.0 * min(mud, water)
            + 10.0 * (mud + water)
            - 0.01 * distance
        )

    if scenario_type == "mud":
        if mud == 0:
            return None
        return (
            1000.0 * mud
            + 10.0 * water
            - 0.01 * distance
        )

    if scenario_type == "water":
        if water == 0:
            return None
        return (
            1000.0 * water
            + 10.0 * mud
            - 0.01 * distance
        )

    if scenario_type == "hazard":
        if non_rock == 0:
            return None
        return (
            1000.0 * non_rock
            + 25.0 * water
            + 10.0 * mud
            - 0.01 * distance
        )

    if scenario_type == "long":
        return distance

    raise ValueError(f"Unknown scenario type: {scenario_type}")


def sufficiently_different(candidate, selected):
    s1 = candidate["start"]
    g1 = candidate["goal"]

    for other in selected:
        s2 = other["start"]
        g2 = other["goal"]

        start_diff = math.hypot(
            s1[0] - s2[0],
            s1[1] - s2[1],
        )

        goal_diff = math.hypot(
            g1[0] - g2[0],
            g1[1] - g2[1],
        )

        if (
            start_diff < MIN_PAIR_DIFFERENCE
            and goal_diff < MIN_PAIR_DIFFERENCE
        ):
            return False

    return True


def select_scenarios(candidates):
    selected = {}

    for scenario_type in SCENARIO_TYPES:
        scored = []

        for candidate in candidates:
            score = score_scenario(candidate, scenario_type)

            if score is not None:
                scored.append((score, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)

        chosen = []

        for score, candidate in scored:
            if sufficiently_different(candidate, chosen):
                item = dict(candidate)
                item["score"] = float(score)
                item.pop("route", None)
                chosen.append(item)

            if len(chosen) == NUM_SCENARIOS:
                break

        if len(chosen) < NUM_SCENARIOS:
            raise RuntimeError(
                f"Could only find {len(chosen)} {scenario_type} "
                f"scenarios; required {NUM_SCENARIOS}."
            )

        selected[scenario_type] = chosen

    return selected


# ============================================================
# SAVE
# ============================================================

def save_scenarios(output_path, terrain_dir, grid, metadata, scenarios):
    result = {
        "metadata": {
            "terrain_dir": str(terrain_dir),
            "scenario_seed": SCENARIO_SEED,
            "num_scenarios_per_type": NUM_SCENARIOS,
            "scenario_types": list(SCENARIO_TYPES),
            "min_distance_cells": MIN_DISTANCE,
            "max_distance_cells": MAX_DISTANCE,
            "border_margin": BORDER_MARGIN,
            "num_candidate_pairs": NUM_PAIRS,
            "min_pair_difference": MIN_PAIR_DIFFERENCE,
            "note": (
                "Coordinates are terrain-specific. Reuse these exact "
                "start/goal pairs for every D and every Monte Carlo seed "
                "on this terrain."
            ),
        },
        "terrain": {
            "grid_width": int(metadata["grid_width"]),
            "grid_height": int(metadata["grid_height"]),
        },
        "scenarios": {},
    }

    for scenario_type, items in scenarios.items():
        result["scenarios"][scenario_type] = []

        for index, item in enumerate(items, start=1):
            result["scenarios"][scenario_type].append({
                "scenario_id": f"{scenario_type}_{index:02d}",
                "start": list(map(int, item["start"])),
                "goal": list(map(int, item["goal"])),
                "distance_cells": float(item["distance_cells"]),
                "line_terrain_counts": item["counts"],
                "selection_score": float(item["score"]),
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    global SCENARIO_SEED

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--terrain-dir",
        type=Path,
        required=True,
        help="Directory containing terrain_grid.json.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SCENARIO_SEED,
        help="Fixed scenario-generation seed.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to terrain-dir/start_goal_scenarios.json.",
    )

    args = parser.parse_args()

    SCENARIO_SEED = args.seed

    terrain_dir = args.terrain_dir.resolve()
    terrain_grid_path = terrain_dir / "terrain_grid.json"

    if not terrain_grid_path.exists():
        raise FileNotFoundError(terrain_grid_path)

    output_path = (
        args.output.resolve()
        if args.output is not None
        else terrain_dir / "start_goal_scenarios.json"
    )

    grid, metadata = load_terrain(terrain_grid_path)

    print("=" * 60)
    print("TERRAIN-SPECIFIC SCENARIO GENERATION")
    print("=" * 60)
    print(f"Terrain:        {terrain_dir}")
    print(f"Grid:           {grid.shape[0]} x {grid.shape[1]}")
    print(f"Scenario seed:  {SCENARIO_SEED}")
    print()

    rng = np.random.default_rng(SCENARIO_SEED)

    candidates = choose_start_goal_candidates(grid, rng)

    print(f"Candidate pairs: {len(candidates)}")

    scenarios = select_scenarios(candidates)

    for scenario_type in SCENARIO_TYPES:
        print(f"\n{scenario_type.upper()}")

        for item in scenarios[scenario_type]:
            print(
                f"  {item['scenario_id']}: "
                f"{item['start']} -> {item['goal']} | "
                f"distance={item['distance_cells']:.1f} | "
                f"mud={item['counts']['mud']} | "
                f"water={item['counts']['water']}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_scenarios(
        output_path,
        terrain_dir,
        grid,
        metadata,
        scenarios,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate fixed start/goal scenarios for ONE terrain, or EXTEND an existing set.

Important experiment rule:
    - Each terrain gets its OWN coordinates.
    - Scenario categories are the same across terrains.
    - The generated scenarios are fixed and reused for every D and MC seed.
    - Scenario generation seed is separate from the PyBullet Monte Carlo seed.

Fresh generation (refuses to overwrite an existing file unless --overwrite):
    python generate_scenarios.py --terrain-dir experiments/terrain_01

Extend an existing file up to 10 scenarios per category. Existing scenarios
are kept exactly as they are; only the missing ones are added:
    python generate_scenarios.py --terrain-dir experiments/terrain_01 \
        --extend --target 10 --seed 1042

Optional filters for the NEW scenarios:
    --max-route-slope 10      keep only pairs whose straight start->goal line
                              never crosses a cell steeper than 10 degrees
                              (a proxy: the planned path can deviate from it)
    --unique-across-categories  a new pair may not be close to ANY existing
                              pair, including pairs in other categories
"""

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Maximum scenarios kept per category (override with --target). A category may
# end up with fewer on a given terrain (e.g. a terrain with no water has no
# "water" scenarios); that is NOT an error, but it is reported.
MAX_SCENARIOS_PER_CATEGORY = 10
MIN_DISTANCE = 40.0
MAX_DISTANCE = 150.0
BORDER_MARGIN = 5
NUM_PAIRS = 10000
MIN_PAIR_DIFFERENCE = 15.0

SCENARIO_SEED = 42
CELL_SIZE_M = 0.5  # metres per grid cell, used for slope

TERRAIN_ROCK = 0
TERRAIN_MUD = 1
TERRAIN_WATER = 2
TERRAIN_WALL = 3

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

    height_grid = np.asarray(
        [
            [np.nan if v is None else v for v in row]
            for row in data["height_grid"]
        ],
        dtype=np.float64,
    )

    return grid, height_grid, metadata


def compute_slope_grid(height_grid, cell_size):
    """Slope in degrees per cell, same definition as the risk model."""
    gx, gy = np.gradient(height_grid, cell_size, cell_size)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


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


def route_max_slope(slope_grid, route):
    """Steepest finite slope (degrees) along the route; inf if none finite."""
    values = np.asarray([slope_grid[x, y] for x, y in route], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else float("inf")


# ============================================================
# CANDIDATE CELLS
# ============================================================

def candidate_cells(grid, height_grid):
    nx, ny = grid.shape

    cells = []

    for x in range(BORDER_MARGIN, nx - BORDER_MARGIN):
        for y in range(BORDER_MARGIN, ny - BORDER_MARGIN):

            # Start/goal must not be water.
            if grid[x, y] == TERRAIN_WATER:
                continue

            # ... must not be a wall cell.
            if grid[x, y] == TERRAIN_WALL:
                continue

            # ... and must have a surface. Cells with no surface are stored
            # as type 0 (rock) in the grid, so check the height as well.
            if not np.isfinite(height_grid[x, y]):
                continue

            cells.append((x, y))

    return np.asarray(cells, dtype=np.int32)


def choose_start_goal_candidates(grid, height_grid, slope_grid, rng,
                                 max_route_slope=None):
    """
    Generate random valid start/goal pairs for this terrain.

    The coordinates are deliberately generated from the current
    terrain grid, so a different terrain gets different valid pairs.
    If max_route_slope is given, pairs whose straight line crosses a cell
    steeper than that (degrees) are dropped.
    """

    cells = candidate_cells(grid, height_grid)

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
        slope = route_max_slope(slope_grid, route)

        if max_route_slope is not None and slope > max_route_slope:
            continue

        candidates.append({
            "start": start,
            "goal": goal,
            "distance_cells": distance,
            "route": route,
            "counts": counts,
            "max_route_slope_deg": slope,
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


# ============================================================
# EXISTING SCENARIOS (for --extend)
# ============================================================

def load_existing_scenarios(path):
    """Read an earlier scenario file. Existing entries are never changed."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing = {scenario_type: [] for scenario_type in SCENARIO_TYPES}

    for scenario_type, items in data.get("scenarios", {}).items():
        existing.setdefault(scenario_type, [])

        for item in items:
            existing[scenario_type].append({
                "scenario_id": item["scenario_id"],
                "start": tuple(item["start"]),
                "goal": tuple(item["goal"]),
                "distance_cells": float(item["distance_cells"]),
                "counts": item["line_terrain_counts"],
                "score": float(item.get("selection_score", 0.0)),
                "max_route_slope_deg": item.get("max_route_slope_deg"),
                "is_new": False,
            })

    return existing, data.get("metadata", {})


def select_scenarios(candidates, existing, target, unique_across_categories):
    """
    Fill every category up to `target` scenarios.

    `existing` holds scenarios that are already fixed. They stay first and
    unchanged; new ones are only appended and must differ from them.
    With an empty `existing` this is the original fresh selection.
    """
    selected = {t: list(items) for t, items in existing.items()}
    for scenario_type in SCENARIO_TYPES:
        selected.setdefault(scenario_type, [])

    for scenario_type in SCENARIO_TYPES:
        kept = selected[scenario_type]
        need = target - len(kept)

        if need <= 0:
            continue

        scored = []

        for candidate in candidates:
            score = score_scenario(candidate, scenario_type)

            if score is not None:
                scored.append((score, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)

        added = 0

        for score, candidate in scored:
            if added >= need:
                break

            if unique_across_categories:
                pool = [it for items in selected.values() for it in items]
            else:
                pool = kept

            if not sufficiently_different(candidate, pool):
                continue

            used_ids = {it["scenario_id"] for it in kept}
            index = len(kept) + 1
            while f"{scenario_type}_{index:02d}" in used_ids:
                index += 1

            item = dict(candidate)
            item["score"] = float(score)
            item["scenario_id"] = f"{scenario_type}_{index:02d}"
            item["is_new"] = True
            item.pop("route", None)
            kept.append(item)
            added += 1

    return selected


# ============================================================
# SAVE
# ============================================================

def save_scenarios(output_path, terrain_dir, grid, metadata, scenarios,
                   target, scenario_seed, extensions):
    # Every category is always present in the file, even when empty.
    for scenario_type in SCENARIO_TYPES:
        scenarios.setdefault(scenario_type, [])

    result = {
        "metadata": {
            "terrain_dir": str(terrain_dir),
            "scenario_seed": scenario_seed,
            "max_scenarios_per_type": target,
            "scenario_counts": {
                scenario_type: len(items)
                for scenario_type, items in scenarios.items()
            },
            "excluded_start_goal_cells": "water, wall, no-surface, border",
            "scenario_types": list(SCENARIO_TYPES),
            "min_distance_cells": MIN_DISTANCE,
            "max_distance_cells": MAX_DISTANCE,
            "border_margin": BORDER_MARGIN,
            "num_candidate_pairs": NUM_PAIRS,
            "min_pair_difference": MIN_PAIR_DIFFERENCE,
            "extensions": extensions,
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

        for item in items:
            slope = item.get("max_route_slope_deg")

            result["scenarios"][scenario_type].append({
                "scenario_id": item["scenario_id"],
                "start": list(map(int, item["start"])),
                "goal": list(map(int, item["goal"])),
                "distance_cells": float(item["distance_cells"]),
                "line_terrain_counts": item["counts"],
                "selection_score": float(item["score"]),
                "max_route_slope_deg": (
                    None if slope is None or not math.isfinite(slope)
                    else round(float(slope), 2)
                ),
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
        help="Scenario-generation seed. Use a NEW seed when extending.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to terrain-dir/start_goal_scenarios.json.",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=MAX_SCENARIOS_PER_CATEGORY,
        help="Scenarios wanted per category (existing ones count).",
    )

    parser.add_argument(
        "--extend",
        action="store_true",
        help="Keep the existing scenario file and only add the missing ones.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a fresh run to replace an existing scenario file.",
    )

    parser.add_argument(
        "--max-route-slope",
        type=float,
        default=None,
        help="Only accept NEW pairs whose straight line stays under this "
             "slope in degrees (e.g. 10).",
    )

    parser.add_argument(
        "--cell-size",
        type=float,
        default=CELL_SIZE_M,
        help="Grid cell size in metres (used for slope).",
    )

    parser.add_argument(
        "--unique-across-categories",
        action="store_true",
        help="New pairs must also differ from pairs in other categories.",
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

    if output_path.exists() and not args.extend and not args.overwrite:
        raise SystemExit(
            f"{output_path} already exists. Use --extend to add scenarios "
            "to it, or --overwrite to regenerate it from scratch."
        )

    if args.extend and not output_path.exists():
        raise FileNotFoundError(
            f"--extend needs an existing file, but {output_path} is missing."
        )

    grid, height_grid, metadata = load_terrain(terrain_grid_path)
    slope_grid = compute_slope_grid(height_grid, args.cell_size)

    existing = {}
    old_meta = {}
    extensions = []

    if args.extend:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_name(
            f"{output_path.stem}.backup_{stamp}.json"
        )
        shutil.copy2(output_path, backup_path)
        existing, old_meta = load_existing_scenarios(output_path)
        extensions = list(old_meta.get("extensions", []))

        # Record the steepest route slope of old scenarios for reporting only.
        for items in existing.values():
            for item in items:
                if item.get("max_route_slope_deg") is None:
                    route = direct_route_cells(grid, item["start"], item["goal"])
                    item["max_route_slope_deg"] = route_max_slope(
                        slope_grid, route
                    )

    print("=" * 60)
    print("TERRAIN-SPECIFIC SCENARIO GENERATION")
    print("=" * 60)
    print(f"Terrain:        {terrain_dir}")
    print(f"Grid:           {grid.shape[0]} x {grid.shape[1]}")
    print(f"Mode:           {'EXTEND' if args.extend else 'FRESH'}")
    print(f"Scenario seed:  {SCENARIO_SEED}")
    print(f"Target/category:{args.target}")
    if args.max_route_slope is not None:
        print(f"Max route slope:{args.max_route_slope} deg (new pairs)")
    if args.extend:
        print(f"Backup saved:   {backup_path}")
    print()

    rng = np.random.default_rng(SCENARIO_SEED)

    candidates = choose_start_goal_candidates(
        grid,
        height_grid,
        slope_grid,
        rng,
        max_route_slope=args.max_route_slope,
    )

    print(f"Candidate pairs: {len(candidates)}")

    scenarios = select_scenarios(
        candidates,
        existing,
        args.target,
        args.unique_across_categories,
    )

    added_counts = {}
    shortfalls = {}

    for scenario_type in SCENARIO_TYPES:
        items = scenarios[scenario_type]
        added_counts[scenario_type] = sum(1 for it in items if it.get("is_new"))

        if len(items) < args.target:
            shortfalls[scenario_type] = args.target - len(items)

        print(f"\n{scenario_type.upper()}  "
              f"(total {len(items)}, new {added_counts[scenario_type]})")

        if not items:
            print("  (no scenarios found for this category)")

        for item in items:
            slope = item.get("max_route_slope_deg")
            slope_text = (
                f"{slope:.1f}" if slope is not None and math.isfinite(slope)
                else "n/a"
            )
            tag = "NEW" if item.get("is_new") else "old"
            print(
                f"  [{tag}] {item['scenario_id']}: "
                f"{item['start']} -> {item['goal']} | "
                f"distance={item['distance_cells']:.1f} | "
                f"mud={item['counts']['mud']} | "
                f"water={item['counts']['water']} | "
                f"max_slope={slope_text}"
            )

    if shortfalls:
        print("\nWARNING: fewer scenarios than the target were found:")
        for scenario_type, missing in shortfalls.items():
            print(f"  {scenario_type}: {missing} short of {args.target}")

    if args.extend:
        extensions.append({
            "date": datetime.now().isoformat(timespec="seconds"),
            "seed": SCENARIO_SEED,
            "target": args.target,
            "added": added_counts,
            "max_route_slope_deg": args.max_route_slope,
            "unique_across_categories": args.unique_across_categories,
        })
        base_seed = old_meta.get("scenario_seed", SCENARIO_SEED)
    else:
        base_seed = SCENARIO_SEED

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_scenarios(
        output_path,
        terrain_dir,
        grid,
        metadata,
        scenarios,
        args.target,
        base_seed,
        extensions,
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
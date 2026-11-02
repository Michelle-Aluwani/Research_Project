
from pathlib import Path
import csv
import math
import re
import shutil
import sys

import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

# Change this only if cmdp_refactored.py is elsewhere.
RISK_MODEL_FOLDER = ROOT

OUTPUT_FOLDER = ROOT / "COLLECTED_PYBULLET_RISK"

# True also creates a detailed CSV showing the risk
# and accumulated risk at every PyBullet position.
# These files can be very large.
WRITE_POINT_DETAILS = False

# Use the SAME risk model as CMDP.
sys.path.insert(0, str(RISK_MODEL_FOLDER))

from cmdp_refactored import (
    load_terrain_grid,
    prepare_risk_model,
)

TERRAIN_PATTERN = re.compile(r"terrain_\d+", re.I)

SCENARIO_PATTERN = re.compile(
    r"(mixed|mud|water|hazard|long)[_-]\d+", re.I
)

D_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9])D[_-]?(\d+(?:\.\d+)?)",
    re.I
)

SEED_PATTERN = re.compile(
    r"seed[_-]?(\d+)", re.I
)

ALPHA_PATTERN = re.compile(
    r"alpha[_-]?(\d+(?:\.\d+)?)", re.I
)


# ============================================================
# IDENTIFY EXPERIMENT
# ============================================================

def identify_run(path, terrain):
    relative = path.relative_to(terrain).as_posix()
    lower = relative.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lower)

    if (
        "risk_weighted_rrt" in normalized
        or "riskweighted_rrt" in normalized
        or "rw_rrt" in normalized
    ):
        planner = "risk_weighted_rrt_star"

    elif "rrt_star" in normalized or "rrtstar" in normalized:
        planner = "rrt_star"

    elif "baseline" in relative.lower().split("/"):
        planner = "baseline_unknown"

    else:
        planner = "cmdp"

    scenario = SCENARIO_PATTERN.search(relative)
    d_value = D_PATTERN.search(relative)
    seed = SEED_PATTERN.search(relative)
    alpha = ALPHA_PATTERN.search(relative)

    return {
        "terrain": terrain.name,
        "planner": planner,
        "scenario": scenario.group(0) if scenario else "",
        "D": d_value.group(1) if d_value else "",
        "alpha": alpha.group(1) if alpha else "",
        "seed": seed.group(1) if seed else "",
        "run_id": "__".join(
            path.relative_to(terrain).parent.parts
        ),
        "source": path.relative_to(ROOT).as_posix(),
    }


# ============================================================
# POSITION TO RISK-GRID CELL
# ============================================================

def position_to_cell(x, y, bounds, resolution):
    gx = math.floor(
        (x - float(bounds["min_x"])) / resolution
    )
    gy = math.floor(
        (y - float(bounds["min_y"])) / resolution
    )
    return gx, gy


def get_cell_risk(cell, risk_grid):
    gx, gy = cell

    if (
        0 <= gx < risk_grid.shape[0]
        and 0 <= gy < risk_grid.shape[1]
    ):
        value = risk_grid[gx, gy]
        if np.isfinite(value):
            return float(value)

    return None


def to_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


# ============================================================
# CALCULATE ACTUAL PYBULLET RISK
# ============================================================

def calculate_execution(
    file,
    terrain,
    risk_grid,
    bounds,
    resolution,
    detail_writer=None
):
    metadata = identify_run(file, terrain)

    samples = 0
    valid_samples = 0
    invalid_samples = 0

    # Accumulated grid-state risk.
    total_risk = 0.0
    risk_cell_visits = 0
    unknown_risk_visits = 0
    unique_cells = set()

    distance_2d = 0.0
    distance_3d = 0.0

    first_xyz = None
    last_xyz = None
    first_time = None
    last_time = None

    previous_xyz = None
    previous_cell = None

    with file.open(
        "r", newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)

        needed = {"actual_x", "actual_y", "actual_z"}
        available = set(reader.fieldnames or [])

        if not needed.issubset(available):
            raise ValueError(
                f"Missing columns: {needed - available}"
            )

        for sample in reader:
            samples += 1

            x = to_number(sample.get("actual_x"))
            y = to_number(sample.get("actual_y"))
            z = to_number(sample.get("actual_z"))
            time_s = to_number(sample.get("time"))

            if None in (x, y, z):
                invalid_samples += 1
                continue

            valid_samples += 1
            current_xyz = (x, y, z)

            if first_xyz is None:
                first_xyz = current_xyz

            last_xyz = current_xyz

            if time_s is not None:
                if first_time is None:
                    first_time = time_s
                last_time = time_s

            added_risk = 0.0

            # ------------------------------------------------
            # DISTANCE ACTUALLY TRAVELLED
            # ------------------------------------------------

            if previous_xyz is not None:
                dx = x - previous_xyz[0]
                dy = y - previous_xyz[1]
                dz = z - previous_xyz[2]

                segment_2d = math.hypot(dx, dy)
                segment_3d = math.sqrt(
                    dx * dx + dy * dy + dz * dz
                )

                distance_2d += segment_2d
                distance_3d += segment_3d

                # Sample between logged points to avoid
                # missing grid cells on a longer movement.
                subdivisions = max(
                    1,
                    math.ceil(
                        segment_2d / (resolution / 4.0)
                    )
                )

                positions = (
                    (
                        previous_xyz[0] + dx * k / subdivisions,
                        previous_xyz[1] + dy * k / subdivisions,
                    )
                    for k in range(1, subdivisions + 1)
                )

            else:
                positions = [(x, y)]

            # ------------------------------------------------
            # ACCUMULATE RISK WHEN ENTERING A GRID CELL
            # ------------------------------------------------

            for px, py in positions:
                cell = position_to_cell(
                    px, py, bounds, resolution
                )

                # Consecutive samples in the same cell
                # do not incur risk repeatedly.
                if cell == previous_cell:
                    continue

                previous_cell = cell
                unique_cells.add(cell)

                cell_risk = get_cell_risk(cell, risk_grid)

                if cell_risk is None:
                    unknown_risk_visits += 1
                    continue

                total_risk += cell_risk
                added_risk += cell_risk
                risk_cell_visits += 1

            # ------------------------------------------------
            # OPTIONAL DETAILED POINT OUTPUT
            # ------------------------------------------------

            if detail_writer is not None:
                actual_cell = position_to_cell(
                    x, y, bounds, resolution
                )

                current_risk = get_cell_risk(
                    actual_cell, risk_grid
                )

                detail_writer.writerow({
                    **metadata,
                    "time_s": time_s,
                    "actual_x": x,
                    "actual_y": y,
                    "actual_z": z,
                    "grid_x": actual_cell[0],
                    "grid_y": actual_cell[1],
                    "position_risk": current_risk,
                    "added_risk": added_risk,
                    "cumulative_risk": total_risk,
                })

            previous_xyz = current_xyz

    # Do not create an execution row without positions.
    if valid_samples == 0:
        return None

    risk_complete = unknown_risk_visits == 0

    return {
        **metadata,
        "total_samples": samples,
        "valid_position_samples": valid_samples,
        "invalid_position_samples": invalid_samples,
        "first_time_s": first_time,
        "last_time_s": last_time,
        "recorded_duration_s": (
            last_time - first_time
            if first_time is not None and last_time is not None
            else ""
        ),
        "start_x": first_xyz[0],
        "start_y": first_xyz[1],
        "start_z": first_xyz[2],
        "end_x": last_xyz[0],
        "end_y": last_xyz[1],
        "end_z": last_xyz[2],
        "actual_distance_2d_m": distance_2d,
        "actual_distance_3d_m": distance_3d,
        "risk_cell_visits": risk_cell_visits,
        "unique_grid_cells": len(unique_cells),
        "unknown_risk_visits": unknown_risk_visits,
        "risk_complete": risk_complete,
        "accumulated_risk": (
            total_risk if risk_complete else ""
        ),
        "partial_accumulated_risk": total_risk,
        "risk_remaining": (
            float(metadata["D"]) - total_risk
            if risk_complete and metadata["D"] != ""
            else ""
        ),
    }


# ============================================================
# CSV COLUMNS
# ============================================================

SUMMARY_COLUMNS = [
    "terrain",
    "planner",
    "scenario",
    "D",
    "alpha",
    "seed",
    "run_id",
    "total_samples",
    "valid_position_samples",
    "invalid_position_samples",
    "first_time_s",
    "last_time_s",
    "recorded_duration_s",
    "start_x",
    "start_y",
    "start_z",
    "end_x",
    "end_y",
    "end_z",
    "actual_distance_2d_m",
    "actual_distance_3d_m",
    "risk_cell_visits",
    "unique_grid_cells",
    "unknown_risk_visits",
    "risk_complete",
    "accumulated_risk",
    "partial_accumulated_risk",
    "risk_remaining",
    "source",
]

DETAIL_COLUMNS = [
    "terrain",
    "planner",
    "scenario",
    "D",
    "alpha",
    "seed",
    "run_id",
    "source",
    "time_s",
    "actual_x",
    "actual_y",
    "actual_z",
    "grid_x",
    "grid_y",
    "position_risk",
    "added_risk",
    "cumulative_risk",
]


# ============================================================
# PROCESS EACH TERRAIN
# ============================================================

def process_terrain(terrain):

    print("\n" + "=" * 65)
    print(f"PROCESSING: {terrain.name}")
    print("=" * 65)

    grid_file = terrain / "terrain_grid.json"

    if not grid_file.exists():
        print("SKIPPED: terrain_grid.json not found")
        return

    # Load and calculate the ORIGINAL risk model.
    print("Loading terrain risk model...")

    terrain_data = load_terrain_grid(grid_file)
    features = prepare_risk_model(terrain_data)

    risk_grid = features.weighted_risk_grid
    bounds = terrain_data.bounds
    resolution = terrain_data.resolution

    # Find PyBullet executions, including CMDP and baselines.
    files = sorted(
        p for p in terrain.rglob("*")
        if p.is_file()
        and re.fullmatch(
            r"pybullet_execution(?:\(\d+\))?\.csv",
            p.name,
            re.IGNORECASE
        )
        and not any(
            part.lower().startswith("_stale_")
            for part in p.relative_to(terrain).parts
        )
    )

    print(f"Execution files found: {len(files)}")

    rows = []

    detail_file = terrain / "pybullet_risk_trace.csv"
    detail_handle = None

    try:
        detail_writer = None

        if WRITE_POINT_DETAILS:
            detail_handle = detail_file.open(
                "w", newline="", encoding="utf-8-sig"
            )
            detail_writer = csv.DictWriter(
                detail_handle,
                fieldnames=DETAIL_COLUMNS
            )
            detail_writer.writeheader()

        for i, file in enumerate(files, 1):
            try:
                result = calculate_execution(
                    file,
                    terrain,
                    risk_grid,
                    bounds,
                    resolution,
                    detail_writer
                )

                if result is None:
                    print(f"[{i}] No positions: {file}")
                    continue

                rows.append(result)

                print(
                    f"[{i}/{len(files)}] "
                    f"{result['planner']} | "
                    f"{result['scenario']} | "
                    f"D={result['D']} | "
                    f"seed={result['seed']} | "
                    f"risk={result['accumulated_risk']}"
                )

            except Exception as error:
                print(f"ERROR: {file}: {error}")

    finally:
        if detail_handle is not None:
            detail_handle.close()

    # --------------------------------------------------------
    # SAVE ONE SUMMARY CSV PER TERRAIN
    # --------------------------------------------------------

    output_file = terrain / "pybullet_accumulated_risk.csv"

    with output_file.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=SUMMARY_COLUMNS
        )
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # COLLECT RENAMED COPIES IN ONE FOLDER
    # --------------------------------------------------------

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    collected_name = (
        f"{terrain.name}_pybullet_accumulated_risk.csv"
    )

    shutil.copy2(
        output_file,
        OUTPUT_FOLDER / collected_name
    )

    if WRITE_POINT_DETAILS:
        shutil.copy2(
            detail_file,
            OUTPUT_FOLDER /
            f"{terrain.name}_pybullet_risk_trace.csv"
        )

    print(f"\nSaved: {output_file}")
    print(f"Collected: {collected_name}")
    print(f"Executions: {len(rows)}")


# ============================================================
# MAIN
# ============================================================

def main():

    terrains = sorted(
        [
            p for p in ROOT.iterdir()
            if p.is_dir()
            and TERRAIN_PATTERN.fullmatch(p.name)
        ],
        key=lambda p: int(p.name.split("_")[1])
    )

    if not terrains:
        print("No terrain folders found.")
        return

    for terrain in terrains:
        try:
            process_terrain(terrain)
        except Exception as error:
            print(f"FAILED {terrain.name}: {error}")

    print("\n" + "=" * 65)
    print("ALL PYBULLET RISK RESULTS COLLECTED")
    print(f"Output: {OUTPUT_FOLDER}")
    print("=" * 65)


if __name__ == "__main__":
    main()

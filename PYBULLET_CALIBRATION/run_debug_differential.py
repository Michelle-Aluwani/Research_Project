import argparse
import bisect
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Shared, name-based OBJ reader (also detects Blender Y-up exports and
# converts them to the Z-up frame used by the grid, planners and PyBullet).
from cmdp_refactored import (
    classify_obj_group,
    group_faces_by_type,
    read_obj_groups,
)


# ============================================================
# CMDP / BASELINE PATH -> PHYSICAL PYBULLET EXECUTION
# ============================================================
#
# Command line (as called by experiment_runner.py / baseline_runner.py):
#
#   python run.py --terrain-dir <terrain_dir> --cmdp-path <path.csv>
#                 --scenario-id <id> --risk-budget <D> --seed <n>
#                 --output-dir <dir> [--headless]
#
# --headless uses PyBullet DIRECT mode (no window, no waiting at the end).
# A robot fall is a valid RESULT: it is recorded and the exit code is 0.

parser = argparse.ArgumentParser(
    description="Execute a planned path in PyBullet (one Monte Carlo trial)."
)
parser.add_argument("--terrain-dir", type=Path, required=True)
parser.add_argument("--cmdp-path", type=Path, required=True)
parser.add_argument("--scenario-id", default="manual")
parser.add_argument("--risk-budget", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--terrain-obj", type=Path, default=None,
                    help="Override the terrain OBJ (default: the single "
                         ".obj in --terrain-dir).")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--tracking-mode", choices=["original", "segment"], default="original",
                    help="original=unchanged waypoint tracking; segment=path projection and recovery")
parser.add_argument("--turn-method", choices=["original", "differential"], default="differential",
                    help="differential uses opposite-direction strides at large heading errors")
parser.add_argument("--turn-sign", type=int, choices=[-1, 1], default=1,
                    help="Reverse differential turning direction if physical yaw response is reversed")
parser.add_argument("--steering-test", choices=["auto", "positive", "negative", "zero"], default="auto",
                    help="Optional fixed-command actuator test; use auto for path tracking")
args = parser.parse_args()

HEADLESS = bool(args.headless)

import pybullet as p  # noqa: E402
import pybullet_data  # noqa: E402

TERRAIN_DIR = args.terrain_dir.resolve()
CMDP_PATH = args.cmdp_path.resolve()
TERRAIN_GRID_JSON = TERRAIN_DIR / "terrain_grid.json"

OUTPUT_DIR = (
    args.output_dir.resolve()
    if args.output_dir is not None
    else TERRAIN_DIR / "run_output"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "pybullet_execution.csv"
STUMBLE_CSV = OUTPUT_DIR / "stumble_log.csv"
RESULT_JSON = OUTPUT_DIR / (
    f"result_{args.scenario_id}_D{args.risk_budget:g}_"
    f"seed{args.seed:03d}.json"
)


def find_terrain_obj(terrain_dir):
    """The single source OBJ in the terrain folder (newterrain_N.obj, ...)."""
    candidates = [
        q for q in sorted(terrain_dir.glob("*.obj"))
        if not q.name.startswith("pybullet_")
        and not q.stem.endswith("_zup")
    ]
    if len(candidates) == 1:
        return candidates[0]
    preferred = [q for q in candidates if q.name.startswith("newterrain")]
    if len(preferred) == 1:
        return preferred[0]
    raise FileNotFoundError(
        f"Expected exactly one source .obj in {terrain_dir}, "
        f"found {[q.name for q in candidates]}"
    )


TERRAIN_OBJ = (
    args.terrain_obj.resolve()
    if args.terrain_obj is not None
    else find_terrain_obj(TERRAIN_DIR)
)


# ============================================================
# SETTINGS
# ============================================================

TIME_STEP = 1.0 / 240.0

MAX_SIMULATION_TIME = 400.0

# The rock OBJ has 2-6 cm bumps at ~0.14 m vertex spacing. This gait lifts
# its feet only ~2-3 cm, so on the raw mesh the robot marches on the spot.
# With this switch ON, the rock COLLISION surface is rebuilt from
# terrain_grid.json "height_grid" (the same 0.5 m surface the planner uses).
# The visual mesh, the grid, the path and all friction values are unchanged.
SMOOTH_ROCK_COLLISION = True   # 58 m at ~0.5-0.8 m/s needs ~100-150 s

WAYPOINT_TOLERANCE = 0.30

DESIRED_SPEED = 0.30

# Laikago motor control (PyBullet position control)
MOTOR_FORCE = 40.0
MOTOR_KP = 0.6
MOTOR_KD = 0.6

# Trot gait (found by sweep; see notes). NOTE: this gait walks toward the
# robot's -x (BACK) axis, so the robot is spawned facing AWAY from the path
# direction (BODY_YAW_OFFSET) and "forward along the path" = body -x.
GAIT_FREQUENCY = 5.5        # Hz
GAIT_SWING = 0.06           # rad, upper-leg swing amplitude
GAIT_LIFT = 0.10            # rad, lower-leg lift amplitude
STANCE_EXT = 0.5            # extended stance (headroom for body levelling)
LEVEL_K_ROLL = 1.6          # roll  feedback: lengthen the low-side legs
LEVEL_K_PITCH = 1.6         # pitch feedback: lengthen the low-end legs
ROLL_PITCH_FILTER_ALPHA = 0.15   # EMA smoothing on roll/pitch fed into leveling
LEVEL_CLAMP_DEG = 20.0           # clamp filtered roll/pitch before using it
BODY_YAW_OFFSET = math.pi   # body yaw = travel heading + pi

# Heading controller
LOOKAHEAD_DISTANCE = 0.8    # m, steer toward a point this far along the path
K_YAW = 2.0                 # turn command per rad of heading error
K_YAW_RATE = 0.0            # (unused)
MAX_TURN = 1.0              # |turn| limit

# Camera
CAMERA_DISTANCE = 8.0
CAMERA_PITCH = -55

# Fall detection
MAX_PITCH_DEG = 45.0
MAX_ROLL_DEG = 45.0

MIN_BODY_Z_MARGIN = -0.15


# ============================================================
# HIDDEN TERRAIN FRICTION MODEL
# ============================================================

# These values are used ONLY by the physical PyBullet model.
# The CMDP does not use these values.

ROCK_FRICTION = 0.85

# Mud far from water
MUD_FRICTION_DRY = 0.65

# Mud immediately beside water
MUD_FRICTION_WET = 0.12

# Water
WATER_FRICTION = 0.05

# Distance over which mud changes from wet -> dry
MUD_WATER_INFLUENCE_DISTANCE = 4.0

# ============================================================
# STUMBLE / SLIP MODEL
# ============================================================
# "Stumble" = a visible wobble/deviation off the path that the body-level
# controller recovers from (robot stays up). "Slip" = a bigger disturbance
# applied in the risky part of a mud patch; it can still be recovered from
# but is tuned to often end in an actual fall.
#
# Inside a mud patch, cells are given a 0..1 ratio (MUD_RATIO_GRID, built
# below): 0 = right at that patch's wet edge (closest to water), 1 = the
# driest / furthest-in point of THAT SPECIFIC patch. This is a ratio, not
# a fixed metre cutoff, because mud patches are not all the same size.
#
# ratio < MUD_STUMBLE_RATIO  -> stumble zone  (wet mud near water)
# ratio >= MUD_STUMBLE_RATIO -> slip zone     (drier mud, deeper in patch)

MUD_STUMBLE_RATIO = 0.25   # ~ first quarter of each mud patch, from its wet edge

# Odds per second of triggering an event while continuously standing in
# each zone (checked every physics tick as chance_per_sec * TIME_STEP).
MUD_STUMBLE_CHANCE_PER_SEC = 0.6
MUD_SLIP_CHANCE_PER_SEC = 0.4

# Rock: a "stumble" (not a fall) can also happen on a steep slope, even
# though rock has high friction, just from bad foot placement on incline.
ROCK_STEEP_SLOPE_DEG = 15.0
ROCK_STUMBLE_CHANCE_PER_SEC = 0.3

# Minimum time between two hazard events, so one hazard doesn't spam the
# log every physics tick (240 Hz) while the robot dwells in one zone.
HAZARD_EVENT_COOLDOWN_S = 1.5

# External push applied to the robot base on a stumble/slip event.
STUMBLE_PUSH_FORCE = 25.0    # N
STUMBLE_PUSH_TORQUE = 6.0    # N*m
SLIP_PUSH_FORCE = 70.0       # N
SLIP_PUSH_TORQUE = 18.0      # N*m

RNG_SEED = int(args.seed)
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


# ============================================================
# CHECK FILES
# ============================================================

for required_file in [
    TERRAIN_OBJ,
    CMDP_PATH,
    TERRAIN_GRID_JSON
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"Could not find:\n{required_file}"
        )


# ============================================================
# LOAD CMDP WAYPOINTS
# ============================================================

waypoints = []

with open(
    CMDP_PATH,
    "r",
    newline="",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    print("CSV columns:", reader.fieldnames)

    required_columns = {"x", "y", "z"}

    if not required_columns.issubset(
        set(reader.fieldnames or [])
    ):

        raise ValueError(
            "cmdp_path_world.csv must contain "
            "x, y and z columns."
        )

    for row in reader:

        waypoints.append(
            (
                float(row["x"]),
                float(row["y"]),
                float(row["z"])
            )
        )


if len(waypoints) < 2:

    raise ValueError(
        "CMDP path must contain at least two waypoints."
    )


START = waypoints[0]
GOAL = waypoints[-1]


# ============================================================
# LOAD TERRAIN GRID
# ============================================================

with open(
    TERRAIN_GRID_JSON,
    "r",
    encoding="utf-8"
) as f:

    terrain_data = json.load(f)


metadata = terrain_data["metadata"]

TERRAIN_GRID = terrain_data["grid"]

GRID_WIDTH = metadata["grid_width"]
GRID_HEIGHT = metadata["grid_height"]

GRID_RESOLUTION = metadata["resolution_m"]


# ============================================================
# TERRAIN BOUNDS
# ============================================================

BOUNDS = metadata["bounds_m"]

MIN_X = BOUNDS["min_x"]
MAX_X = BOUNDS["max_x"]

MIN_Y = BOUNDS["min_y"]
MAX_Y = BOUNDS["max_y"]

MIN_Z = BOUNDS["min_z"]
MAX_Z = BOUNDS["max_z"]


print()
print("========================================")
print("TERRAIN GRID")
print("========================================")

print(
    f"Grid:       {GRID_WIDTH} x {GRID_HEIGHT}"
)

print(
    f"Resolution: {GRID_RESOLUTION:.3f} m"
)

print(
    f"Bounds X:   {MIN_X:.3f} -> {MAX_X:.3f}"
)

print(
    f"Bounds Y:   {MIN_Y:.3f} -> {MAX_Y:.3f}"
)


# ============================================================
# GRID LOOKUP
# ============================================================

def world_to_grid(x, y):

    gx = int(
        math.floor(
            (x - MIN_X) /
            GRID_RESOLUTION
        )
    )

    gy = int(
        math.floor(
            (y - MIN_Y) /
            GRID_RESOLUTION
        )
    )

    if (
        gx < 0 or
        gx >= GRID_WIDTH or
        gy < 0 or
        gy >= GRID_HEIGHT
    ):

        return None

    return gx, gy


def terrain_at_world(x, y):

    cell = world_to_grid(x, y)

    if cell is None:
        return None

    gx, gy = cell

    return int(TERRAIN_GRID[gx][gy])


# ============================================================
# SLOPE GRID (for detecting steep rock -> stumble)
# ============================================================

def _fill_nan_grid(grid):

    grid = grid.copy()

    while np.isnan(grid).any():

        pad = np.pad(grid, 1, constant_values=np.nan)

        nb = np.stack([
            pad[:-2, 1:-1], pad[2:, 1:-1],
            pad[1:-1, :-2], pad[1:-1, 2:]
        ])

        cnt = np.sum(~np.isnan(nb), axis=0)
        s = np.nansum(nb, axis=0)

        fill = np.isnan(grid) & (cnt > 0)

        if not fill.any():
            grid[np.isnan(grid)] = np.nanmean(grid)
            break

        grid[fill] = s[fill] / cnt[fill]

    return grid


HEIGHT_GRID = _fill_nan_grid(
    np.array(terrain_data["height_grid"], dtype=float)
)

_slope_gy, _slope_gx = np.gradient(HEIGHT_GRID, GRID_RESOLUTION)
SLOPE_GRID_DEG = np.degrees(np.arctan(np.hypot(_slope_gx, _slope_gy)))


def slope_deg_at_world(x, y):
    """Local terrain slope (degrees) from the 0.5 m height grid."""

    cell = world_to_grid(x, y)

    if cell is None:
        return 0.0

    gx, gy = cell

    return float(SLOPE_GRID_DEG[gx, gy])


# ============================================================
# PRECOMPUTE DISTANCE FROM MUD CELLS TO WATER
# ============================================================

print()
print("Calculating mud distance from water...")


water_cells = []

for x in range(GRID_WIDTH):

    for y in range(GRID_HEIGHT):

        if int(TERRAIN_GRID[x][y]) == 2:

            water_cells.append(
                (x, y)
            )


if len(water_cells) == 0:

    # Some terrains have no water. Mud is then "dry" everywhere.
    print("No water cells in this terrain (mud will be treated as dry).")


# ============================================================
# DISTANCE TO WATER
# ============================================================

# The terrain grid is only 163 x 166, with 399 water cells.
# For this size, the direct calculation is sufficiently small
# and avoids introducing another dependency.

def distance_to_water_world(x, y):

    cell = world_to_grid(x, y)

    if cell is None:

        return float("inf")

    gx, gy = cell

    best_distance = float("inf")

    for wx, wy in water_cells:

        dx = (
            gx - wx
        ) * GRID_RESOLUTION

        dy = (
            gy - wy
        ) * GRID_RESOLUTION

        distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        if distance < best_distance:

            best_distance = distance

    return best_distance


print(
    f"Water cells: {len(water_cells)}"
)


# ============================================================
# MUD PATCH RATIO GRID (stumble vs. slip-and-fall zones)
# ============================================================
# Mud patches are not all the same size, so instead of a fixed
# "stumble within N metres of water" cutoff, every mud cell gets a
# ratio 0..1 that is relative to ITS OWN connected patch: 0 = right at
# that patch's wet edge (closest to water), 1 = the driest / deepest
# point of that same patch.

print()
print("Building mud patch ratio grid...")

_tg_arr = np.array(TERRAIN_GRID, dtype=int)

_mud_cell_distance = {}   # (gx, gy) -> distance to water, metres

for _gx in range(GRID_WIDTH):

    for _gy in range(GRID_HEIGHT):

        if _tg_arr[_gx, _gy] == 1:

            _wx = MIN_X + (_gx + 0.5) * GRID_RESOLUTION
            _wy = MIN_Y + (_gy + 0.5) * GRID_RESOLUTION

            _mud_cell_distance[(_gx, _gy)] = distance_to_water_world(_wx, _wy)


_visited = set()
MUD_RATIO_GRID = {}   # (gx, gy) -> ratio 0..1
mud_patch_count = 0

for _start in _mud_cell_distance:

    if _start in _visited:
        continue

    # flood-fill this connected mud patch (4-connectivity)
    _patch = []
    _queue = [_start]
    _visited.add(_start)

    while _queue:

        _cx, _cy = _queue.pop()
        _patch.append((_cx, _cy))

        for _nx, _ny in ((_cx + 1, _cy), (_cx - 1, _cy),
                          (_cx, _cy + 1), (_cx, _cy - 1)):

            if (_nx, _ny) in _mud_cell_distance and (_nx, _ny) not in _visited:
                _visited.add((_nx, _ny))
                _queue.append((_nx, _ny))

    _patch_max_dist = max(_mud_cell_distance[c] for c in _patch)
    _patch_max_dist = max(_patch_max_dist, 1e-6)

    for _c in _patch:
        if math.isfinite(_patch_max_dist):
            MUD_RATIO_GRID[_c] = _mud_cell_distance[_c] / _patch_max_dist
        else:
            MUD_RATIO_GRID[_c] = 1.0   # no water at all: dry mud

    mud_patch_count += 1


print(f"Mud patches found: {mud_patch_count}")


def mud_ratio_at_world(x, y):
    """0 = this mud patch's wet edge, 1 = its driest/deepest point.
    Returns None if (x, y) is not currently on mud."""

    cell = world_to_grid(x, y)

    if cell is None:
        return None

    return MUD_RATIO_GRID.get(cell)


# ============================================================
# MUD FRICTION
# ============================================================

def mud_friction_from_distance(distance):

    if not math.isfinite(distance):

        return MUD_FRICTION_DRY

    influence = (
        MUD_WATER_INFLUENCE_DISTANCE
    )

    if distance >= influence:

        return MUD_FRICTION_DRY

    ratio = (
        distance /
        influence
    )

    return (
        MUD_FRICTION_WET
        +
        ratio *
        (
            MUD_FRICTION_DRY
            -
            MUD_FRICTION_WET
        )
    )


def current_terrain_properties(x, y):

    terrain_id = terrain_at_world(
        x,
        y
    )

    if terrain_id == 0:

        return (
            "rock",
            ROCK_FRICTION,
            None
        )

    if terrain_id == 1:

        distance = (
            distance_to_water_world(
                x,
                y
            )
        )

        friction = (
            mud_friction_from_distance(
                distance
            )
        )

        return (
            "mud",
            friction,
            distance
        )

    if terrain_id == 2:

        return (
            "water",
            WATER_FRICTION,
            0.0
        )

    return (
        "unknown",
        ROCK_FRICTION,
        None
    )


# ============================================================
# OBJ GROUPS (by object name, Z-up)
# ============================================================

print()
print("========================================")
print("READING OBJ GROUPS")
print("========================================")

obj_vertices, obj_groups, obj_info = read_obj_groups(TERRAIN_OBJ)

print(f"OBJ: {TERRAIN_OBJ.name}")
print(f"Up axis in file: {obj_info['up_axis_in_file']} "
      f"(converted to Z-up: {obj_info['converted']})")
print(f"Vertices: {len(obj_vertices)}")

for name, faces in obj_groups.items():
    print(f"{name}: {len(faces)} triangles "
          f"[{classify_obj_group(name) or 'ignored'}]")

_faces_by_type = group_faces_by_type(obj_groups)

rock_faces = _faces_by_type.get("rock")
mud_faces = _faces_by_type.get("mud")
water_faces = _faces_by_type.get("water")
wall_faces = _faces_by_type.get("wall")

if rock_faces is None:
    raise ValueError("Could not find Rock groups in OBJ (name: terrain_rock).")

print()
print("Combined terrain meshes:")
print(f"Rock:  {len(rock_faces)} triangles")
print(f"Mud:   {0 if mud_faces is None else len(mud_faces)} triangles")
print(f"Water: {0 if water_faces is None else len(water_faces)} triangles")
print(f"Wall:  {0 if wall_faces is None else len(wall_faces)} triangles")


# ============================================================
# WRITE COLLISION OBJ
# ============================================================

def write_collision_obj(
    output_path,
    vertices,
    faces
):

    faces = np.asarray(faces, dtype=np.int64)

    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)

    mesh_vertices = vertices[used]
    new_faces = inverse.reshape(-1, 3) + 1

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("# PyBullet terrain collision mesh\n")

        f.write(
            "\n".join(
                f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}"
                for v in mesh_vertices
            )
        )
        f.write("\n")

        f.write(
            "\n".join(
                f"f {a} {b} {c}" for a, b, c in new_faces
            )
        )
        f.write("\n")

    return len(mesh_vertices)


def write_zup_visual_obj(src, dst):
    """Copy of the OBJ with vertices/normals rotated Y-up -> Z-up
    ((x, y, z) -> (x, -z, y)), for the GUI visual only."""

    with open(src, "r", encoding="utf-8", errors="ignore") as fin, \
            open(dst, "w", encoding="utf-8") as fout:

        for line in fin:

            if line.startswith("v ") or line.startswith("vn "):

                parts = line.split()
                x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
                fout.write(f"{parts[0]} {x:.9f} {-z:.9f} {y:.9f}\n")

            else:

                fout.write(line)

    mtl = Path(src).with_suffix(".mtl")

    if mtl.is_file():

        (Path(dst).parent / mtl.name).write_bytes(mtl.read_bytes())


# ============================================================
# PYBULLET COLLISION OBJ FILES  (per run, in the output folder, so
# parallel jobs never overwrite each other's files)
# ============================================================

ROCK_COLLISION_OBJ = OUTPUT_DIR / "pybullet_rock_collision.obj"
MUD_COLLISION_OBJ = OUTPUT_DIR / "pybullet_mud_collision.obj"
WATER_COLLISION_OBJ = OUTPUT_DIR / "pybullet_water_collision.obj"
WALL_COLLISION_OBJ = OUTPUT_DIR / "pybullet_wall_collision.obj"


# ============================================================
# START PYBULLET
# ============================================================

print()
print("========================================")
print("STARTING PYBULLET")
print("========================================")

p.connect(
    p.DIRECT if HEADLESS else p.GUI
)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath()
)

p.setGravity(
    0,
    0,
    -9.81
)

p.setTimeStep(
    TIME_STEP
)

p.setRealTimeSimulation(
    0
)

p.setPhysicsEngineParameter(
    numSolverIterations=100
)


# ============================================================
# LOAD TERRAIN VISUAL
# ============================================================

print()
print("Loading terrain visual...")

if not HEADLESS:

    _visual_obj = TERRAIN_OBJ

    if obj_info["converted"]:

        _visual_obj = OUTPUT_DIR / "terrain_visual_zup.obj"
        write_zup_visual_obj(TERRAIN_OBJ, _visual_obj)

    terrain_visual = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(_visual_obj),
        meshScale=[1, 1, 1]
    )

    terrain_visual_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=terrain_visual,
        basePosition=[0, 0, 0]
    )


# ============================================================
# CREATE COLLISION OBJ FILES
# ============================================================

print()
print("========================================")
print("CREATING COLLISION MESH FILES")
print("========================================")


if SMOOTH_ROCK_COLLISION:

    # The raw rock mesh is not used for collision when the smoothed
    # surface is on (it used to be created and then switched off).
    rock_vertex_count = 0

else:

    rock_vertex_count = write_collision_obj(
        ROCK_COLLISION_OBJ, obj_vertices, rock_faces
    )

mud_vertex_count = (
    0 if mud_faces is None
    else write_collision_obj(MUD_COLLISION_OBJ, obj_vertices, mud_faces)
)

water_vertex_count = (
    0 if water_faces is None
    else write_collision_obj(WATER_COLLISION_OBJ, obj_vertices, water_faces)
)

wall_vertex_count = (
    0 if wall_faces is None
    else write_collision_obj(WALL_COLLISION_OBJ, obj_vertices, wall_faces)
)

print(f"Rock collision vertices: {rock_vertex_count}")
print(f"Mud collision vertices: {mud_vertex_count}")
print(f"Water collision vertices: {water_vertex_count}")
print(f"Wall collision vertices: {wall_vertex_count}")


# ============================================================
# LOAD TERRAIN COLLISION BY TYPE
# ============================================================

print()
print("Creating terrain collision bodies...")


# ------------------------------------------------------------
# ROCK
# ------------------------------------------------------------

rock_collision = None
rock_body = None

if not SMOOTH_ROCK_COLLISION:

    rock_collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=str(ROCK_COLLISION_OBJ),
        meshScale=[1, 1, 1],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    rock_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=rock_collision,
        basePosition=[0, 0, 0]
    )

if SMOOTH_ROCK_COLLISION:

    _hg = np.array(terrain_data["height_grid"], dtype=float)

    # fill missing cells from valid neighbours
    while np.isnan(_hg).any():
        _pad = np.pad(_hg, 1, constant_values=np.nan)
        _nb = np.stack([
            _pad[:-2, 1:-1], _pad[2:, 1:-1],
            _pad[1:-1, :-2], _pad[1:-1, 2:]
        ])
        _cnt = np.sum(~np.isnan(_nb), axis=0)
        _sum = np.nansum(_nb, axis=0)
        _fill = np.isnan(_hg) & (_cnt > 0)
        if not _fill.any():
            _hg[np.isnan(_hg)] = np.nanmean(_hg)
            break
        _hg[_fill] = _sum[_fill] / _cnt[_fill]

    _nx, _ny = _hg.shape
    _tg = np.array(TERRAIN_GRID, dtype=int)  # same [gx][gy] shape as _hg

    # Don't cut holes over mud/water: that left a real gap in the collision
    # (no rock quad, and the real mud/water mesh doesn't necessarily reach
    # the exact same footprint), which is what the robot was tripping on
    # right at the boundary. Instead just dip the rock plane a few cm below
    # the recorded surface height under mud/water cells, so the actual
    # mud/water collision mesh (unchanged, at its real height) sits just
    # above it and is what actually gets contacted there. Geometry stays
    # continuous - no lip, no hole.
    MUD_WATER_CLEARANCE = 0.05  # m
    _hg_rock = _hg.copy()
    # Only mud (1) / water (2) cells are dipped; wall cells (3) are not.
    _hg_rock[np.isin(_tg, (1, 2))] -= MUD_WATER_CLEARANCE

    _smooth_obj = OUTPUT_DIR / "pybullet_rock_smooth_collision.obj"

    with open(_smooth_obj, "w") as _f:
        for _i in range(_nx):
            for _j in range(_ny):
                _f.write(
                    f"v {MIN_X + (_i + 0.5) * GRID_RESOLUTION:.4f} "
                    f"{MIN_Y + (_j + 0.5) * GRID_RESOLUTION:.4f} "
                    f"{_hg_rock[_i, _j]:.4f}\n"
                )
        for _i in range(_nx - 1):
            for _j in range(_ny - 1):
                _a = _i * _ny + _j + 1
                _b = (_i + 1) * _ny + _j + 1
                _c = (_i + 1) * _ny + _j + 2
                _d = _i * _ny + _j + 2
                _f.write(f"f {_a} {_b} {_c}\nf {_a} {_c} {_d}\n")

    # use the smooth surface as the rock collision
    rock_collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=str(_smooth_obj),
        meshScale=[1, 1, 1],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    rock_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=rock_collision,
        basePosition=[0, 0, 0]
    )

    print("Rock collision: smoothed 0.5 m surface from height_grid")


# ------------------------------------------------------------
# MUD
# ------------------------------------------------------------

mud_body = None

if mud_faces is not None:

    mud_collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=str(MUD_COLLISION_OBJ),
        meshScale=[1, 1, 1],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    mud_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=mud_collision,
        basePosition=[0, 0, 0]
    )


# ------------------------------------------------------------
# WATER
# ------------------------------------------------------------

water_body = None

if water_faces is not None:

    water_collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=str(WATER_COLLISION_OBJ),
        meshScale=[1, 1, 1],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    water_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=water_collision,
        basePosition=[0, 0, 0]
    )


# ------------------------------------------------------------
# WALLS  (static collision, same construction as mud / water; the
# grid and planners treat these cells as non-traversable)
# ------------------------------------------------------------

wall_body = None

if wall_faces is not None:

    wall_collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=str(WALL_COLLISION_OBJ),
        meshScale=[1, 1, 1],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    wall_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=wall_collision,
        basePosition=[0, 0, 0]
    )


# ============================================================
# INITIAL FRICTION
# ============================================================

p.changeDynamics(
    rock_body,
    -1,
    lateralFriction=ROCK_FRICTION,
    rollingFriction=0.01,
    spinningFriction=0.01
)

if mud_body is not None:

    p.changeDynamics(
        mud_body,
        -1,
        lateralFriction=MUD_FRICTION_DRY,
        rollingFriction=0.01,
        spinningFriction=0.01
    )

if water_body is not None:

    p.changeDynamics(
        water_body,
        -1,
        lateralFriction=WATER_FRICTION,
        rollingFriction=0.001,
        spinningFriction=0.001
    )

if wall_body is not None:

    p.changeDynamics(
        wall_body,
        -1,
        lateralFriction=ROCK_FRICTION,
        rollingFriction=0.01,
        spinningFriction=0.01
    )


print()
print("Rock collision loaded.")
print(f"Mud collision loaded: {mud_body is not None}")
print(f"Water collision loaded: {water_body is not None}")
print(f"Wall collision loaded: {wall_body is not None}")


# ============================================================
# DRAW CMDP PATH
# ============================================================

print()
print("Drawing CMDP path...")


for i in range(
    0 if HEADLESS else len(waypoints) - 1
):

    p.addUserDebugLine(
        waypoints[i],
        waypoints[i + 1],
        lineWidth=3,
        lifeTime=0
    )


# ============================================================
# START MARKER
# ============================================================

if not HEADLESS:

    start_marker_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.25,
        rgbaColor=[1, 0, 0, 1]
    )

    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=start_marker_visual,
        basePosition=START
    )


# ============================================================
# GOAL MARKER
# ============================================================

if not HEADLESS:

    goal_marker_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.30,
        rgbaColor=[0, 1, 0, 1]
    )

    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_marker_visual,
        basePosition=GOAL
    )


# ============================================================
# LOAD MINITAUR
# ============================================================

print()
print("Loading Laikago...")


def raycast_ground_z(x, y):
    """Terrain surface height under (x, y). Called BEFORE the robot exists,
    so the only collision bodies are the terrain meshes."""
    hit = p.rayTest([x, y, MAX_Z + 2.0], [x, y, MIN_Z - 2.0])[0]
    if hit[0] < 0:
        return None
    return hit[3][2]


WAYPOINT_GROUND_Z = []
for wx, wy, wz in waypoints:
    gz = raycast_ground_z(wx, wy)
    WAYPOINT_GROUND_Z.append(gz if gz is not None else wz - 0.4)


def ground_z_near(x, y):
    """Ground height at the waypoint nearest to (x, y) (used for fall check)."""
    best = min(
        range(len(waypoints)),
        key=lambda k: (waypoints[k][0] - x) ** 2 + (waypoints[k][1] - y) ** 2
    )
    return WAYPOINT_GROUND_Z[best]


# Face the robot along the first ~1 m of the path
start_heading = 0.0
for wx, wy, _ in waypoints[1:]:
    if math.hypot(wx - START[0], wy - START[1]) >= 1.0:
        start_heading = math.atan2(wy - START[1], wx - START[0])
        break

print(f"Ground z at start: {WAYPOINT_GROUND_Z[0]:.3f} "
      f"(path z {START[2]:.3f})")
print(f"Start heading: {math.degrees(start_heading):.1f} deg")

start_body_yaw = start_heading + BODY_YAW_OFFSET


def ground_slopes_body_frame(x, y, yaw):
    """Terrain slope under the robot, in the robot's own frame (4 ray casts)."""
    c, s = math.cos(yaw), math.sin(yaw)

    def gz(fx, fy):
        g = raycast_ground_z(x + c * fx - s * fy, y + s * fx + c * fy)
        return g if g is not None else WAYPOINT_GROUND_Z[0]

    dz_dx = (gz(0.3, 0.0) - gz(-0.3, 0.0)) / 0.6
    dz_dy = (gz(0.0, 0.15) - gz(0.0, -0.15)) / 0.3
    return dz_dx, dz_dy


_dzdx, _dzdy = ground_slopes_body_frame(START[0], START[1], start_body_yaw)
start_pitch = -math.atan(_dzdx)      # uphill ahead => nose up (negative pitch)
start_roll = math.atan(_dzdy)        # left side higher => positive roll
print(f"Start slope in body frame: pitch {math.degrees(start_pitch):+.1f} deg, "
      f"roll {math.degrees(start_roll):+.1f} deg")

# Laikago stands ~0.51-0.55 m above ground and topples if dropped from
# more than ~5 cm, so it is placed almost touching the ground and
# aligned with the local slope.
robot_start_position = [
    START[0],
    START[1],
    WAYPOINT_GROUND_Z[0] + 0.54
]


robot_id = p.loadURDF(
    "laikago/laikago_toes_zup.urdf",
    basePosition=robot_start_position,
    baseOrientation=p.getQuaternionFromEuler(
        [start_roll, start_pitch, start_body_yaw]
    ),
    useFixedBase=False
)

print(
    "Laikago loaded:",
    robot_id
)


# ============================================================
# FIND MINITAUR JOINTS AND LINKS
# ============================================================

joint_name_to_id = {}
link_name_to_id = {}

for joint_id in range(
    p.getNumJoints(robot_id)
):

    info = p.getJointInfo(
        robot_id,
        joint_id
    )

    # info[1] = joint name
    joint_name = info[1].decode(
        "utf-8"
    )

    # info[12] = child link name
    link_name = info[12].decode(
        "utf-8"
    )

    joint_name_to_id[
        joint_name
    ] = joint_id

    link_name_to_id[
        link_name
    ] = joint_id



LEG_ORDER = ["FR", "FL", "RR", "RL"]

# hip (abduction), upper leg, lower leg joint per leg
LEG_JOINT_IDS = {
    leg: (
        joint_name_to_id[f"{leg}_hip_motor_2_chassis_joint"],
        joint_name_to_id[f"{leg}_upper_leg_2_hip_motor_joint"],
        joint_name_to_id[f"{leg}_lower_leg_2_upper_leg_joint"],
    )
    for leg in LEG_ORDER
}

motor_ids = [j for leg in LEG_ORDER for j in LEG_JOINT_IDS[leg]]
knee_ids = [LEG_JOINT_IDS[leg][2] for leg in LEG_ORDER]

# Toe links (fixed joints "jtoeXX")
FOOT_LINK_IDS = [joint_name_to_id[f"jtoe{leg}"] for leg in LEG_ORDER]

print()
print("Laikago leg joints (hip, upper, lower):", LEG_JOINT_IDS)
print("Toe links:", FOOT_LINK_IDS)


# ============================================================
# LAIKAGO STANCE
# ============================================================
# Found empirically: upper = -0.67, lower = +1.25 stands at ~0.51 m.
# Decreasing (upper, lower) together LENGTHENS the leg.

UPPER0 = -0.67
LOWER0 = 1.25


def leg_angles(extension, swing=0.0, lift=0.0):
    """(hip, upper, lower) for one leg. extension>0 lengthens the leg."""
    return (
        0.0,
        UPPER0 - 0.15 * extension + swing,
        LOWER0 - 0.30 * extension + lift
    )


for leg in LEG_ORDER:

    for joint_id, angle in zip(
        LEG_JOINT_IDS[leg], leg_angles(STANCE_EXT)
    ):
        p.resetJointState(robot_id, joint_id, angle)


# ============================================================
# JOINT CONTROL
# ============================================================

def set_motor_targets(targets, knee_targets=None):
    """targets: {leg: (hip, upper, lower)}"""

    for leg in LEG_ORDER:

        for joint_id, angle in zip(LEG_JOINT_IDS[leg], targets[leg]):

            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                positionGain=MOTOR_KP,
                velocityGain=MOTOR_KD,
                force=MOTOR_FORCE
            )


# ============================================================
# TROT GAIT WITH BODY-LEVELLING FEEDBACK
# ============================================================
# Diagonal legs move together: (FR, RL) and (FL, RR).
# Levelling: legs on the low side of the body are lengthened
# (roll>0 => right side low; pitch>0 => nose down).
# turn>0 makes the LEFT legs' swing smaller and the RIGHT legs' larger,
# which (for this backward-walking gait) yaws the body clockwise.

def gait_targets(t, roll_deg, pitch_deg, turn=0.0, ramp=1.0, turn_method="original", heading_error=0.0):

    k = min(1.0, t / ramp)

    phase = 2.0 * math.pi * GAIT_FREQUENCY * t

    leg_phase = {
        "FR": phase, "RL": phase,
        "FL": phase + math.pi, "RR": phase + math.pi
    }

    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)

    targets = {}

    for leg in LEG_ORDER:

        left = leg in ("FL", "RL")
        front = leg in ("FR", "FL")

        extension = (
            STANCE_EXT
            + LEVEL_K_ROLL * roll * (-1.0 if left else 1.0)
            + LEVEL_K_PITCH * pitch * (1.0 if front else -1.0)
        )
        extension = max(0.0, min(1.2, extension))

        if turn_method == "differential":
            # For large heading errors, make the two sides step in OPPOSITE
            # directions, instead of stopping one side and doubling the other.
            # This is a diagnostic turning gait, NOT a guaranteed solution.
            # Smoothly blend walking into a pivot over 15..60 degrees.
            a = abs(math.degrees(heading_error))
            pivot_fraction = max(0.0, min(1.0, (a - 15.0) / 45.0))
            side = -1.0 if left else 1.0
            # At full pivot: left/right strides have opposite signs.
            # At zero pivot: original normal forward gait.
            stride = (1.0 - pivot_fraction) * (1.0 + side * turn) + pivot_fraction * (side * turn)
            swing = -GAIT_SWING * k * stride * math.sin(leg_phase[leg])
        else:
            side_scale = (1.0 - turn) if left else (1.0 + turn)
            swing = -GAIT_SWING * k * side_scale * math.sin(leg_phase[leg])
        lift = -GAIT_LIFT * k * max(0.0, math.cos(leg_phase[leg]))

        targets[leg] = leg_angles(extension, swing, lift)

    return targets


# ============================================================
# ORIENTATION
# ============================================================

def get_pitch_roll():

    position, orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    roll, pitch, yaw = (
        p.getEulerFromQuaternion(
            orientation
        )
    )

    return (
        math.degrees(pitch),
        math.degrees(roll),
        math.degrees(yaw)
    )


# ============================================================
# FALL DETECTION
# ============================================================

def robot_has_fallen():

    pitch, roll, _ = (
        get_pitch_roll()
    )

    if abs(pitch) > MAX_PITCH_DEG:

        return True

    if abs(roll) > MAX_ROLL_DEG:

        return True

    position, _ = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    # Height above the terrain (START z is NOT ground level and the
    # terrain climbs ~1.5 m toward the goal, so compare to local ground).
    if position[2] < ground_z_near(position[0], position[1]) + 0.25:

        return True

    return False


# ============================================================
# FOOT CONTACT INFORMATION
# ============================================================




def get_foot_contact_count():

    count = 0

    for link_id in FOOT_LINK_IDS:

        contacts = p.getContactPoints(
            bodyA=robot_id,
            linkIndexA=link_id
        )

        if len(contacts) > 0:

            count += 1

    return count


# ============================================================
# STUMBLE / SLIP DETECTION, LOGGING AND PHYSICAL PUSH
# ============================================================

stumble_log = []

last_hazard_event = None   # most recent event dict; used to explain a fall
_last_hazard_time = -999.0


def _log_hazard(sim_time, x, y, terrain_name, kind, reason, ratio=None, slope=None):

    global last_hazard_event

    event = {
        "time": sim_time,
        "x": x,
        "y": y,
        "terrain": terrain_name,
        "kind": kind,          # "stumble" or "slip"
        "reason": reason,
        "ratio": ratio,
        "slope_deg": slope,
    }

    stumble_log.append(event)
    last_hazard_event = event

    print(
        f"[{sim_time:6.1f}s] STUMBLE ({kind.upper()}) at "
        f"({x:.2f}, {y:.2f}) on {terrain_name.upper()} - {reason}"
    )


def apply_push(force_mag, torque_mag, yaw_deg):
    """Lateral push + roll/pitch torque on the robot base - this is what
    makes a stumble/slip visible (a wobble and a deviation off the path)
    without directly touching the gait or fall-detection logic."""

    angle = math.radians(yaw_deg) + random.uniform(-1.0, 1.0)

    force = [
        force_mag * math.cos(angle),
        force_mag * math.sin(angle),
        0.0,
    ]

    torque = [
        torque_mag * random.uniform(-1.0, 1.0),
        torque_mag * random.uniform(-1.0, 1.0),
        0.0,
    ]

    p.applyExternalForce(robot_id, -1, force, [0, 0, 0], p.WORLD_FRAME)
    p.applyExternalTorque(robot_id, -1, torque, p.WORLD_FRAME)


def evaluate_terrain_hazard(sim_time, x, y, terrain_name, water_distance, yaw_deg):
    """Called once per physics step. Decides whether the robot should
    stumble or slip right now, applies the physical push if so, and logs
    it. Debounced by HAZARD_EVENT_COOLDOWN_S so one hazard zone does not
    spam an event every 1/240 s tick."""

    global _last_hazard_time

    if sim_time - _last_hazard_time < HAZARD_EVENT_COOLDOWN_S:
        return

    if terrain_name == "mud":

        ratio = mud_ratio_at_world(x, y)

        if ratio is None:
            return

        if ratio < MUD_STUMBLE_RATIO:

            # Wet mud near water: stumble, but the levelling controller
            # should keep the robot up.
            if random.random() < MUD_STUMBLE_CHANCE_PER_SEC * TIME_STEP:

                apply_push(STUMBLE_PUSH_FORCE, STUMBLE_PUSH_TORQUE, yaw_deg)

                _log_hazard(
                    sim_time, x, y, terrain_name, "stumble",
                    reason=(
                        f"wet mud {water_distance:.2f} m from water "
                        f"(patch ratio {ratio:.2f}, near wet edge)"
                    ),
                    ratio=ratio,
                )

                _last_hazard_time = sim_time

        else:

            # Deeper / drier part of the patch: bigger disturbance, tuned
            # to often (not always) end in an actual fall.
            if random.random() < MUD_SLIP_CHANCE_PER_SEC * TIME_STEP:

                apply_push(SLIP_PUSH_FORCE, SLIP_PUSH_TORQUE, yaw_deg)

                _log_hazard(
                    sim_time, x, y, terrain_name, "slip",
                    reason=(
                        f"mud {water_distance:.2f} m from water "
                        f"(patch ratio {ratio:.2f}, deep in patch)"
                    ),
                    ratio=ratio,
                )

                _last_hazard_time = sim_time

    elif terrain_name == "rock":

        slope = slope_deg_at_world(x, y)

        if slope >= ROCK_STEEP_SLOPE_DEG:

            if random.random() < ROCK_STUMBLE_CHANCE_PER_SEC * TIME_STEP:

                apply_push(
                    STUMBLE_PUSH_FORCE * 0.7,
                    STUMBLE_PUSH_TORQUE * 0.7,
                    yaw_deg
                )

                _log_hazard(
                    sim_time, x, y, terrain_name, "stumble",
                    reason=f"steep rock slope ({slope:.1f} deg)",
                    slope=slope,
                )

                _last_hazard_time = sim_time


def explain_fall(sim_time, x, y, pitch_deg, roll_deg, terrain_name, water_distance):
    """Best-effort reason string for why the robot just fell, for the log."""

    if abs(pitch_deg) > MAX_PITCH_DEG or abs(roll_deg) > MAX_ROLL_DEG:

        # If a slip event just happened here, that is almost certainly why.
        if (
            last_hazard_event is not None
            and last_hazard_event["kind"] == "slip"
            and sim_time - last_hazard_event["time"] < HAZARD_EVENT_COOLDOWN_S + 1.0
        ):

            ratio = last_hazard_event.get("ratio")

            return (
                f"slipped in mud, {water_distance:.2f} m from water "
                f"(patch ratio {ratio:.2f}) -> lost balance "
                f"(pitch {pitch_deg:.1f} deg, roll {roll_deg:.1f} deg)"
            )

        if terrain_name == "mud":

            ratio = mud_ratio_at_world(x, y)
            ratio_txt = f"{ratio:.2f}" if ratio is not None else "n/a"

            return (
                f"encountered mud {water_distance:.2f} m from water "
                f"(patch ratio {ratio_txt}) -> lost balance "
                f"(pitch {pitch_deg:.1f} deg, roll {roll_deg:.1f} deg)"
            )

        slope = slope_deg_at_world(x, y)

        if slope >= ROCK_STEEP_SLOPE_DEG:

            return (
                f"high slope ({slope:.1f} deg) on {terrain_name} -> lost "
                f"balance (pitch {pitch_deg:.1f} deg, roll {roll_deg:.1f} deg)"
            )

        return (
            f"lost balance on {terrain_name} "
            f"(pitch {pitch_deg:.1f} deg, roll {roll_deg:.1f} deg)"
        )

    # Not a pitch/roll tip-over -> the body height dropped near/through
    # the terrain (foot sink, mesh gap, etc).
    return "body height dropped below the local terrain surface (lost footing)"


# ============================================================
# APPLY HIDDEN TERRAIN FRICTION
# ============================================================

last_friction = None
last_terrain = None


def update_terrain_friction(
    robot_x,
    robot_y
):

    global last_friction
    global last_terrain

    terrain_name, friction, distance = (
        current_terrain_properties(
            robot_x,
            robot_y
        )
    )


    # --------------------------------------------------------
    # Select physical terrain body
    # --------------------------------------------------------

    if terrain_name == "rock":

        body_id = rock_body

    elif terrain_name == "mud":

        body_id = mud_body

    elif terrain_name == "water":

        body_id = water_body

    else:

        body_id = rock_body


    # --------------------------------------------------------
    # Apply friction
    # --------------------------------------------------------

    if body_id is not None:

        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=friction
        )


    # --------------------------------------------------------
    # Print only meaningful changes
    # --------------------------------------------------------

    if (
        terrain_name != last_terrain
        or
        last_friction is None
        or
        abs(
            friction -
            last_friction
        ) > 0.03
    ):

        if terrain_name == "mud":

            print(
                f"Terrain: MUD | "
                f"distance to water: "
                f"{distance:.2f} m | "
                f"friction: "
                f"{friction:.3f}"
            )

        else:

            print(
                f"Terrain: "
                f"{terrain_name.upper()} | "
                f"friction: "
                f"{friction:.3f}"
            )

        last_terrain = terrain_name

        last_friction = friction


    return (
        terrain_name,
        friction,
        distance
    )


# ============================================================
# INITIAL CAMERA
# ============================================================

if not HEADLESS:

    p.resetDebugVisualizerCamera(
        cameraDistance=CAMERA_DISTANCE,
        cameraYaw=0,
        cameraPitch=CAMERA_PITCH,
        cameraTargetPosition=START
    )


# ============================================================
# LET ROBOT SETTLE
# ============================================================

print()
print("Allowing Laikago to settle...")

pos, orn = p.getBasePositionAndOrientation(robot_id)
r, pitch, yaw = p.getEulerFromQuaternion(orn)

print()
print("BEFORE SETTLING")
print(f"Position: {pos}")
print(f"Roll:  {math.degrees(r):.2f}°")
print(f"Pitch: {math.degrees(pitch):.2f}°")
print(f"Yaw:   {math.degrees(yaw):.2f}°")

for _ in range(360):

    _pitch, _roll, _ = get_pitch_roll()

    set_motor_targets(gait_targets(0.0, _roll, _pitch))

    p.stepSimulation()

pos, orn = p.getBasePositionAndOrientation(robot_id)
r, pitch, yaw = p.getEulerFromQuaternion(orn)

print()
print("AFTER SETTLING")
print(f"Position: {pos}")
print(f"Roll:  {math.degrees(r):.2f}°")
print(f"Pitch: {math.degrees(pitch):.2f}°")
print(f"Yaw:   {math.degrees(yaw):.2f}°")

# ============================================================
# EXECUTE CMDP PATH
# ============================================================

print()
print("========================================")
print("EXECUTING CMDP PATH")
print("========================================")


# DEBUG: cumulative distance along the original polyline (XY metres).
path_s = [0.0]
for _i in range(1, len(waypoints)):
    path_s.append(path_s[-1] + math.hypot(
        waypoints[_i][0] - waypoints[_i-1][0],
        waypoints[_i][1] - waypoints[_i-1][1]))
path_length = path_s[-1]
tracking_progress = 0.0
tracking_records = []
last_debug_target_angle = None
last_debug_waypoint = 1
max_path_error = 0.0
sum_path_error = 0.0
n_path_error = 0


def project_on_path(x, y, min_s=0.0, max_s=None):
    """Closest XY point on polyline within allowed arc-length interval."""
    if max_s is None:
        max_s = path_length
    best = None
    for j in range(len(waypoints)-1):
        s0, s1 = path_s[j], path_s[j+1]
        if s1 < min_s or s0 > max_s or s1 <= s0:
            continue
        ax, ay = waypoints[j][:2]
        bx, by = waypoints[j+1][:2]
        vx, vy = bx-ax, by-ay
        t = max(0.0, min(1.0, ((x-ax)*vx + (y-ay)*vy) / (vx*vx+vy*vy)))
        t = max(t, max(0.0, (min_s-s0)/(s1-s0)))
        t = min(t, min(1.0, (max_s-s0)/(s1-s0)))
        qx, qy = ax+t*vx, ay+t*vy
        d = math.hypot(x-qx, y-qy)
        if best is None or d < best[0]:
            best = (d, s0+t*(s1-s0), qx, qy, j)
    return best


def point_at_s(s):
    s = max(0.0, min(path_length, s))
    j = max(0, min(len(waypoints)-2, bisect.bisect_right(path_s, s)-1))
    length = path_s[j+1]-path_s[j]
    t = (s-path_s[j])/length if length > 0 else 0.0
    return (waypoints[j][0] + t*(waypoints[j+1][0]-waypoints[j][0]),
            waypoints[j][1] + t*(waypoints[j+1][1]-waypoints[j][1]))

print(f"TRACKING DIAGNOSTIC: mode={args.tracking_mode}, steering={args.steering_test}, path_length={path_length:.2f} m")

current_waypoint = 1

simulation_time = 0.0

gait_time = 0.0

smoothed_roll_deg = 0.0
smoothed_pitch_deg = 0.0

trajectory = []

fallen = False

fall_time = None

fall_reason = None

max_pitch_seen = 0.0

max_roll_seen = 0.0

touched_wall = False

wall_contact_time = None

step_counter = 0


terrain_time = {

    "rock": 0.0,
    "mud": 0.0,
    "water": 0.0

}


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

while (
    current_waypoint <
    len(waypoints)
):


    # --------------------------------------------------------
    # ROBOT STATE
    # --------------------------------------------------------

    position, orientation = (
        p.getBasePositionAndOrientation(
            robot_id
        )
    )

    robot_x = position[0]

    robot_y = position[1]

    robot_z = position[2]


    pitch_deg, roll_deg, yaw_deg = (
        get_pitch_roll()
    )


    max_pitch_seen = max(
        max_pitch_seen,
        abs(pitch_deg)
    )

    max_roll_seen = max(
        max_roll_seen,
        abs(roll_deg)
    )

    # ----------------------------------------------------------
    # SMOOTH ROLL/PITCH FOR THE LEVELING CONTROLLER
    # ----------------------------------------------------------
    # Feeding the raw, noisy per-step roll/pitch straight into a
    # high-gain P controller running at the gait frequency is what
    # was causing the oscillation-and-tip-over (robot rolled to
    # -45 deg within ~20 gait cycles on flat ground). Low-pass
    # filtering + clamping removes that resonance.

    smoothed_roll_deg = (
        ROLL_PITCH_FILTER_ALPHA * roll_deg
        + (1.0 - ROLL_PITCH_FILTER_ALPHA) * smoothed_roll_deg
    )

    smoothed_pitch_deg = (
        ROLL_PITCH_FILTER_ALPHA * pitch_deg
        + (1.0 - ROLL_PITCH_FILTER_ALPHA) * smoothed_pitch_deg
    )

    level_roll_deg = max(-LEVEL_CLAMP_DEG, min(LEVEL_CLAMP_DEG, smoothed_roll_deg))
    level_pitch_deg = max(-LEVEL_CLAMP_DEG, min(LEVEL_CLAMP_DEG, smoothed_pitch_deg))


    # --------------------------------------------------------
    # TERRAIN / FRICTION
    # --------------------------------------------------------

    terrain_name, friction, water_distance = (
        update_terrain_friction(
            robot_x,
            robot_y
        )
    )


    if terrain_name in terrain_time:

        terrain_time[
            terrain_name
        ] += TIME_STEP


    # --------------------------------------------------------
    # STUMBLE / SLIP HAZARDS (stay up but wobble/deviate, or
    # possibly fall - see STUMBLE / SLIP MODEL settings above)
    # --------------------------------------------------------

    evaluate_terrain_hazard(
        simulation_time,
        robot_x,
        robot_y,
        terrain_name,
        water_distance,
        yaw_deg
    )


    # --------------------------------------------------------
    # FALL DETECTION
    # --------------------------------------------------------

    if robot_has_fallen():

        if not fallen:

            fallen = True

            fall_time = (
                simulation_time
            )

            fall_reason = explain_fall(
                simulation_time,
                robot_x,
                robot_y,
                pitch_deg,
                roll_deg,
                terrain_name,
                water_distance
            )

            print()
            print("========================================")
            print("ROBOT FALL DETECTED")
            print("========================================")

            print(
                f"Time:  "
                f"{simulation_time:.3f} s"
            )

            print(
                f"Pitch: "
                f"{pitch_deg:.2f}°"
            )

            print(
                f"Roll:  "
                f"{roll_deg:.2f}°"
            )

            print(
                f"Reason: {fall_reason}"
            )

            print(
                "Physical execution stopped."
            )

        break


    # Full-route distance is a DIAGNOSTIC ONLY (not used for control).
    global_nearest = project_on_path(robot_x, robot_y)
    path_error = global_nearest[0]
    max_path_error = max(max_path_error, path_error)
    sum_path_error += path_error
    n_path_error += 1
    prev_waypoint = current_waypoint

    if args.tracking_mode == "original":
        # --------------------------------------------------------
        # TARGET WAYPOINT
        # --------------------------------------------------------

        target_x, target_y, target_z = (
            waypoints[
                current_waypoint
            ]
        )


        dx = (
            target_x -
            robot_x
        )

        dy = (
            target_y -
            robot_y
        )


        horizontal_distance = math.sqrt(
            dx * dx +
            dy * dy
        )


        # --------------------------------------------------------
        # WAYPOINT REACHED
        # --------------------------------------------------------

        if (
            horizontal_distance <=
            WAYPOINT_TOLERANCE
        ):

            current_waypoint += 1

            print(
                f"Reached waypoint "
                f"{current_waypoint - 1} / "
                f"{len(waypoints) - 1}"
            )

            continue


        # Skip waypoints already passed (a later waypoint is closer)
        while current_waypoint + 1 < len(waypoints):

            nx, ny, _ = waypoints[current_waypoint + 1]

            if math.hypot(nx - robot_x, ny - robot_y) < horizontal_distance:

                current_waypoint += 1
                target_x, target_y, target_z = waypoints[current_waypoint]
                horizontal_distance = math.hypot(
                    target_x - robot_x, target_y - robot_y
                )

            else:

                break


        # --------------------------------------------------------
        # DESIRED HEADING (pure-pursuit style lookahead point)
        # --------------------------------------------------------

        look = current_waypoint

        while (
            look < len(waypoints) - 1
            and math.hypot(
                waypoints[look][0] - robot_x,
                waypoints[look][1] - robot_y
            ) < LOOKAHEAD_DISTANCE
        ):
            look += 1

        look_x, look_y, _ = waypoints[look]

        desired_yaw = math.atan2(
            look_y - robot_y,
            look_x - robot_x
        )

        target_s = path_s[min(look, len(path_s)-1)]
        local_nearest = global_nearest
    else:
        # Search only a nearby section ahead of the last route progress.
        # This prevents a robot far from the path from jumping to a distant
        # (possibly geometrically nearby) portion of a winding route.
        local_nearest = project_on_path(
            robot_x, robot_y,
            max(0.0, tracking_progress-1.0),
            min(path_length, tracking_progress+3.0))
        if local_nearest is None:
            local_nearest = global_nearest
        # Progress never goes backward. The route itself is NEVER modified.
        tracking_progress = max(tracking_progress, local_nearest[1])
        target_s = min(path_length, tracking_progress+LOOKAHEAD_DISTANCE)
        look_x, look_y = point_at_s(target_s)
        desired_yaw = math.atan2(look_y-robot_y, look_x-robot_x)
        current_waypoint = min(len(waypoints)-1,
                               max(1, bisect.bisect_right(path_s, tracking_progress)))
        if tracking_progress >= path_length-WAYPOINT_TOLERANCE and \
                math.hypot(robot_x-waypoints[-1][0], robot_y-waypoints[-1][1]) <= WAYPOINT_TOLERANCE:
            current_waypoint = len(waypoints)
            print("[TRACK] Final waypoint reached within tolerance")
            continue

    # Original heading computation follows, including BODY_YAW_OFFSET.
    # --------------------------------------------------------
    # CURRENT YAW (radians)
    # --------------------------------------------------------

    _, _, current_yaw = p.getEulerFromQuaternion(orientation)

    yaw_error = (desired_yaw + BODY_YAW_OFFSET) - current_yaw

    while yaw_error > math.pi:
        yaw_error -= 2.0 * math.pi

    while yaw_error < -math.pi:
        yaw_error += 2.0 * math.pi

    yaw_rate = p.getBaseVelocity(robot_id)[1][2]


    # --------------------------------------------------------
    # STEERING  (turn > 0 = turn left)
    # --------------------------------------------------------

    steering = max(
        -MAX_TURN,
        min(
            MAX_TURN,
            -(K_YAW * yaw_error - K_YAW_RATE * yaw_rate)
        )
    )


    if args.steering_test == "positive":
        steering = 0.5
    elif args.steering_test == "negative":
        steering = -0.5
    elif args.steering_test == "zero":
        steering = 0.0

    # Diagnostics once per simulated second. No physics parameters changed.
    if step_counter % 240 == 0:
        actual_travel_yaw = (current_yaw - BODY_YAW_OFFSET + math.pi) % (2*math.pi) - math.pi
        heading_jump = 0.0 if last_debug_target_angle is None else math.degrees(
            (desired_yaw-last_debug_target_angle+math.pi) % (2*math.pi)-math.pi)
        print(f"[TRACK] t={simulation_time:5.1f}s mode={args.tracking_mode} "
              f"wp={current_waypoint:3d} look=({look_x:+.2f},{look_y:+.2f}) "
              f"route_error={path_error:.2f}m local_error={local_nearest[0]:.2f}m "
              f"heading_err={math.degrees(yaw_error):+.1f}deg "
              f"target_change={heading_jump:+.1f}deg "
              f"turn={steering:+.2f} method={args.turn_method} body_yaw={math.degrees(current_yaw):+.1f}deg "
              f"travel_yaw={math.degrees(actual_travel_yaw):+.1f}deg "
              f"yaw_rate={math.degrees(yaw_rate):+.1f}deg/s")
        if abs(heading_jump) > 35:
            print("[WARNING] Sudden desired-heading change: inspect waypoint/route geometry")
        if current_waypoint-prev_waypoint > 1:
            print(f"[WARNING] Waypoint jump: {prev_waypoint} -> {current_waypoint}")
        last_debug_target_angle = desired_yaw
        tracking_records.append({
            "time_s": round(simulation_time, 3), "mode": args.tracking_mode,
            "robot_x": robot_x, "robot_y": robot_y,
            "waypoint": current_waypoint, "look_x": look_x, "look_y": look_y,
            "target_s_m": target_s, "route_error_m": path_error,
            "local_route_error_m": local_nearest[0],
            "desired_travel_yaw_deg": math.degrees(desired_yaw),
            "actual_travel_yaw_deg": math.degrees(actual_travel_yaw),
            "heading_error_deg": math.degrees(yaw_error),
            "target_change_deg": heading_jump, "steering": steering,
            "yaw_rate_deg_s": math.degrees(yaw_rate),
            "pitch_deg": pitch_deg, "roll_deg": roll_deg,
        })

    # --------------------------------------------------------
    # GAIT
    # --------------------------------------------------------

    motor_targets = gait_targets(
        gait_time,
        level_roll_deg,
        level_pitch_deg,
        steering * args.turn_sign if args.turn_method == "differential" else steering,
        turn_method=args.turn_method,
        heading_error=yaw_error
    )

    set_motor_targets(motor_targets)

    gait_time += TIME_STEP

    if int(simulation_time / 5.0) != int((simulation_time + TIME_STEP) / 5.0):

        goal_d = math.hypot(GOAL[0] - robot_x, GOAL[1] - robot_y)

        print(
            f"t={simulation_time:6.1f}s  wp {current_waypoint}/"
            f"{len(waypoints) - 1}  pos=({robot_x:.2f}, {robot_y:.2f})  "
            f"goal dist={goal_d:.2f} m  yaw err={math.degrees(yaw_error):+.0f} deg  "
            f"pitch={pitch_deg:+.0f} roll={roll_deg:+.0f}"
        )


    # --------------------------------------------------------
    # PHYSICS
    # --------------------------------------------------------

    p.stepSimulation()

    simulation_time += TIME_STEP

    step_counter += 1

    # Cheap wall-contact check every 0.05 s (physical wall collision proof).
    if (
        wall_body is not None
        and not touched_wall
        and step_counter % 12 == 0
        and len(p.getContactPoints(bodyA=robot_id, bodyB=wall_body)) > 0
    ):

        touched_wall = True
        wall_contact_time = simulation_time
        print(f"[{simulation_time:6.1f}s] Robot touched a WALL.")


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    if not HEADLESS:

        p.resetDebugVisualizerCamera(
            cameraDistance=CAMERA_DISTANCE,
            cameraYaw=0,
            cameraPitch=CAMERA_PITCH,
            cameraTargetPosition=[
                robot_x,
                robot_y,
                robot_z
            ]
        )


    # --------------------------------------------------------
    # ACTUAL VELOCITY
    # --------------------------------------------------------

    linear_velocity, angular_velocity = (
        p.getBaseVelocity(
            robot_id
        )
    )


    speed = math.sqrt(
        linear_velocity[0] ** 2 +
        linear_velocity[1] ** 2
    )


    # --------------------------------------------------------
    # FOOT CONTACTS
    # --------------------------------------------------------

    contact_count = (
        get_foot_contact_count()
    )


    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    trajectory.append(
        (
            simulation_time,
            current_waypoint,
            robot_x,
            robot_y,
            robot_z,
            pitch_deg,
            roll_deg,
            speed,
            contact_count,
            terrain_name,
            friction,
            (
                water_distance
                if water_distance is not None
                else -1.0
            )
        )
    )


    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    if (
        simulation_time >=
        MAX_SIMULATION_TIME
    ):

        print()
        print(
            "Maximum simulation time reached."
        )

        break


# Save separate diagnostic CSV even when a fall ends the run.
tracking_csv = OUTPUT_DIR / "path_tracking_diagnostics.csv"
if tracking_records:
    with tracking_csv.open("w", newline="", encoding="utf-8") as _f:
        _writer = csv.DictWriter(_f, fieldnames=list(tracking_records[0].keys()))
        _writer.writeheader()
        _writer.writerows(tracking_records)
print(f"[TRACK SUMMARY] mean route error={sum_path_error/max(1,n_path_error):.3f}m "
      f"max route error={max_path_error:.3f}m")
print(f"[TRACK SUMMARY] diagnostics saved to: {tracking_csv}")

# ============================================================
# STOP MOTORS
# ============================================================

for motor_id in motor_ids:

    p.setJointMotorControl2(
        bodyIndex=robot_id,
        jointIndex=motor_id,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=MOTOR_FORCE
    )


for knee_id in knee_ids:

    p.setJointMotorControl2(
        bodyIndex=robot_id,
        jointIndex=knee_id,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0,
        force=MOTOR_FORCE
    )


# ============================================================
# FINAL POSITION
# ============================================================

final_position, final_orientation = (
    p.getBasePositionAndOrientation(
        robot_id
    )
)


final_x = final_position[0]

final_y = final_position[1]

final_z = final_position[2]


distance_from_goal = math.sqrt(
    (
        final_x -
        GOAL[0]
    ) ** 2
    +
    (
        final_y -
        GOAL[1]
    ) ** 2
)


reached_goal = (
    distance_from_goal <=
    WAYPOINT_TOLERANCE
)


# ============================================================
# SAVE ACTUAL TRAJECTORY
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "time",
        "cmdp_waypoint_index",
        "actual_x",
        "actual_y",
        "actual_z",
        "pitch_deg",
        "roll_deg",
        "horizontal_speed",
        "foot_contacts",
        "terrain",
        "friction",
        "distance_to_water"
    ])

    writer.writerows(
        trajectory
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("========================================")
print("PYBULLET RESULT")
print("========================================")

print(
    f"Simulation time: "
    f"{simulation_time:.3f} s"
)

print(
    f"Final position: "
    f"({final_x:.3f}, "
    f"{final_y:.3f}, "
    f"{final_z:.3f})"
)

print(
    f"Goal position: "
    f"({GOAL[0]:.3f}, "
    f"{GOAL[1]:.3f}, "
    f"{GOAL[2]:.3f})"
)

print(
    f"Distance from goal: "
    f"{distance_from_goal:.3f} m"
)

print(
    f"Reached goal: "
    f"{'YES' if reached_goal else 'NO'}"
)

print(
    f"Robot fell: "
    f"{'YES' if fallen else 'NO'}"
)

if fall_time is not None:

    print(
        f"Fall time: "
        f"{fall_time:.3f} s"
    )

    print(
        f"Fall reason: {fall_reason}"
    )


print()
print("Stumble / slip events during run:")

if len(stumble_log) == 0:

    print("  (none)")

else:

    for _event in stumble_log:

        print(
            f"  [{_event['time']:6.1f}s] "
            f"{_event['kind'].upper():7s} on "
            f"{_event['terrain'].upper():5s} at "
            f"({_event['x']:.2f}, {_event['y']:.2f}) - "
            f"{_event['reason']}"
        )

    print(f"  Total: {len(stumble_log)} event(s)")


with open(STUMBLE_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "time", "x", "y", "terrain", "kind", "reason", "ratio", "slope_deg"
    ])

    for _event in stumble_log:

        writer.writerow([
            _event["time"], _event["x"], _event["y"], _event["terrain"],
            _event["kind"], _event["reason"], _event["ratio"], _event["slope_deg"]
        ])

print()
print(f"Stumble/slip log saved to: {STUMBLE_CSV}")


print()
print(
    f"Maximum |pitch|: "
    f"{max_pitch_seen:.2f}°"
)

print(
    f"Maximum |roll|: "
    f"{max_roll_seen:.2f}°"
)


print()
print("Terrain exposure:")

for terrain_name in [
    "rock",
    "mud",
    "water"
]:

    print(
        f"  {terrain_name}: "
        f"{terrain_time[terrain_name]:.3f} s"
    )


print()
print(
    "Actual trajectory saved to:"
)

print(
    OUTPUT_CSV
)


# ============================================================
# SAVE RESULT JSON  (this is what the runners look for: result_*.json)
# ============================================================

timed_out = bool(
    (not fallen)
    and (not reached_goal)
    and simulation_time >= MAX_SIMULATION_TIME
)

result = {
    "terrain": TERRAIN_DIR.name,
    "scenario_id": args.scenario_id,
    "risk_budget": float(args.risk_budget),
    "seed": int(args.seed),
    "path_file": str(CMDP_PATH),
    "reached_goal": bool(reached_goal),
    "fell": bool(fallen),
    "fall_time": None if fall_time is None else float(fall_time),
    "fall_reason": fall_reason,
    "timed_out": timed_out,
    "simulation_time": float(simulation_time),
    "final_position": [float(final_x), float(final_y), float(final_z)],
    "goal_position": [float(GOAL[0]), float(GOAL[1]), float(GOAL[2])],
    "distance_from_goal": float(distance_from_goal),
    "waypoints_total": len(waypoints) - 1,
    "waypoints_reached": int(max(0, current_waypoint - 1)),
    "max_abs_pitch_deg": float(max_pitch_seen),
    "max_abs_roll_deg": float(max_roll_seen),
    "terrain_time_s": {k: float(v) for k, v in terrain_time.items()},
    "stumble_events": int(sum(1 for e in stumble_log if e["kind"] == "stumble")),
    "slip_events": int(sum(1 for e in stumble_log if e["kind"] == "slip")),
    "touched_wall": bool(touched_wall),
    "wall_contact_time": wall_contact_time,
    "obj_up_axis_in_file": obj_info["up_axis_in_file"],
    "trajectory_csv": str(OUTPUT_CSV),
}

with open(RESULT_JSON, "w", encoding="utf-8") as f:

    json.dump(result, f, indent=2)

print()
print(f"Result saved to: {RESULT_JSON}")


# ============================================================
# KEEP WINDOW OPEN (GUI only) / EXIT (headless)
# ============================================================

if HEADLESS:

    p.disconnect()

    # Exit code 0 even if the robot fell: a fall is a valid result.
    sys.exit(0)


print()
print("========================================")
print("PYBULLET CONTROLS")
print("========================================")

print(
    "Mouse drag: rotate camera"
)

print(
    "Scroll:     zoom"
)

print(
    "Right drag: move camera"
)

print()

print(
    "Close the PyBullet window when finished."
)


while p.isConnected():

    time.sleep(0.1)

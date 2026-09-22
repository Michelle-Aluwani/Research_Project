# ============================================================
# RRT* AND RISK-WEIGHTED RRT*
# ============================================================
#
# Standard RRT*:
#     Minimize accumulated Euclidean path length.
#
# Risk-weighted RRT*:
#     Minimize:
#
#       J(p) = sum [ alpha*d(a)
#                  + (1-alpha)*beta*h(a) ]
#
#     where:
#       d(a) = Euclidean edge length
#       h(a) = probability of at least one hazard along edge
#       alpha = distance/risk trade-off
#       beta = risk normalization coefficient
#
# ============================================================

import numpy as np
import math
import random
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import sys

# Allow importing the project's existing CMDP terrain/risk implementation.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cmdp_refactored import load_terrain_grid, prepare_risk_model


# ============================================================
# RRT* PARAMETERS
# ============================================================

RRT_MAX_ITERATIONS = 5000

# Maximum distance an RRT* extension can travel per iteration
RRT_STEP_SIZE = 2.0

# Radius used for neighbourhood search
RRT_NEIGHBOR_RADIUS = 4.0

# Number of samples taken along an edge when checking validity/risk
EDGE_SAMPLE_RESOLUTION = 0.25

# Probability of directly attempting to sample the goal
GOAL_SAMPLE_RATE = 0.10

# Distance within which a node is considered to have reached goal
GOAL_TOLERANCE = 1.0

# Random seed
RRT_RANDOM_SEED = 42


# ============================================================
# RISK-WEIGHTED RRT* PARAMETERS
# ============================================================

# alpha from your proposal:
#
# alpha = 1.0 -> distance only
# alpha = 0.0 -> risk only
#
# A value in the middle balances the two.
RISK_RRT_ALPHA = 0.5

# beta normalises risk relative to distance.
#
# This can be calibrated from the terrain/edge statistics.
RISK_RRT_BETA = 1.0  # overwritten by the terrain/scenario configuration


# ============================================================
# TERRAIN SETUP
# ============================================================

# This code expects the same variables used by your CMDP:
#
# weighted_risk_grid
# height_grid
#
# weighted_risk_grid:
#     risk value at each grid cell
#
# height_grid:
#     finite value = traversable terrain
#     NaN / inf    = invalid/no terrain
#
# Grid indexing:
#     [x, y]
#
# Your grid resolution:
GRID_RESOLUTION = 0.5


# ============================================================
# VALID TERRAIN MASK
# ============================================================

valid_terrain_mask = np.isfinite(height_grid)

GRID_WIDTH, GRID_HEIGHT = weighted_risk_grid.shape

# Actual terrain-world coordinate information.  The original RRT* algorithm
# remains in grid-index space; only the final path export uses these values.
TERRAIN_BOUNDS = None
TERRAIN_HEIGHT_GRID = None


# ============================================================
# NODE
# ============================================================

@dataclass
class RRTNode:

    x: float
    y: float

    parent: object = None

    # Cost used by the particular planner
    cost: float = 0.0

    # Physical distance travelled from start
    distance_cost: float = 0.0

    # Accumulated hazard probability
    risk_cost: float = 0.0


# ============================================================
# GRID / WORLD CONVERSION
# ============================================================

def grid_to_world(x, y):
    """
    Convert grid indices to the project's terrain-world cell-centre
    coordinates.

    The RRT* itself still operates in grid-index space.
    """
    if TERRAIN_BOUNDS is None:
        raise RuntimeError("Terrain bounds have not been configured.")

    return (
        TERRAIN_BOUNDS["min_x"] + (x + 0.5) * GRID_RESOLUTION,
        TERRAIN_BOUNDS["min_y"] + (y + 0.5) * GRID_RESOLUTION,
    )


def world_to_grid(x, y):
    """
    Convert physical coordinates to the project's grid-index coordinates.
    """
    if TERRAIN_BOUNDS is None:
        raise RuntimeError("Terrain bounds have not been configured.")

    return (
        (x - TERRAIN_BOUNDS["min_x"]) / GRID_RESOLUTION - 0.5,
        (y - TERRAIN_BOUNDS["min_y"]) / GRID_RESOLUTION - 0.5,
    )


# ============================================================
# VALID POINT
# ============================================================

def is_valid_point(x, y):

    gx = int(round(x))
    gy = int(round(y))

    if gx < 0 or gx >= GRID_WIDTH:
        return False

    if gy < 0 or gy >= GRID_HEIGHT:
        return False

    if not valid_terrain_mask[gx, gy]:
        return False

    return True


# ============================================================
# POINT RISK
# ============================================================

def point_risk(x, y):

    gx = int(round(x))
    gy = int(round(y))

    if gx < 0 or gx >= GRID_WIDTH:
        return 1.0

    if gy < 0 or gy >= GRID_HEIGHT:
        return 1.0

    risk = weighted_risk_grid[gx, gy]

    if not np.isfinite(risk):
        return 1.0

    return float(np.clip(risk, 0.0, 1.0))


# ============================================================
# EDGE SAMPLING
# ============================================================

def sample_edge(x1, y1, x2, y2):

    distance_grid = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    distance_m = distance_grid * GRID_RESOLUTION

    number_of_samples = max(
        2,
        int(math.ceil(
            distance_m / EDGE_SAMPLE_RESOLUTION
        ))
    )

    xs = np.linspace(
        x1,
        x2,
        number_of_samples
    )

    ys = np.linspace(
        y1,
        y2,
        number_of_samples
    )

    return xs, ys


# ============================================================
# EDGE VALIDITY
# ============================================================

def edge_is_valid(x1, y1, x2, y2):

    xs, ys = sample_edge(
        x1,
        y1,
        x2,
        y2
    )

    for x, y in zip(xs, ys):

        if not is_valid_point(x, y):
            return False

    return True


# ============================================================
# EDGE DISTANCE
# ============================================================

def edge_distance(x1, y1, x2, y2):

    distance_grid = math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

    return distance_grid * GRID_RESOLUTION


# ============================================================
# EDGE HAZARD PROBABILITY
# ============================================================

def edge_hazard_probability(x1, y1, x2, y2):

    """
    Calculate h(a), the hazard probability along an edge.

    For sampled terrain risks r_i:

        h(a) = 1 - product(1-r_i)

    This represents the probability of at least one
    hazardous event occurring along the edge, assuming
    the sampled hazard events are conditionally independent.

    The result is always in [0, 1].
    """

    xs, ys = sample_edge(
        x1,
        y1,
        x2,
        y2
    )

    survival_probability = 1.0

    for x, y in zip(xs, ys):

        risk = point_risk(x, y)

        survival_probability *= (1.0 - risk)

    hazard_probability = 1.0 - survival_probability

    return float(
        np.clip(
            hazard_probability,
            0.0,
            1.0
        )
    )


# ============================================================
# STANDARD RRT* EDGE COST
# ============================================================

def standard_rrt_edge_cost(
    x1,
    y1,
    x2,
    y2
):

    """
    Standard RRT* cost:

        cost(a) = d(a)

    where d(a) is physical Euclidean distance.
    """

    return edge_distance(
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# RISK-WEIGHTED RRT* EDGE COST
# ============================================================

def risk_weighted_rrt_edge_cost(
    x1,
    y1,
    x2,
    y2,
    alpha=RISK_RRT_ALPHA,
    beta=RISK_RRT_BETA
):

    """
    Proposal formulation:

        Cost(x1,x2)
          = alpha*d(x1,x2)
          + (1-alpha)*beta*h(x1,x2)

    """

    d = edge_distance(
        x1,
        y1,
        x2,
        y2
    )

    h = edge_hazard_probability(
        x1,
        y1,
        x2,
        y2
    )

    cost = (
        alpha * d
        +
        (1.0 - alpha)
        * beta
        * h
    )

    return float(cost)


# ============================================================
# STEER
# ============================================================

def steer(
    from_node,
    target_x,
    target_y,
    max_distance
):

    dx = target_x - from_node.x
    dy = target_y - from_node.y

    distance = math.sqrt(
        dx * dx +
        dy * dy
    )

    if distance <= max_distance:

        new_x = target_x
        new_y = target_y

    else:

        scale = max_distance / distance

        new_x = (
            from_node.x +
            dx * scale
        )

        new_y = (
            from_node.y +
            dy * scale
        )

    return new_x, new_y


# ============================================================
# NEAREST NODE
# ============================================================

def nearest_node(
    nodes,
    x,
    y
):

    best_node = None
    best_distance = float("inf")

    for node in nodes:

        distance = (
            (node.x - x) ** 2 +
            (node.y - y) ** 2
        )

        if distance < best_distance:

            best_distance = distance
            best_node = node

    return best_node


# ============================================================
# NEAR NODES
# ============================================================

def near_nodes(
    nodes,
    x,
    y,
    radius
):

    radius_squared = radius ** 2

    nearby = []

    for node in nodes:

        distance_squared = (
            (node.x - x) ** 2 +
            (node.y - y) ** 2
        )

        if distance_squared <= radius_squared:

            nearby.append(node)

    return nearby


# ============================================================
# RECONSTRUCT PATH
# ============================================================

def reconstruct_path(node):

    path = []

    current = node

    while current is not None:

        path.append(
            (current.x, current.y)
        )

        current = current.parent

    path.reverse()

    return path


# ============================================================
# PATH METRICS
# ============================================================

def calculate_path_metrics(path):

    if path is None or len(path) < 2:

        return {
            "path_length": np.inf,
            "path_risk": np.inf,
            "number_of_edges": 0
        }

    total_distance = 0.0

    total_hazard = 0.0

    survival_probability = 1.0

    for i in range(len(path) - 1):

        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        d = edge_distance(
            x1,
            y1,
            x2,
            y2
        )

        h = edge_hazard_probability(
            x1,
            y1,
            x2,
            y2
        )

        total_distance += d

        # Accumulated probability of at least one
        # hazard occurring anywhere along the path.
        survival_probability *= (1.0 - h)

    total_hazard = (
        1.0 - survival_probability
    )

    return {
        "path_length": total_distance,
        "path_risk": total_hazard,
        "number_of_edges": len(path) - 1
    }


# ============================================================
# GENERIC RRT* SOLVER
# ============================================================

def run_rrt_star(
    start,
    goal,
    weighted=False,
    alpha=RISK_RRT_ALPHA,
    beta=RISK_RRT_BETA,
    max_iterations=RRT_MAX_ITERATIONS,
    step_size=RRT_STEP_SIZE,
    neighbor_radius=RRT_NEIGHBOR_RADIUS,
    goal_sample_rate=GOAL_SAMPLE_RATE,
    goal_tolerance=GOAL_TOLERANCE,
    random_seed=RRT_RANDOM_SEED
):

    random.seed(random_seed)
    np.random.seed(random_seed)

    start_x, start_y = start
    goal_x, goal_y = goal

    # --------------------------------------------------------
    # Check start and goal
    # --------------------------------------------------------

    if not is_valid_point(
        start_x,
        start_y
    ):

        raise ValueError(
            "RRT*: start position is not valid."
        )

    if not is_valid_point(
        goal_x,
        goal_y
    ):

        raise ValueError(
            "RRT*: goal position is not valid."
        )

    # --------------------------------------------------------
    # Create root
    # --------------------------------------------------------

    root = RRTNode(
        x=float(start_x),
        y=float(start_y),
        parent=None,
        cost=0.0,
        distance_cost=0.0,
        risk_cost=0.0
    )

    nodes = [root]

    best_goal_node = None
    best_goal_cost = float("inf")

    # --------------------------------------------------------
    # Main RRT* loop
    # --------------------------------------------------------

    for iteration in range(max_iterations):

        # ----------------------------------------------------
        # Random sample
        # ----------------------------------------------------

        if random.random() < goal_sample_rate:

            sample_x = goal_x
            sample_y = goal_y

        else:

            sample_x = random.uniform(
                0,
                GRID_WIDTH - 1
            )

            sample_y = random.uniform(
                0,
                GRID_HEIGHT - 1
            )

        # ----------------------------------------------------
        # Find nearest node
        # ----------------------------------------------------

        nearest = nearest_node(
            nodes,
            sample_x,
            sample_y
        )

        # ----------------------------------------------------
        # Steer
        # ----------------------------------------------------

        new_x, new_y = steer(
            nearest,
            sample_x,
            sample_y,
            step_size
        )

        # ----------------------------------------------------
        # Check new point
        # ----------------------------------------------------

        if not is_valid_point(
            new_x,
            new_y
        ):

            continue

        # ----------------------------------------------------
        # Check edge
        # ----------------------------------------------------

        if not edge_is_valid(
            nearest.x,
            nearest.y,
            new_x,
            new_y
        ):

            continue

        # ----------------------------------------------------
        # Find neighbouring nodes
        # ----------------------------------------------------

        neighbours = near_nodes(
            nodes,
            new_x,
            new_y,
            neighbor_radius
        )

        # ----------------------------------------------------
        # Choose best parent
        # ----------------------------------------------------

        best_parent = nearest

        if weighted:

            best_edge_cost = (
                risk_weighted_rrt_edge_cost(
                    nearest.x,
                    nearest.y,
                    new_x,
                    new_y,
                    alpha,
                    beta
                )
            )

        else:

            best_edge_cost = (
                standard_rrt_edge_cost(
                    nearest.x,
                    nearest.y,
                    new_x,
                    new_y
                )
            )

        best_cost = (
            nearest.cost +
            best_edge_cost
        )

        for neighbour in neighbours:

            if neighbour is nearest:

                continue

            if not edge_is_valid(
                neighbour.x,
                neighbour.y,
                new_x,
                new_y
            ):

                continue

            if weighted:

                edge_cost = (
                    risk_weighted_rrt_edge_cost(
                        neighbour.x,
                        neighbour.y,
                        new_x,
                        new_y,
                        alpha,
                        beta
                    )
                )

            else:

                edge_cost = (
                    standard_rrt_edge_cost(
                        neighbour.x,
                        neighbour.y,
                        new_x,
                        new_y
                    )
                )

            candidate_cost = (
                neighbour.cost +
                edge_cost
            )

            if candidate_cost < best_cost:

                best_cost = candidate_cost
                best_parent = neighbour
                best_edge_cost = edge_cost

        # ----------------------------------------------------
        # Create new node
        # ----------------------------------------------------

        new_node = RRTNode(
            x=float(new_x),
            y=float(new_y),
            parent=best_parent,
            cost=best_cost
        )

        # ----------------------------------------------------
        # Store physical distance and risk information
        # ----------------------------------------------------

        edge_d = edge_distance(
            best_parent.x,
            best_parent.y,
            new_node.x,
            new_node.y
        )

        edge_h = edge_hazard_probability(
            best_parent.x,
            best_parent.y,
            new_node.x,
            new_node.y
        )

        new_node.distance_cost = (
            best_parent.distance_cost +
            edge_d
        )

        new_node.risk_cost = (
            1.0 -
            (
                (1.0 - best_parent.risk_cost)
                *
                (1.0 - edge_h)
            )
        )

        nodes.append(new_node)

        # ----------------------------------------------------
        # Rewire
        # ----------------------------------------------------

        for neighbour in neighbours:

            if neighbour is best_parent:

                continue

            if not edge_is_valid(
                new_node.x,
                new_node.y,
                neighbour.x,
                neighbour.y
            ):

                continue

            if weighted:

                edge_cost = (
                    risk_weighted_rrt_edge_cost(
                        new_node.x,
                        new_node.y,
                        neighbour.x,
                        neighbour.y,
                        alpha,
                        beta
                    )
                )

            else:

                edge_cost = (
                    standard_rrt_edge_cost(
                        new_node.x,
                        new_node.y,
                        neighbour.x,
                        neighbour.y
                    )
                )

            new_candidate_cost = (
                new_node.cost +
                edge_cost
            )

            if new_candidate_cost < neighbour.cost:

                neighbour.parent = new_node
                neighbour.cost = new_candidate_cost

                # Recalculate distance/risk information
                neighbour_edge_d = edge_distance(
                    new_node.x,
                    new_node.y,
                    neighbour.x,
                    neighbour.y
                )

                neighbour_edge_h = (
                    edge_hazard_probability(
                        new_node.x,
                        new_node.y,
                        neighbour.x,
                        neighbour.y
                    )
                )

                neighbour.distance_cost = (
                    new_node.distance_cost +
                    neighbour_edge_d
                )

                neighbour.risk_cost = (
                    1.0 -
                    (
                        (1.0 - new_node.risk_cost)
                        *
                        (1.0 - neighbour_edge_h)
                    )
                )

        # ----------------------------------------------------
        # Check goal connection
        # ----------------------------------------------------

        distance_to_goal = math.sqrt(
            (new_node.x - goal_x) ** 2 +
            (new_node.y - goal_y) ** 2
        )

        if distance_to_goal <= goal_tolerance:

            if edge_is_valid(
                new_node.x,
                new_node.y,
                goal_x,
                goal_y
            ):

                if weighted:

                    goal_edge_cost = (
                        risk_weighted_rrt_edge_cost(
                            new_node.x,
                            new_node.y,
                            goal_x,
                            goal_y,
                            alpha,
                            beta
                        )
                    )

                else:

                    goal_edge_cost = (
                        standard_rrt_edge_cost(
                            new_node.x,
                            new_node.y,
                            goal_x,
                            goal_y
                        )
                    )

                candidate_goal_cost = (
                    new_node.cost +
                    goal_edge_cost
                )

                if candidate_goal_cost < best_goal_cost:

                    goal_node = RRTNode(
                        x=float(goal_x),
                        y=float(goal_y),
                        parent=new_node,
                        cost=candidate_goal_cost
                    )

                    goal_edge_d = edge_distance(
                        new_node.x,
                        new_node.y,
                        goal_x,
                        goal_y
                    )

                    goal_edge_h = (
                        edge_hazard_probability(
                            new_node.x,
                            new_node.y,
                            goal_x,
                            goal_y
                        )
                    )

                    goal_node.distance_cost = (
                        new_node.distance_cost +
                        goal_edge_d
                    )

                    goal_node.risk_cost = (
                        1.0 -
                        (
                            (1.0 - new_node.risk_cost)
                            *
                            (1.0 - goal_edge_h)
                        )
                    )

                    best_goal_node = goal_node
                    best_goal_cost = candidate_goal_cost

    # ========================================================
    # No path found
    # ========================================================

    if best_goal_node is None:

        return None

    # ========================================================
    # Reconstruct
    # ========================================================

    path = reconstruct_path(
        best_goal_node
    )

    metrics = calculate_path_metrics(
        path
    )

    metrics["rrt_cost"] = best_goal_node.cost
    metrics["distance_cost"] = best_goal_node.distance_cost
    metrics["risk_cost"] = best_goal_node.risk_cost
    metrics["iterations"] = max_iterations
    metrics["alpha"] = alpha if weighted else None
    metrics["beta"] = beta if weighted else None

    return {
        "path": path,
        "metrics": metrics,
        "nodes": nodes
    }


# ============================================================
# STANDARD RRT*
# ============================================================

def run_standard_rrt_star(
    start,
    goal,
    random_seed=RRT_RANDOM_SEED
):

    return run_rrt_star(
        start=start,
        goal=goal,
        weighted=False,
        random_seed=random_seed
    )


# ============================================================
# RISK-WEIGHTED RRT*
# ============================================================

def run_risk_weighted_rrt_star(
    start,
    goal,
    alpha=RISK_RRT_ALPHA,
    beta=RISK_RRT_BETA,
    random_seed=RRT_RANDOM_SEED
):

    return run_rrt_star(
        start=start,
        goal=goal,
        weighted=True,
        alpha=alpha,
        beta=beta,
        random_seed=random_seed
    )



# ============================================================
# PROJECT INTEGRATION
# ============================================================

def configure_terrain(terrain_dir):
    """
    Load the same terrain_grid.json used by CMDP and construct the same
    weighted-risk representation used by the CMDP risk model.

    This does not use any hidden PyBullet information.
    """
    global GRID_RESOLUTION
    global valid_terrain_mask
    global GRID_WIDTH, GRID_HEIGHT
    global weighted_risk_grid, height_grid
    global TERRAIN_BOUNDS, TERRAIN_HEIGHT_GRID

    terrain_dir = Path(terrain_dir).resolve()
    terrain_json = terrain_dir / "terrain_grid.json"

    terrain = load_terrain_grid(terrain_json)
    features = prepare_risk_model(terrain)

    GRID_RESOLUTION = float(terrain.resolution)
    GRID_WIDTH, GRID_HEIGHT = terrain.grid.shape

    height_grid = terrain.height_grid
    weighted_risk_grid = features.weighted_risk_grid
    valid_terrain_mask = np.isfinite(height_grid)

    TERRAIN_BOUNDS = dict(terrain.bounds)
    TERRAIN_HEIGHT_GRID = height_grid

    return terrain, features


def path_to_world_csv(path, csv_path):
    """
    Convert the RRT* grid-index path to the same x,y,z CSV format consumed
    by the existing run.py.

    The first and last points therefore use the exact same terrain cell-centre
    coordinates as CMDP for the same start/goal cells.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for x, y in path:
        gx = int(round(x))
        gy = int(round(y))

        if not (0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT):
            raise ValueError(f"Path point is outside terrain grid: {(x, y)}")

        z = float(TERRAIN_HEIGHT_GRID[gx, gy])
        if not np.isfinite(z):
            raise ValueError(f"Path point has invalid terrain height: {(x, y)}")

        wx, wy = grid_to_world(x, y)
        rows.append((float(wx), float(wy), z))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        writer.writerows(rows)


def calculate_beta(start, goal):
    """
    Calculate the approved risk-normalisation coefficient:

        beta = representative edge distance /
               representative edge hazard probability

    The representative edge is the fixed start-to-goal scenario edge,
    evaluated with the exact edge-distance and edge-hazard definitions used
    by the supplied RRT* implementation.

    Thus beta is:
        - terrain/scenario specific,
        - calculated once,
        - independent of alpha,
        - independent of Monte Carlo seed,
        - and never taken from hidden PyBullet information.
    """
    x1, y1 = start
    x2, y2 = goal

    d = edge_distance(x1, y1, x2, y2)
    h = edge_hazard_probability(x1, y1, x2, y2)

    if h <= 0.0:
        # A completely zero-risk direct scenario has no finite direct-edge
        # ratio. Use the median hazard of valid 8-connected terrain edges
        # while keeping the same physical edge-distance definition.
        hazards = []

        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if not valid_terrain_mask[x, y]:
                    continue

                for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    nx = x + dx
                    ny = y + dy

                    if (
                        0 <= nx < GRID_WIDTH
                        and 0 <= ny < GRID_HEIGHT
                        and valid_terrain_mask[nx, ny]
                    ):
                        hazards.append(
                            edge_hazard_probability(x, y, nx, ny)
                        )

        if not hazards:
            raise ValueError("Cannot calculate beta: no valid terrain edges.")

        h = float(np.median(np.asarray(hazards, dtype=np.float64)))

        if h <= 0.0:
            # The entire terrain is risk-free. In that case the risk term
            # contributes nothing regardless of beta, so use the physical
            # grid-edge distance as the finite normalization reference.
            return float(GRID_RESOLUTION)

    return float(d / h)


def load_scenario(scenario_file, scenario_id):
    with open(scenario_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for scenario_type, items in data["scenarios"].items():
        for item in items:
            if item["scenario_id"] == scenario_id:
                return item

    raise KeyError(f"Scenario not found: {scenario_id}")


def save_result(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Standard RRT* / Risk-weighted RRT* baseline."
    )

    parser.add_argument("--terrain-dir", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument(
        "--algorithm",
        choices=("rrt_star", "risk_weighted_rrt_star"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--path-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    terrain_dir = args.terrain_dir.resolve()
    scenario_file = (
        args.scenario_file.resolve()
        if args.scenario_file is not None
        else terrain_dir / "start_goal_scenarios.json"
    )

    scenario = load_scenario(scenario_file, args.scenario_id)

    start = tuple(map(int, scenario["start"]))
    goal = tuple(map(int, scenario["goal"]))

    configure_terrain(terrain_dir)

    if args.algorithm == "rrt_star":
        result = run_standard_rrt_star(
            start,
            goal,
            random_seed=args.seed,
        )
        alpha = None
        beta = None

    else:
        if args.alpha is None:
            raise ValueError(
                "--alpha is required for risk_weighted_rrt_star."
            )

        if not (0.0 <= args.alpha <= 1.0):
            raise ValueError("--alpha must be in [0, 1].")

        beta = (
            float(args.beta)
            if args.beta is not None
            else calculate_beta(start, goal)
        )

        result = run_risk_weighted_rrt_star(
            start,
            goal,
            alpha=float(args.alpha),
            beta=beta,
            random_seed=args.seed,
        )
        alpha = float(args.alpha)

    if result is None:
        output = {
            "terrain_id": terrain_dir.name,
            "scenario_id": args.scenario_id,
            "algorithm": args.algorithm,
            "start": list(start),
            "goal": list(goal),
            "seed": int(args.seed),
            "alpha": alpha,
            "beta": beta,
            "path_found": False,
            "path_length": float("inf"),
            "planned_risk": float("inf"),
            "planner_cost": float("inf"),
            "number_of_nodes": 0,
        }
        save_result(output, args.result_output)
        print(json.dumps(output, indent=2))
        return 2

    path = result["path"]
    metrics = result["metrics"]

    path_to_world_csv(path, args.path_output)

    output = {
        "terrain_id": terrain_dir.name,
        "scenario_id": args.scenario_id,
        "algorithm": args.algorithm,
        "start": list(start),
        "goal": list(goal),
        "seed": int(args.seed),
        "alpha": alpha,
        "beta": beta,
        "path_found": True,
        "path_length": float(metrics["path_length"]),
        "planned_risk": float(metrics["path_risk"]),
        "planner_cost": float(metrics["rrt_cost"]),
        "distance_cost": float(metrics["distance_cost"]),
        "risk_cost": float(metrics["risk_cost"]),
        "number_of_nodes": int(len(result["nodes"])),
        "iterations": int(metrics["iterations"]),
        "path_grid_points": int(len(path)),
        "path_output": str(Path(args.path_output).resolve()),
    }

    save_result(output, args.result_output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

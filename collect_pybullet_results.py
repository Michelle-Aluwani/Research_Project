
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parent

SCENARIO_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:mixed|mud|water|hazard|long)[_-]\d+)",
    re.IGNORECASE
)
D_RE = re.compile(
    r"(?<![A-Za-z0-9])D[_-]?(\d+(?:\.\d+)?)(?![\d.])",
    re.IGNORECASE
)
SEED_RE = re.compile(
    r"(?<![A-Za-z0-9])seed[_-]?(\d+)(?!\d)",
    re.IGNORECASE
)


def flatten_json(obj, prefix=""):
    result = {}

    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            result.update(flatten_json(value, name))
        elif isinstance(value, list):
            result[name] = json.dumps(value, ensure_ascii=False)
        else:
            result[name] = value

    return result


def is_pybullet_result(file):
    name = file.name.lower()

    # Exclude planning-only results.
    if name == "cmdp_result.json":
        return False

    return (
        name == "result.json"
        or name.startswith("result_")
        or name.startswith("result-")
        or name.startswith("pybullet_result")
        or name.startswith("simulation_result")
    )


def find_nested_key(data, target):
    if isinstance(data, dict):
        if target in data:
            return data[target]

        for value in data.values():
            found = find_nested_key(value, target)
            if found is not None:
                return found

    elif isinstance(data, list):
        for value in data:
            found = find_nested_key(value, target)
            if found is not None:
                return found

    return None


def extract_metadata(file, terrain, data):
    name = file.relative_to(terrain).as_posix()
    lower = name.lower()

    scenario_match = SCENARIO_RE.search(name)
    d_match = D_RE.search(name)
    seed_match = SEED_RE.search(name)

    scenario = (
        scenario_match.group(1)
        if scenario_match
        else data.get("scenario_id", data.get("scenario", ""))
    )

    d_value = (
        d_match.group(1)
        if d_match
        else data.get("D", data.get("risk_budget_D", ""))
    )

    seed = (
        seed_match.group(1)
        if seed_match
        else data.get("seed", "")
    )

    if (
        "risk_weighted_rrt_star" in lower
        or "risk_weighted_rrtstar" in lower
        or "riskweighted" in lower
    ):
        planner = "risk_weighted_rrt_star"
    elif "rrt_star" in lower or "rrtstar" in lower:
        planner = "rrt_star"
    elif "baseline" in lower:
        planner = "baseline"
    else:
        planner = "cmdp"

    return {
        "planner": planner,
        "D": d_value,
        "scenario": scenario,
        "scenario_type": (
            re.split(r"[_-]\d+$", str(scenario))[0]
            if scenario else ""
        ),
        "seed": seed,
    }


def process_terrain(terrain):
    print(f"\nProcessing {terrain.name}")

    rows = []

    for file in sorted(terrain.rglob("*.json")):
        if not is_pybullet_result(file):
            continue

        try:
            with file.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                print(f"SKIPPED: {file}")
                continue

            metadata = extract_metadata(file, terrain, data)

            # Require evidence of a specific seed or a
            # PyBullet execution result.
            has_execution_reference = (
                find_nested_key(data, "trajectory_csv") is not None
                or find_nested_key(data, "path_file") is not None
            )

            if (
                metadata["seed"] == ""
                and not has_execution_reference
                and "pybullet" not in file.name.lower()
            ):
                print(f"SKIPPED (not identifiable): {file}")
                continue

            row = metadata
            values = flatten_json(data)

            # Keep explicit trajectory and path references.
            for field in ("trajectory_csv", "path_file"):
                value = find_nested_key(data, field)
                if value is not None and field not in values:
                    values[field] = value

            for key, value in values.items():
                column = key if key not in row and key != "source" \
                    else f"json.{key}"
                row[column] = value

            row["source"] = file.relative_to(ROOT).as_posix()
            rows.append(row)

        except (OSError, ValueError) as error:
            print(f"ERROR: {file}: {error}")

    if not rows:
        print("No PyBullet result JSON files found.")
        return

    fixed_columns = [
        "planner", "D", "scenario", "scenario_type", "seed"
    ]

    preferred_columns = [
        "trajectory_csv", "path_file"
    ]

    all_keys = set().union(*(row.keys() for row in rows))

    preferred_columns = [
        col for col in preferred_columns if col in all_keys
    ]

    other_columns = sorted(
        all_keys
        - set(fixed_columns)
        - set(preferred_columns)
        - {"source"}
    )

    columns = (
        fixed_columns
        + preferred_columns
        + other_columns
        + ["source"]
    )

    output = terrain / "pybullet_results.csv"

    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output}")
    print(f"Execution results: {len(rows)}")
    print(f"Columns: {len(columns)}")


def main():
    terrains = sorted(
        [
            p for p in ROOT.iterdir()
            if p.is_dir()
            and re.fullmatch(r"terrain_\d+", p.name, re.I)
        ],
        key=lambda p: int(p.name.split("_")[1])
    )

    for terrain in terrains:
        process_terrain(terrain)

    print("\nPyBullet collection complete.")


if __name__ == "__main__":
    main()

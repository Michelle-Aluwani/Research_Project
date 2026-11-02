
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


def extract_metadata(file, terrain, data):
    relative = file.relative_to(terrain)
    full_name = relative.as_posix()

    scenario_match = SCENARIO_RE.search(full_name)
    d_match = D_RE.search(full_name)

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

    return {
        "planner": "cmdp",
        "D": d_value,
        "scenario": scenario,
        "scenario_type": (
            re.split(r"[_-]\d+$", str(scenario))[0]
            if scenario else ""
        ),
    }


def process_terrain(terrain):
    print(f"\nProcessing {terrain.name}")

    rows = []

    for file in sorted(terrain.rglob("*")):
        if not file.is_file():
            continue

        if file.name.lower() != "cmdp_result.json":
            continue

        relative_parts = [
            p.lower() for p in file.relative_to(terrain).parts
        ]

        # Do not include baseline results.
        if "baseline" in relative_parts:
            continue

        try:
            with file.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                print(f"SKIPPED (not a JSON object): {file}")
                continue

            row = extract_metadata(file, terrain, data)
            original_values = flatten_json(data)

            # Preserve all JSON fields without overwriting metadata.
            for key, value in original_values.items():
                column = key if key not in row and key != "source" \
                    else f"json.{key}"
                row[column] = value

            row["source"] = file.relative_to(ROOT).as_posix()
            rows.append(row)

        except (OSError, ValueError) as error:
            print(f"ERROR: {file}: {error}")

    if not rows:
        print("No CMDP results found.")
        return

    fixed_columns = [
        "planner", "D", "scenario", "scenario_type"
    ]

    other_columns = sorted(
        set().union(*(row.keys() for row in rows))
        - set(fixed_columns)
        - {"source"}
    )

    columns = fixed_columns + other_columns + ["source"]

    output = terrain / "cmdp_results.csv"

    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output}")
    print(f"Experiments: {len(rows)}")
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

    print("\nCMDP collection complete.")


if __name__ == "__main__":
    main()

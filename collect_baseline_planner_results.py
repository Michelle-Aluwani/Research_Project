
from pathlib import Path
import csv
import json
import re

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

RESULT_FILENAMES = {
    "planner_results.json",
    "planner_result.json",
}

SCENARIO_RE = re.compile(
    r"(mixed|mud|water|hazard|long)[_-]\d+",
    re.IGNORECASE
)

D_RE = re.compile(
    r"(?<![A-Za-z0-9])D[_-]?(\d+(?:\.\d+)?)",
    re.IGNORECASE
)

ALPHA_RE = re.compile(
    r"alpha[_-]?(\d+(?:\.\d+)?)",
    re.IGNORECASE
)

SEED_RE = re.compile(
    r"seed[_-]?(\d+)",
    re.IGNORECASE
)


# ============================================================
# FLATTEN JSON FIELDS
# ============================================================

def flatten_json(data, prefix=""):
    result = {}

    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            result.update(flatten_json(value, name))

        elif isinstance(value, list):
            result[name] = json.dumps(
                value,
                ensure_ascii=False
            )

        else:
            result[name] = value

    return result


# ============================================================
# IDENTIFY THE BASELINE ALGORITHM
# ============================================================

def identify_planner(path, data):
    values = [
        str(path),
        str(data.get("planner", "")),
        str(data.get("algorithm", "")),
        str(data.get("method", "")),
        str(data.get("planner_type", "")),
    ]

    text = "_".join(values).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", text)

    if (
        "risk_weighted_rrt" in normalized
        or "riskweighted_rrt" in normalized
        or "rw_rrt" in normalized
    ):
        return "risk_weighted_rrt_star"

    if (
        "rrt_star" in normalized
        or "rrtstar" in normalized
        or "rrt_" in normalized
    ):
        return "rrt_star"

    return (
        data.get("planner")
        or data.get("algorithm")
        or "unknown"
    )


# ============================================================
# EXTRACT EXPERIMENT INFORMATION
# ============================================================

def extract_metadata(file, terrain, data):
    relative = file.relative_to(terrain).as_posix()

    scenario_match = SCENARIO_RE.search(relative)
    d_match = D_RE.search(relative)
    alpha_match = ALPHA_RE.search(relative)
    seed_match = SEED_RE.search(relative)

    scenario = (
        scenario_match.group(0)
        if scenario_match
        else data.get("scenario_id", data.get("scenario", ""))
    )

    scenario_type = (
        re.split(r"[_-]\d+$", str(scenario))[0]
        if scenario else ""
    )

    return {
        "planner": identify_planner(relative, data),

        "scenario": scenario,

        "scenario_type": scenario_type,

        "D": (
            d_match.group(1)
            if d_match
            else data.get("D", data.get("risk_budget_D", ""))
        ),

        "alpha": (
            alpha_match.group(1)
            if alpha_match
            else data.get("alpha", "")
        ),

        "seed": (
            seed_match.group(1)
            if seed_match
            else data.get("seed", "")
        ),
    }


# ============================================================
# PROCESS ONE TERRAIN
# ============================================================

def process_terrain(terrain):
    print("\n" + "=" * 65)
    print(f"PROCESSING: {terrain.name}")
    print("=" * 65)

    baseline_folder = terrain / "baseline"

    if not baseline_folder.is_dir():
        print("No baseline folder found.")
        return

    rows = []
    files_found = 0

    for file in sorted(baseline_folder.rglob("*")):

        if not file.is_file():
            continue

        if file.name.lower() not in RESULT_FILENAMES:
            continue

        files_found += 1

        try:
            with file.open(
                "r",
                encoding="utf-8-sig"
            ) as f:
                data = json.load(f)

            # Handle JSON objects and lists of result objects.
            if isinstance(data, dict):
                records = [data]

            elif isinstance(data, list):
                records = data

            else:
                print(f"SKIPPED: {file}")
                continue

            for index, record in enumerate(records, start=1):

                if not isinstance(record, dict):
                    continue

                row = extract_metadata(
                    file, terrain, record
                )

                values = flatten_json(record)

                # Preserve all JSON fields, including ones
                # that share names with metadata columns.
                for key, value in values.items():
                    column = (
                        key
                        if key not in row and key != "source"
                        else f"json.{key}"
                    )
                    row[column] = value

                if len(records) > 1:
                    row["record_index"] = index

                row["source"] = (
                    file.relative_to(ROOT).as_posix()
                )

                rows.append(row)

            print(
                f"[COLLECTED] "
                f"{file.relative_to(terrain)}"
            )

        except (OSError, ValueError) as error:
            print(f"ERROR: {file}: {error}")

    # --------------------------------------------------------
    # BUILD CSV COLUMNS
    # --------------------------------------------------------

    fixed_columns = [
        "planner",
        "scenario",
        "scenario_type",
        "D",
        "alpha",
        "seed",
    ]

    all_columns = set()

    for row in rows:
        all_columns.update(row.keys())

    extra_columns = sorted(
        all_columns
        - set(fixed_columns)
        - {"source"}
    )

    columns = (
        fixed_columns
        + extra_columns
        + ["source"]
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output_file = (
        terrain / "baseline_planner_results.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nCOLLECTION COMPLETE")
    print(f"JSON files found: {files_found}")
    print(f"Result rows: {len(rows)}")
    print(f"CSV columns: {len(columns)}")
    print(f"Saved: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():

    terrains = sorted(
        [
            folder
            for folder in ROOT.iterdir()
            if (
                folder.is_dir()
                and re.fullmatch(
                    r"terrain_\d+",
                    folder.name,
                    re.IGNORECASE
                )
            )
        ],
        key=lambda p: int(p.name.split("_")[1])
    )

    if not terrains:
        print("No terrain folders found.")
        return

    print(f"Found {len(terrains)} terrains.")

    for terrain in terrains:
        process_terrain(terrain)

    print("\nALL BASELINE RESULTS COLLECTED.")


if __name__ == "__main__":
    main()

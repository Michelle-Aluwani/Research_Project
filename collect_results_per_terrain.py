
from pathlib import Path
import csv
import json
import zipfile
import re

# ============================================================
# CONFIGURATION
# ============================================================

# The script must be in the same directory as terrain_1,
# terrain_2, terrain_3, etc.
ROOT_FOLDER = Path(__file__).resolve().parent

CMDP_OUTPUT = ROOT_FOLDER / "cmdp"
BASELINE_OUTPUT = ROOT_FOLDER / "baseline"

# Files to collect
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl"}

# Scenario names, for example: mud_01, hazard_05, mixed_02
SCENARIO_PATTERN = re.compile(
    r"^(mixed|mud|water|hazard|long)_\d+$",
    re.IGNORECASE
)

TERRAIN_PATTERN = re.compile(r"^terrain_\d+$", re.IGNORECASE)


# ============================================================
# CONVERT FILE CONTENT INTO JSON
# ============================================================

def convert_to_json(file_path):
    extension = file_path.suffix.lower()

    if extension in {".csv", ".tsv"}:
        delimiter = "," if extension == ".csv" else "\t"

        with file_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            rows = list(reader)

        return json.dumps(rows, indent=2, ensure_ascii=False)

    if extension == ".json":
        with file_path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

        return json.dumps(data, indent=2, ensure_ascii=False)

    if extension == ".jsonl":
        rows = []

        with file_path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))

        return json.dumps(rows, indent=2, ensure_ascii=False)

    return None


# ============================================================
# CREATE UNIQUE JSON FILENAMES
# ============================================================

def make_filename(file_path, terrain_folder):
    relative_path = file_path.relative_to(terrain_folder)

    # Include the terrain, scenario, D value, seed,
    # original filename, etc.
    parts = [terrain_folder.name] + list(relative_path.parts)

    name = "__".join(parts)

    # Replace the original extension with .json
    return str(Path(name).with_suffix(".json"))


# ============================================================
# CREATE A ZIP FROM A SET OF FOLDERS
# ============================================================

def create_results_zip(terrain_folder, source_folders,
                       output_folder, result_type):

    output_folder.mkdir(parents=True, exist_ok=True)

    zip_name = f"{terrain_folder.name}_{result_type}_results.zip"
    zip_path = output_folder / zip_name

    files = []

    # Search every selected folder recursively.
    for folder in source_folders:
        for file in folder.rglob("*"):
            if (
                file.is_file()
                and file.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                files.append(file)

    files = sorted(files, key=lambda p: str(p).lower())

    if not files:
        print(f"  No {result_type} files found.")
        return

    converted_count = 0
    failed_count = 0
    used_names = set()

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6
    ) as archive:

        for file in files:
            try:
                json_content = convert_to_json(file)

                if json_content is None:
                    continue

                filename = make_filename(file, terrain_folder)

                # Prevent overwriting when CSV and JSON files
                # share the same original filename.
                original_name = Path(filename)
                counter = 2

                while filename in used_names:
                    filename = (
                        f"{original_name.stem}"
                        f"__{counter}.json"
                    )
                    counter += 1

                used_names.add(filename)

                archive.writestr(filename, json_content)

                converted_count += 1

                print(
                    f"  [{converted_count}] "
                    f"{file.relative_to(terrain_folder)}"
                    f" -> {filename}"
                )

            except Exception as error:
                failed_count += 1
                print(f"  FAILED: {file} ({error})")

    print(f"\n  ZIP CREATED: {zip_path.name}")
    print(f"  Files included: {converted_count}")
    print(f"  Failed conversions: {failed_count}")
    print(
        f"  ZIP size: "
        f"{zip_path.stat().st_size / (1024 * 1024):.2f} MB"
    )


# ============================================================
# PROCESS EACH TERRAIN
# ============================================================

def process_terrain(terrain_folder):

    print("\n" + "=" * 65)
    print(f"PROCESSING {terrain_folder.name}")
    print("=" * 65)

    # --------------------------------------------------------
    # CMDP: All scenario folders outside baseline
    # --------------------------------------------------------

    cmdp_scenarios = sorted(
        [
            folder
            for folder in terrain_folder.iterdir()
            if (
                folder.is_dir()
                and SCENARIO_PATTERN.fullmatch(folder.name)
            )
        ],
        key=lambda p: p.name.lower()
    )

    print(f"\nCMDP scenarios found: {len(cmdp_scenarios)}")

    for scenario in cmdp_scenarios:
        print(f"  - {scenario.name}")

    create_results_zip(
        terrain_folder,
        cmdp_scenarios,
        CMDP_OUTPUT,
        "cmdp"
    )

    # --------------------------------------------------------
    # BASELINE: Everything inside terrain_n/baseline
    # --------------------------------------------------------

    baseline_folder = terrain_folder / "baseline"

    if baseline_folder.is_dir():

        print("\nCollecting baseline results...")

        create_results_zip(
            terrain_folder,
            [baseline_folder],
            BASELINE_OUTPUT,
            "baseline"
        )

    else:
        print("\nNo baseline folder found.")


# ============================================================
# MAIN
# ============================================================

def main():

    terrain_folders = sorted(
        [
            folder
            for folder in ROOT_FOLDER.iterdir()
            if (
                folder.is_dir()
                and TERRAIN_PATTERN.fullmatch(folder.name)
            )
        ],
        key=lambda p: int(p.name.split("_")[1])
    )

    if not terrain_folders:
        print("No terrain folders found!")
        print(f"Checked: {ROOT_FOLDER}")
        return

    print(f"Found {len(terrain_folders)} terrain folders.")

    for terrain_folder in terrain_folders:
        process_terrain(terrain_folder)

    print("\n" + "=" * 65)
    print("ALL TERRAIN RESULTS COLLECTED")
    print("=" * 65)

    print(f"\nCMDP ZIPs stored in: {CMDP_OUTPUT}")
    print(f"Baseline ZIPs stored in: {BASELINE_OUTPUT}")


if __name__ == "__main__":
    main()

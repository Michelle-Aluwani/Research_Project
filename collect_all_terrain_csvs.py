
from pathlib import Path
import shutil
import re

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_FOLDER = Path(__file__).resolve().parent

OUTPUT_FOLDER = ROOT_FOLDER / "COLLECTED_TERRAIN_CSVS"

CSV_FILES = [
    "cmdp_results.csv",
    "baseline_planner_results.csv",
    "pybullet_results.csv",
]


# ============================================================
# COLLECT CSV FILES FROM ALL TERRAINS
# ============================================================

def collect_csvs():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    terrain_folders = sorted(
        [
            folder
            for folder in ROOT_FOLDER.iterdir()
            if folder.is_dir()
            and re.fullmatch(
                r"terrain_\d+",
                folder.name,
                re.IGNORECASE
            )
        ],
        key=lambda p: int(p.name.split("_")[1])
    )

    if not terrain_folders:
        print("No terrain folders found.")
        return

    total_copied = 0
    total_missing = 0

    print(f"Found {len(terrain_folders)} terrains.")

    for terrain in terrain_folders:

        print("\n" + "=" * 60)
        print(f"PROCESSING: {terrain.name}")
        print("=" * 60)

        for csv_name in CSV_FILES:

            source = terrain / csv_name

            if not source.is_file():
                print(f"MISSING: {csv_name}")
                total_missing += 1
                continue

            # Include the terrain name to prevent
            # filenames from different terrains colliding.
            new_name = f"{terrain.name}_{csv_name}"

            destination = OUTPUT_FOLDER / new_name

            # Copy without changing original data.
            # Rerunning updates the previously collected
            # CSV for that same terrain.
            shutil.copy2(source, destination)

            total_copied += 1

            print(f"COPIED: {new_name}")

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)

    print(f"Terrains processed: {len(terrain_folders)}")
    print(f"CSV files copied: {total_copied}")
    print(f"Missing CSV files: {total_missing}")
    print(f"Output folder: {OUTPUT_FOLDER}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    collect_csvs()

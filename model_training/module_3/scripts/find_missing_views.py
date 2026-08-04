"""
find_missing_views.py

Scans dataset/health_labelled/<species>/<damage_level>/top|bottom/*
and reports every leaf whose image exists in only ONE of top/bottom.

Usage (Windows):
    D:\\Python313\\python.exe find_missing_views.py

Edit ROOT below if your path differs.
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict

# ---- EDIT THIS if needed ----
ROOT = Path(r"D:\Desktop\UoM\Academic\FYP\ayurveda-recognition\model_training\module_3\dataset\health_labelled")
VIEWS = ("top", "bottom")
IMG_EXTS = {".jpg", ".jpeg", ".png"}
OUT_CSV = Path("missing_views_report.csv")
# ------------------------------

def leaf_key_from_filename(fname: str) -> str:
    """
    Strip extension and any trailing view/variant markers so that
    e.g. 'image_25.jpg', 'PXL_20240101_010101.jpg', 'test_003.jpg'
    map to a stable leaf identifier. We use the filename stem as-is
    (no augmentation variants exist on disk -- those are generated
    in-memory per your pipeline), so the stem itself IS the leaf id
    within a given species/damage_level/view folder.
    """
    stem = Path(fname).stem
    return stem


def main():
    if not ROOT.exists():
        print(f"ERROR: root path does not exist: {ROOT}")
        return

    rows = []
    species_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])

    total_checked = 0
    total_missing = 0

    for species_dir in species_dirs:
        species = species_dir.name
        level_dirs = sorted([d for d in species_dir.iterdir() if d.is_dir()])

        for level_dir in level_dirs:
            level = level_dir.name

            view_files = {}
            for view in VIEWS:
                view_path = level_dir / view
                if not view_path.exists():
                    view_files[view] = set()
                    continue
                files = {
                    leaf_key_from_filename(f.name)
                    for f in view_path.iterdir()
                    if f.is_file() and f.suffix.lower() in IMG_EXTS
                }
                view_files[view] = files

            all_leaf_ids = view_files["top"] | view_files["bottom"]
            total_checked += len(all_leaf_ids)

            for leaf_id in sorted(all_leaf_ids):
                in_top = leaf_id in view_files["top"]
                in_bottom = leaf_id in view_files["bottom"]
                if in_top and in_bottom:
                    continue  # fine, both views present

                missing_view = "bottom" if in_top else "top"
                total_missing += 1
                rows.append({
                    "species": species,
                    "damage_level": level,
                    "leaf_id": leaf_id,
                    "missing_view": missing_view,
                    "expected_path": str(level_dir / missing_view / f"{leaf_id}.jpg (or .jpeg/.png)"),
                })

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "damage_level", "leaf_id", "missing_view", "expected_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Checked {total_checked} unique leaf_ids across {len(species_dirs)} species folders.")
    print(f"Found {total_missing} leaves missing one view.")
    print(f"Full report written to: {OUT_CSV.resolve()}")
    print()
    if rows:
        print("Preview (first 20):")
        for r in rows[:20]:
            print(f"  [{r['species']}/{r['damage_level']}] {r['leaf_id']} -- missing '{r['missing_view']}' view")


if __name__ == "__main__":
    main()
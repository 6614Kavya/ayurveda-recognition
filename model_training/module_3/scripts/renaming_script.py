"""
rename_health_images.py

Renames raw camera images (e.g. PXL_20260507_061626518.jpg) inside
dataset/health/<species>/<level>/<view>/ into simple sequential names
(e.g. image_01.jpg, image_02.jpg, ...), per folder.

Why sequential-per-folder (not global): filenames only need to be unique
within their own folder path, since the full image_path (species/level/view/filename)
is what's stored as the unique key downstream. Keeping numbering local to each
folder also keeps the rename mapping easy to review.

Ordering: sorts images by EXIF DateTimeOriginal when available (chronological
capture order), falling back to filename sort if EXIF is missing. This does NOT
attempt top/bottom pairing -- that's a separate script. It just gives you a
stable, sensible numbering before you go pick out the 3 test images per folder.

Safety:
- Defaults to DRY RUN. Nothing is renamed until you pass --apply.
- Skips any file already starting with "test_" (won't touch designated test images).
- Skips any file already starting with "image_" (won't re-rename already-processed
  images if you run this script again on a folder you've already touched). New
  numbering picks up after the highest existing image_N index, so no collisions.
- Writes a rename_log.csv into each folder it touches, mapping old -> new name,
  so you can always trace a renamed file back to its original camera filename.
- Skips non-image files automatically.

WHERE TO PUT THIS FILE:
    Place it at the root of your project, e.g.:
        ayurveda-recognition/model_training/module_3/renaming_script.py
    (create a `scripts/` folder next to `preprocessing/` and `feature_extraction/`
    if you don't have one already -- this is a one-off utility, not part of the
    importable pipeline package, so it doesn't need to live inside preprocessing/.)

HOW TO RUN:
    1. Install Pillow if you don't already have it:
           pip install Pillow

    2. Dry run first (always do this before --apply):
           python renaming_script.py --root dataset/health

       This prints every planned rename without touching any files.

    3. Once the planned renames look correct, actually apply them:
           python renaming_script.py --root dataset/health --apply

    4. To rename just one species (e.g. while testing), narrow the root:
           python renaming_script.py --root dataset/health/beli --apply

Adjust ROOT default below or always pass --root explicitly.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Reverse-lookup for the EXIF DateTimeOriginal tag id
DATETIME_ORIGINAL_TAG = None
for _tag_id, _tag_name in TAGS.items():
    if _tag_name == "DateTimeOriginal":
        DATETIME_ORIGINAL_TAG = _tag_id
        break


def get_capture_time(path: Path):
    """Return EXIF DateTimeOriginal as a sortable string, or None if unavailable."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif and DATETIME_ORIGINAL_TAG in exif:
                return exif[DATETIME_ORIGINAL_TAG]  # format: 'YYYY:MM:DD HH:MM:SS'
    except Exception:
        pass
    return None


def find_leaf_folders(root: Path):
    """
    Yield every folder that directly contains image files, i.e. the
    .../<species>/<level>/<view>/ leaf directories.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        images = [f for f in path.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            yield path, images


def plan_renames(folder: Path, images: list[Path]):
    """
    Decide the old_name -> new_name mapping for one folder.
    Files already named test_* or image_* (i.e. already renamed) are left
    untouched and excluded from renumbering. New numbering continues after
    the highest existing image_N index, so already-renamed files are never
    overwritten or collided with.
    """
    def is_already_renamed(f: Path):
        name = f.name.lower()
        return name.startswith("test_") or name.startswith("image_")

    protected = [f for f in images if is_already_renamed(f)]
    to_rename = [f for f in images if f not in protected]

    # Find the highest existing image_N index among protected files so new
    # numbering doesn't collide with already-renamed images.
    start = 1
    for f in protected:
        stem = f.stem  # e.g. "image_07"
        if stem.lower().startswith("image_"):
            suffix_num = stem.split("_", 1)[1]
            if suffix_num.isdigit():
                start = max(start, int(suffix_num) + 1)

    # Sort by EXIF capture time when available, else fall back to filename.
    def sort_key(f: Path):
        t = get_capture_time(f)
        return (0, t) if t else (1, f.name)

    to_rename.sort(key=sort_key)

    plan = []
    for i, f in enumerate(to_rename, start=start):
        new_name = f"image_{i:02d}{f.suffix.lower()}"
        plan.append((f, folder / new_name))

    if protected:
        print(f"    (skipping {len(protected)} already-renamed file(s) in {folder})")

    return plan


def apply_plan(folder: Path, plan: list[tuple[Path, Path]]):
    """Perform the renames and append to a rename_log.csv in the folder."""
    log_path = folder / "rename_log.csv"
    write_header = not log_path.exists()

    # Rename via a temp suffix first to avoid collisions when old/new names overlap
    temp_pairs = []
    for old, new in plan:
        temp = old.with_suffix(old.suffix + ".tmp_rename")
        old.rename(temp)
        temp_pairs.append((temp, new, old.name))

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["old_filename", "new_filename"])
        for temp, new, old_name in temp_pairs:
            temp.rename(new)
            writer.writerow([old_name, new.name])


def main():
    parser = argparse.ArgumentParser(description="Sequentially rename health-branch leaf images.")
    parser.add_argument("--root", type=str, default="dataset/health",
                         help="Root folder to walk (default: dataset/health)")
    parser.add_argument("--apply", action="store_true",
                         help="Actually rename files. Without this flag, it's a dry run.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root folder not found: {root.resolve()}")
        return

    total_planned = 0
    for folder, images in find_leaf_folders(root):
        plan = plan_renames(folder, images)
        if not plan:
            continue
        print(f"\n{folder} ({len(plan)} file(s) to rename):")
        for old, new in plan:
            print(f"  {old.name}  ->  {new.name}")
        total_planned += len(plan)

        if args.apply:
            apply_plan(folder, plan)

    print(f"\nTotal files {'renamed' if args.apply else 'planned for renaming'}: {total_planned}")
    if not args.apply:
        print("This was a DRY RUN. Re-run with --apply to actually rename files.")


if __name__ == "__main__":
    main()
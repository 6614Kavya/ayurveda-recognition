"""
rename_last3_as_test.py

Renames the last 3 images in every health dataset folder as:
    test_001.jpg
    test_002.jpg
    test_003.jpg

Folder structure expected:
dataset/health/
    species/
        level/
            view/
                image_01.jpg
                image_02.jpg
                ...

Ordering:
- Uses EXIF DateTimeOriginal when available
- Falls back to filename sorting

Safety:
- Dry run by default
- Use --apply to actually rename
- Skips existing test_*.jpg files
- Creates rename_log.csv in each affected folder
"""

import argparse
import csv
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# Find EXIF DateTimeOriginal ID
DATETIME_ORIGINAL_TAG = None
for tag_id, tag_name in TAGS.items():
    if tag_name == "DateTimeOriginal":
        DATETIME_ORIGINAL_TAG = tag_id
        break


def get_capture_time(path: Path):
    """Get original capture timestamp from EXIF."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()

            if exif and DATETIME_ORIGINAL_TAG in exif:
                return exif[DATETIME_ORIGINAL_TAG]

    except Exception:
        pass

    return None


def find_image_folders(root: Path):
    """Find folders containing images."""
    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue

        images = [
            f for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images:
            yield folder, images


def sort_images(images):
    """Sort chronologically using EXIF, fallback filename."""

    def key(f):
        t = get_capture_time(f)

        if t:
            return (0, t)

        return (1, f.name)

    return sorted(images, key=key)


def create_plan(folder, images):

    # If this folder already has test images, skip completely
    existing_tests = [
        f for f in images
        if f.name.lower().startswith("test_")
    ]

    if existing_tests:
        print(
            f"    Skipping {folder} "
            f"(already contains {len(existing_tests)} test image(s))"
        )
        return []


    # Only non-test images are considered
    available = [
        f for f in images
        if not f.name.lower().startswith("test_")
    ]


    if len(available) < 3:
        return []


    ordered = sort_images(available)

    last_three = ordered[-3:]


    plan = []

    for idx, old in enumerate(last_three, start=1):

        new = folder / f"test_{idx:03d}{old.suffix.lower()}"

        plan.append((old, new))


    return plan

def apply_plan(folder, plan):

    log_file = folder / "rename_log.csv"

    write_header = not log_file.exists()

    temp_pairs = []
    for old, new in plan:
        if new.exists():
            print(f"Skipping {old.name}: {new.name} already exists")
        continue
    
    # temporary rename avoids collisions
    for old, new in plan:

        temp = old.with_suffix(old.suffix + ".tmp_test")

        old.rename(temp)

        temp_pairs.append(
            (temp, new, old.name)
        )

    
    with open(log_file, "a", newline="") as f:

        writer = csv.writer(f)

        if write_header:
            writer.writerow(
                ["old_filename", "new_filename"]
            )


        for temp, new, old_name in temp_pairs:

            temp.rename(new)

            writer.writerow(
                [old_name, new.name]
            )


def main():

    parser = argparse.ArgumentParser(
        description="Rename last 3 health images as test_001-003"
    )

    parser.add_argument(
        "--root",
        default="dataset/health"
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files"
    )

    args = parser.parse_args()


    root = Path(args.root)

    if not root.exists():

        print("Folder not found:", root)

        return


    total = 0


    for folder, images in find_image_folders(root):

        plan = create_plan(folder, images)

        if not plan:
            continue


        print("\n", folder)

        for old, new in plan:

            print(
                f"  {old.name} -> {new.name}"
            )

        total += len(plan)


        if args.apply:

            apply_plan(folder, plan)


    print(
        "\nTotal:",
        total,
        "files",
        "renamed" if args.apply else "planned"
    )


    if not args.apply:

        print(
            "DRY RUN ONLY. Use --apply to rename."
        )


if __name__ == "__main__":
    main()
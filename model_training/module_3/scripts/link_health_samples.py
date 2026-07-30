"""
Run locally from module_3/:

    python link_health_samples2.py --species nil_awariya --levels damaged_low damaged_mid damaged_high --n-per-level 6

Hard-links N images per view (top/bottom) from EACH named health-level
subfolder under dataset/health/<species>/ into dataset/raw/<species>/<view>/.

Unlike the first version, --levels must be given explicitly -- it will
NOT auto-discover every subfolder, so a folder like "healthy" (whose
images may already overlap your existing raw training set) is never
picked up unless you name it yourself.
"""
import argparse
import subprocess
import sys
from pathlib import Path

IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]


def hardlink(src: Path, dest: Path):
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/H", str(dest), str(src)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  FAILED: {src.name} -> {dest.name}")
        print(f"    {result.stderr.strip()}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, help="e.g. nil_awariya")
    ap.add_argument("--levels", nargs="+", required=True,
                     help="exact folder names to include, e.g. damaged_low damaged_mid damaged_high")
    ap.add_argument("--health-root", default="dataset/health")
    ap.add_argument("--raw-root", default="dataset/raw")
    ap.add_argument("--n-per-level", type=int, default=6)
    ap.add_argument("--views", nargs="+", default=["top", "bottom"])
    args = ap.parse_args()

    health_species_dir = Path(args.health_root) / args.species
    if not health_species_dir.exists():
        print(f"ERROR: {health_species_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    total_linked = 0
    for level_name in args.levels:
        level_dir = health_species_dir / level_name
        if not level_dir.exists():
            print(f"  ERROR: {level_dir} not found -- check the exact folder name. Skipping.")
            continue

        for view in args.views:
            src_dir = level_dir / view
            if not src_dir.exists():
                print(f"  (skip) {src_dir} not found")
                continue

            imgs = []
            for ext in IMG_EXTS:
                imgs.extend(src_dir.glob(ext))
            imgs = sorted(set(imgs))[:args.n_per_level]

            dest_dir = Path(args.raw_root) / args.species / view
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img in imgs:
                new_name = f"HEALTH_{level_name}_{img.name}"
                dest = dest_dir / new_name
                if dest.exists():
                    print(f"  (exists, skip) {dest}")
                    continue
                if hardlink(img, dest):
                    total_linked += 1
                    print(f"  linked: {level_name}/{view}/{img.name} -> {dest}")

    print(f"\nDone. {total_linked} hard links created under {args.raw_root}/{args.species}/")


if __name__ == "__main__":
    main()
"""
Run locally from module_3/:

    python unlink_health_samples.py --species nil_awariya

Removes only the hard links previously created by link_health_samples.py
(identifiable by their "HEALTH_" filename prefix) from dataset/raw/<species>/.

Safe: a hard link is just a second filename pointing at the same file data.
Deleting this filename does NOT delete or touch the underlying file, which
still exists under its original name in dataset/health/. Your original
camera-named train images in dataset/raw/ are untouched -- only files
starting with "HEALTH_" are removed.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, help="e.g. nil_awariya")
    ap.add_argument("--raw-root", default="dataset/raw")
    ap.add_argument("--views", nargs="+", default=["top", "bottom"])
    ap.add_argument("--dry-run", action="store_true",
                     help="List what would be removed without deleting")
    args = ap.parse_args()

    total = 0
    for view in args.views:
        vdir = Path(args.raw_root) / args.species / view
        if not vdir.exists():
            print(f"  (skip) {vdir} not found")
            continue
        matches = sorted(vdir.glob("HEALTH_*"))
        for f in matches:
            if args.dry_run:
                print(f"  [dry-run] would remove: {f}")
            else:
                f.unlink()
                print(f"  removed: {f}")
            total += 1

    verb = "Would remove" if args.dry_run else "Removed"
    print(f"\n{verb} {total} HEALTH_* link(s) under {args.raw_root}/{args.species}/")
    if args.dry_run:
        print("Re-run without --dry-run to actually delete.")


if __name__ == "__main__":
    main()
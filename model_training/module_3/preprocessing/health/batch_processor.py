"""
preprocessing/health/batch_processor.py

Health-branch batch processor for the OLD, LABELED dataset layout:

    dataset/health_labelled/<species>/<level>/<top|bottom>/*.jpg
    level in {healthy, low, mid, high}

This is a variant of preprocessing/health/batch_processor.py, which was
rewritten to walk the NEWER, flattened, unlabeled layout
(<species>/<top|bottom>/*.jpg, no level folder at all) -- see that
file's own docstring for why. Running that script against a labeled
tree produces "no 'top' folder under <species>" for every species,
because it looks for top/bottom directly under species and the level
folder sits in between.

Use THIS script specifically to regenerate health_features_top.csv /
health_features_bottom.csv WITH a "level" column, which is required by
models/health/train_health_index.py's fit_health_index_binary() (needs
level to know which leaves are "healthy" for baseline normalization,
and to build the healthy-vs-unhealthy binary fit target).

    python -m preprocessing.health.batch_processor --dataset-root dataset/health_labelled --out-dir processed/features --workers 8 --augment

Everything else (parallelization, de-duplicated masking, augmentation,
CSV writing, qc handling) is unchanged from batch_processor.py -- only
the folder-walking, leaf_id construction, and the added "level" field
differ. Keep both scripts: this one for (re-)training the baseline-
normalized model, the flattened one for whatever the unlabeled
production dataset is used for.
"""
import argparse
import csv
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

from preprocessing.shared.resize import letterbox_resize
from preprocessing.shared.masking import select_mask, qc_check
from preprocessing.shared.augmentation import augment_health_resized_with_masks
from preprocessing.health.pipeline import run_health_pipeline_from_resized, _extract_all_features

VIEWS = ("top", "bottom")
LEVELS = ("healthy", "damaged_low", "damaged_mid", "damaged_high")
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# "level" added vs. the flattened script's BASE_COLS -- this is the whole
# point of this variant.
BASE_COLS = ["leaf_id", "variant_id", "species", "level", "view",
             "is_test", "is_augmented", "source_path"]


def _iter_images(dataset_root: Path):
    """Yield (species, level, view, image_path) for every image in the tree.

    Walks <species>/<level>/<top|bottom>/*.jpg. Any species subfolder
    that isn't one of LEVELS is skipped with a warning (rather than
    silently misreading it as a view), so a typo'd or unexpected level
    folder name surfaces immediately instead of quietly dropping images.
    """
    for species_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for level_dir in sorted(p for p in species_dir.iterdir() if p.is_dir()):
            if level_dir.name not in LEVELS:
                print(f"[warn] '{level_dir.name}' under {species_dir.name} is not a "
                      f"recognised level {LEVELS}, skipping")
                continue
            for view in VIEWS:
                view_dir = level_dir / view
                if not view_dir.is_dir():
                    print(f"[warn] no '{view}' folder under {species_dir.name}/{level_dir.name}, "
                          f"skipping that view")
                    continue
                for img_path in sorted(
                    p for ext in IMG_EXTS for p in view_dir.glob(f"*{ext}")
                ):
                    yield species_dir.name, level_dir.name, view, img_path


def _make_leaf_id(species: str, level: str, img_path: Path) -> str:
    # level folded into leaf_id: the same filename (e.g. image_01.jpg) can
    # legitimately recur across different level folders for the same
    # species, so level must be part of the uniqueness key here (unlike
    # the flattened script, where there's only one folder per species/view
    # and no collision risk).
    return f"{species}__{level}__{img_path.stem}"


def _process_one(task):
    """
    Worker function: fully processes ONE image (original row + augmented
    variants if requested) and returns (view, rows, status, img_path_str).
    Mirrors batch_processor.py's _process_one, with level threaded through
    base_fields and leaf_id.
    """
    species, level, view, img_path_str, augment, n_aug = task
    img_path = Path(img_path_str)

    img_bgr = cv2.imread(img_path_str)
    if img_bgr is None:
        return view, [], "read_fail", img_path_str

    is_test = img_path.stem.startswith("test_")
    leaf_id = _make_leaf_id(species, level, img_path)

    base_fields = {
        "leaf_id": leaf_id, "variant_id": 0, "species": species, "level": level, "view": view,
        "is_test": is_test, "is_augmented": False, "source_path": img_path_str,
    }

    resized, _ = letterbox_resize(img_bgr)
    mask_final, mask_choice, diag = select_mask(resized)
    mask_before_holefill = diag.get("mask_before_holefill")
    rachis_mask = diag.get("rachis_mask")

    qc_ok, qc_reason = qc_check(diag)
    if not qc_ok:
        row = {"image_path": img_path_str, "qc_pass": False, "qc_reason": qc_reason}
        row.update(base_fields)
        return view, [row], "qc_fail", img_path_str

    # health branch: masked_raw only, NO enhancement, ever
    masked_raw = cv2.bitwise_and(resized, resized, mask=mask_final.astype(np.uint8))

    row = {"image_path": img_path_str, "qc_pass": True, "mask_choice": mask_choice}
    row.update(_extract_all_features(masked_raw, mask_final, mask_before_holefill, rachis_mask))
    row.update(base_fields)
    rows = [row]

    if augment and not is_test:
        variants = augment_health_resized_with_masks(
            resized, mask_final, mask_before_holefill, rachis_mask, n=n_aug
        )
        for i, (aug_img, aug_mask, aug_mask_bhf, aug_rachis) in enumerate(variants):
            aug_row = run_health_pipeline_from_resized(
                aug_img, aug_mask, aug_mask_bhf, aug_rachis,
                image_path=f"{img_path_str}#aug{i}",
            )
            aug_row.update({
                "leaf_id": leaf_id, "variant_id": i + 1, "species": species, "level": level, "view": view,
                "is_test": False, "is_augmented": True, "source_path": img_path_str,
            })
            rows.append(aug_row)

    return view, rows, "ok", img_path_str


def run_batch(dataset_root: Path, out_dir: Path, augment: bool = False, n_aug: int = 6,
              workers: int = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_view = {"top": [], "bottom": []}
    n_ok, n_qc_fail, n_read_fail = 0, 0, 0

    workers = workers or os.cpu_count() or 1

    tasks = [
        (species, level, view, str(img_path), augment, n_aug)
        for species, level, view, img_path in _iter_images(dataset_root)
    ]

    if not tasks:
        print(f"[warn] no images found under {dataset_root}")
        print(f"  Expected layout: {dataset_root}/<species>/<level>/<top|bottom>/*.jpg, "
              f"level in {LEVELS}")
        return

    print(f"[info] {len(tasks)} images, {workers} worker process(es), augment={augment}")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_one, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing leaves", unit="img"):
            view, rows, status, img_path_str = future.result()

            if status == "read_fail":
                print(f"[error] could not read {img_path_str}, skipping")
                n_read_fail += 1
                continue

            rows_by_view[view].extend(rows)
            if status == "ok":
                n_ok += 1
            else:
                n_qc_fail += 1

    for view in VIEWS:
        rows = rows_by_view[view]
        if not rows:
            print(f"[warn] no rows for view={view}, skipping CSV write")
            continue
        # union of all keys across rows (qc-failed rows have fewer columns)
        all_cols = list(BASE_COLS)
        for r in rows:
            for k in r.keys():
                if k not in all_cols:
                    all_cols.append(k)
        out_path = out_dir / f"health_features_{view}_augmented.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] wrote {len(rows)} rows -> {out_path}")

    print(f"\n[summary] qc_pass={n_ok}, qc_fail={n_qc_fail}, read_fail={n_read_fail}")


def main():
    parser = argparse.ArgumentParser(
        description="VedaVision health-branch batch feature extraction (LABELED layout)"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/health_labelled"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed/features_labelled"))
    parser.add_argument("--augment", action="store_true", help="also process augmented variants of train images")
    parser.add_argument("--n-aug", type=int, default=6)
    parser.add_argument("--workers", type=int, default=None,
                         help="number of worker processes (default: os.cpu_count())")
    args = parser.parse_args()

    run_batch(args.dataset_root, args.out_dir, augment=args.augment, n_aug=args.n_aug,
              workers=args.workers)


if __name__ == "__main__":
    main()
"""
VedaVision — Batch Processor
==============================
Processes the full dataset folder hierarchy and produces:

  TRAIN mode  (--mode train):
  • features/vedavision_features_train.csv        — all successful train rows
  • features/vedavision_features_train_clf.csv    — features-only (no metadata), for classifier
  • features/checkpoint_*.csv                     — partial CSVs every CHECKPOINT_EVERY images

  TEST mode   (--mode test):
  • features/vedavision_features_test.csv         — all successful test rows
  • features/vedavision_features_test_clf.csv     — features-only (no metadata), for evaluation

  Both modes:
  • diagnostics/failures.csv                      — QC-failed / exception rows
  • diagnostics/failures/<species>/<view>/        — copies of failed images
  • diagnostics/per_image/                        — per-image JSON diagnostics
  • diagnostics/report.json                       — full batch summary

Image naming convention (REQUIRED):
    train images → filename starts with  "train_"   e.g. train_001.jpg
    test  images → filename starts with  "test_"    e.g. test_001.jpg

    Same leaf ID (e.g. test_003) must appear in both top/ and bottom/ for that species.
    This enforces leaf-level train/test separation (no pseudo-replication).

Dataset folder structure expected:
    dataset/raw/
        <species>/
            top/
                train_001.jpg ... train_030.jpg
                test_001.jpg  ... test_005.jpg
            bottom/
                train_001.jpg ... train_030.jpg
                test_001.jpg  ... test_005.jpg

Run:
    python -m preprocessing.batch_processor --data dataset/raw --out processed --mode train
    python -m preprocessing.batch_processor --data dataset/raw --out processed --mode test
"""

import cv2
import json
import shutil
import traceback
import argparse
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.species_id.pipeline import run_pipeline
from preprocessing.config import (
    VIEWS, IMG_EXTS, CHECKPOINT_EVERY,
    QC_MIN_COVERAGE, QC_MAX_COVERAGE
)


# ── Folder helpers ────────────────────────────────────────────────────────────

def _is_test_image(img_path: Path) -> bool:
    """
    Test images are identified by a 'test_' prefix in their filename stem.
    e.g.  test_001.jpg  → True
          PXL_20260508_075950959.jpg → False

    Only the 5 test images per species/view need renaming to test_001..test_005.
    All other camera-named images are automatically treated as train.
    """
    return img_path.stem.lower().startswith("test_")


def _collect_images(data_root: Path,
                    mode: str = "train") -> list[tuple[Path, str, str]]:
    """
    Walk dataset/raw/<species>/<view>/ and return list of (img_path, species, view)
    filtered by mode.

    mode='train' → camera-named images (PXL_*, IMG_*, anything NOT starting with 'test_')
    mode='test'  → images whose filename starts with 'test_' (test_001..test_005)

    Only your 5 test images per species/view need renaming. Train images keep
    their original camera names (PXL_20260508_075950959.jpg etc.) unchanged.
    """
    if mode not in ("train", "test"):
        raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")

    tasks = []
    for sp_dir in sorted(data_root.iterdir()):
        if not sp_dir.is_dir():
            continue
        for view in VIEWS:
            vdir = sp_dir / view
            if not vdir.exists():
                continue
            imgs = []
            for ext in IMG_EXTS:
                imgs.extend(vdir.glob(ext))
            imgs = sorted(set(imgs))

            for img_p in imgs:
                is_test = _is_test_image(img_p)
                if mode == "test" and is_test:
                    tasks.append((img_p, sp_dir.name, view))
                elif mode == "train" and not is_test:
                    tasks.append((img_p, sp_dir.name, view))
    return tasks


def _make_output_dirs(out_root: Path) -> dict[str, Path]:
    dirs = {
        "features"    : out_root / "features",
        "diagnostics" : out_root / "diagnostics",
        "per_image"   : out_root / "diagnostics" / "per_image",
        "failures"    : out_root / "diagnostics" / "failures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _save_processed_images(img_masked: np.ndarray,
                            img_sharp: np.ndarray,
                            img_path: Path,
                            species: str,
                            view: str,
                            out_root: Path):
    """
    Save masked_raw (health branch input) and enhanced (CNN input) images.

    Folder layout:
        ../dataset/processed/<species>/<view>/masked_raw/<filename>
        ../dataset/processed/<species>/<view>/enhanced/<filename>
    """
    stem = img_path.stem
    for subdir, img in [("masked_raw", img_masked), ("enhanced", img_sharp)]:
        dest = out_root / species / view / subdir
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / f"{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(rows: list[dict], dirs: dict, idx: int):
    if not rows:
        return
    ckpt_path = dirs["features"] / f"checkpoint_{idx:05d}.csv"
    pd.DataFrame(rows).to_csv(ckpt_path, index=False)


def _load_checkpoint(dirs: dict) -> tuple[list[dict], set]:
    """
    Resume from latest checkpoint if present.
    Returns (existing_rows, processed_paths_set).
    """
    ckpts = sorted(dirs["features"].glob("checkpoint_*.csv"))
    if not ckpts:
        return [], set()
    df = pd.read_csv(ckpts[-1])
    rows  = df.to_dict("records")
    paths = set(df["image_path"].tolist())
    print(f"[RESUME] Loaded {len(rows)} rows from {ckpts[-1].name}")
    return rows, paths


# ── Main batch loop ───────────────────────────────────────────────────────────

def run_batch(data_root: Path,
              out_root: Path,
              mode: str = "train",
              save_images: bool = True,
              resume: bool = True) -> dict:
    """
    Process the dataset in either train or test mode.

    Parameters
    ----------
    data_root   : path to dataset/raw/
    out_root    : path to processed/ output folder
    mode        : 'train' processes camera-named images (PXL_*, IMG_*, etc.)
                  'test'  processes images prefixed with 'test_' (test_001..test_005)
                  In test mode: no augmentation, features extracted directly for evaluation.
    save_images : save masked_raw + enhanced image files
    resume      : skip images already in checkpoint (train mode only)

    Returns
    -------
    summary dict
    """
    dirs  = _make_output_dirs(out_root)
    tasks = _collect_images(data_root, mode=mode)

    if not tasks:
        print(f"[ERROR] No {mode} images found under {data_root}")
        print(f"  Reminder: test images must be named test_001.jpg ... test_005.jpg")
        print(f"  Train images keep their original camera names (PXL_*, IMG_*, etc.)")
        return {}

    print(f"[MODE: {mode.upper()}] Found {len(tasks)} images across "
          f"{len(set(t[1] for t in tasks))} species")

    # Resume only makes sense for train (test set is tiny, always re-run cleanly)
    do_resume = resume and (mode == "train")
    success_rows, done_paths = _load_checkpoint(dirs) if do_resume else ([], set())
    failure_rows = []

    n_success = len(success_rows)
    n_fail    = 0
    n_skip    = 0
    start_ts  = datetime.now().isoformat()

    # Progress bar
    with tqdm(total=len(tasks), unit="img", desc=f"Batch [{mode}]") as pbar:
        for i, (img_path, species, view) in enumerate(tasks):

            pbar.set_postfix(species=species[:12], view=view)

            # Skip already processed
            if str(img_path) in done_paths:
                n_skip += 1
                pbar.update(1)
                continue

            # ── Per-image try/except ───────────────────────────────────────
            try:
                feats, info = run_pipeline(img_path, species, view)

                diag_payload = {
                    "image_path" : str(img_path),
                    "species"    : species,
                    "view_side"  : view,
                    "qc_passed"  : info["qc_passed"],
                    "qc_reason"  : info["qc_reason"],
                    "mask_diag"  : info.get("mask_diag", {}),
                }

                if feats is None:
                    # QC fail
                    _record_failure(img_path, species, view,
                                    info["qc_reason"], "QC_FAIL",
                                    dirs, failure_rows)
                    n_fail += 1
                else:
                    success_rows.append(feats)
                    n_success += 1

                    # Save processed images
                    if save_images and info.get("img_masked") is not None:
                        _save_processed_images(
                            info["img_masked"],
                            info["img_sharp"],
                            img_path, species, view, out_root / "images"
                        )

                # Per-image JSON diagnostic
                json_path = dirs["per_image"] / f"{img_path.stem}_{view}.json"
                with open(json_path, "w") as jf:
                    json.dump(diag_payload, jf, indent=2)

            except Exception as exc:
                tb = traceback.format_exc()
                _record_failure(img_path, species, view,
                                str(exc), "EXCEPTION",
                                dirs, failure_rows, traceback_str=tb)
                n_fail += 1

            # Memory cleanup
            gc.collect()
            pbar.update(1)

            # Checkpoint
            if (i + 1) % CHECKPOINT_EVERY == 0:
                _save_checkpoint(success_rows, dirs, i + 1)
                tqdm.write(f"  [CHECKPOINT] {n_success} ok, {n_fail} fail at image {i+1}")

    # ── Final outputs ─────────────────────────────────────────────────────────
    METADATA_COLS = [
        "species",
        "view_side",
        "image_path",
        "mask_choice",
        "coverage_pct",
        "vein_coverage_pct",        # diagnostic from vein.py
        "vein_roi_scale",           # diagnostic from vein.py
    ]

    if success_rows:
        df_success = pd.DataFrame(success_rows)
        feature_cols = [c for c in df_success.columns if c not in METADATA_COLS]

        # Mode-specific output filenames keep train and test CSVs clearly separate
        full_csv = dirs["features"] / f"vedavision_features_{mode}.csv"
        clf_csv  = dirs["features"] / f"vedavision_features_{mode}_clf.csv"

        df_success.to_csv(full_csv, index=False)
        df_success[feature_cols + ["species"]].to_csv(clf_csv, index=False)

        print(f"\n✓ [{mode.upper()}] Features saved: {len(df_success)} rows × "
              f"{len(feature_cols)} feature cols")
        print(f"  Full CSV (with metadata) : {full_csv}")
        print(f"  Classifier CSV (no meta) : {clf_csv}")
        print(f"  Metadata cols excluded   : {METADATA_COLS}")

    if failure_rows:
        df_fail = pd.DataFrame(failure_rows)
        df_fail.to_csv(dirs["diagnostics"] / "failures.csv", index=False)
        print(f"✗ Failures logged: {len(df_fail)} rows → {dirs['diagnostics'] / 'failures.csv'}")

    summary = {
        "mode"          : mode,
        "run_start"     : start_ts,
        "run_end"       : datetime.now().isoformat(),
        "total_images"  : len(tasks),
        "n_success"     : n_success,
        "n_fail"        : n_fail,
        "n_skip"        : n_skip,
        "success_rate"  : round(n_success / max(len(tasks) - n_skip, 1) * 100, 2),
        "failures_by_species": _count_by_species(failure_rows),
    }
    with open(dirs["diagnostics"] / f"report_{mode}.json", "w") as jf:
        json.dump(summary, jf, indent=2)

    print(f"\n[{mode.upper()}] Batch complete — {n_success} ok, {n_fail} fail, {n_skip} skipped")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    return summary


# ── Failure helpers ───────────────────────────────────────────────────────────

def _record_failure(img_path: Path, species: str, view: str,
                    reason: str, fail_type: str,
                    dirs: dict, failure_rows: list,
                    traceback_str: str = ""):
    """
    Log a failure row and copy the failing image to diagnostics/failures/<species>/<view>/.
    Saving failed images separately fulfils the interim report requirement.
    """
    failure_rows.append({
        "image_path"  : str(img_path),
        "species"     : species,
        "view_side"   : view,
        "fail_type"   : fail_type,
        "reason"      : reason,
        "traceback"   : traceback_str,
        "timestamp"   : datetime.now().isoformat(),
    })

    # Copy failed image to failure folder (separate per-species folder)
    fail_dir = dirs["failures"] / species / view
    fail_dir.mkdir(parents=True, exist_ok=True)
    dest     = fail_dir / img_path.name
    if not dest.exists():
        try:
            shutil.copy2(str(img_path), str(dest))
        except Exception:
            pass  # Don't crash the batch over a copy failure


def _count_by_species(failure_rows: list) -> dict:
    counts = {}
    for row in failure_rows:
        sp = row["species"]
        counts[sp] = counts.get(sp, 0) + 1
    return counts


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VedaVision Batch Processor")
    parser.add_argument("--data",   default="dataset/raw",  help="Path to dataset/raw/")
    parser.add_argument("--out",    default="processed",    help="Output root directory")
    parser.add_argument("--mode",   default="train",        choices=["train", "test"],
                        help=(
                            "train: process camera-named images (PXL_*, IMG_*, etc.) "
                            "with augmentation. "
                            "test: process images prefixed with 'test_' (test_001..test_005) "
                            "without augmentation, for evaluation only."
                        ))
    parser.add_argument("--no-images", action="store_true", help="Skip saving image files")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoints)")
    args = parser.parse_args()

    run_batch(
        data_root   = Path(args.data),
        out_root    = Path(args.out),
        mode        = args.mode,
        save_images = not args.no_images,
        resume      = not args.no_resume,
    )
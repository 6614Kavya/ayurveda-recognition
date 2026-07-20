"""
VedaVision — Augmented Mask QC Tool (v2)
===========================================
Diagnostic-only script. Does NOT modify batch_processor.py or any
production pipeline file. Safe to run repeatedly / delete afterwards.

WHAT CHANGED FROM v1
----------------------
v1 always used the OLD approach internally (augment_raw_with_original +
run_pipeline with img_bgr_override) — meaning even after you implement the
mask-warping fix in batch_processor.py, running v1 would still show you
the OLD, still-vulnerable masking behaviour, since v1 never called the
new functions.

v2 adds a --mode flag so you can run the SAME source image through BOTH
approaches and directly compare:

    --mode old   -> augment_raw_with_original() + run_pipeline()
                    (re-masks each variant by colour — vulnerable to
                    shadow landing on the leaf)

    --mode new   -> augment_resized_with_mask_and_original() +
                    run_pipeline_from_resized()
                    (masks once on the clean image, then geometrically
                    warps that mask alongside each variant — shadow-immune)

Run the same image with both flags and compare the two output grids
side-by-side for your before/after evidence.

NEW: --force-shadow flag temporarily sets RandomShadow's p=1.0 for this
run only (production augmentation.py is untouched) so you don't have to
wait for a lucky 30% draw to see the shadow case. Revert is automatic —
it only patches the transform object created inside this script's own
process, never touches your actual augmentation.py file on disk.

Usage
-----
    # OLD behaviour (before fix) -- to see the vulnerable baseline
    python -m preprocessing.tools.check_augmented_masks \\
        --image dataset/raw/beli/top/PXL_....jpg --species beli --view top \\
        --mode old --force-shadow --seed 42 --out diagnostics/before

    # NEW behaviour (after fix) -- same image, same seed, same forced shadow
    python -m preprocessing.tools.check_augmented_masks \\
        --image dataset/raw/beli/top/PXL_....jpg --species beli --view top \\
        --mode new --force-shadow --seed 42 --out diagnostics/after

Compare diagnostics/before/*.jpg vs diagnostics/after/*.jpg on the exact
same shadow placement -- that's your apples-to-apples evidence.
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

import albumentations as A

from preprocessing.shared.resize import letterbox_resize
from preprocessing.shared.masking import select_mask, qc_check
from preprocessing.shared.augmentation import (
    augment_raw_with_original,
    N_AUGMENTATIONS,
)
from preprocessing.species_id.pipeline import run_pipeline
from preprocessing.config import TARGET_LONG

# Only needed for --mode new. Import lazily-safe: if you haven't added
# these two functions yet (from the earlier fix), --mode old still works
# standalone; --mode new will raise a clear ImportError telling you what's missing.
try:
    from preprocessing.shared.augmentation import augment_resized_with_mask_and_original
    from preprocessing.species_id.pipeline import run_pipeline_from_resized
    _NEW_MODE_AVAILABLE = True
except ImportError:
    _NEW_MODE_AVAILABLE = False


def _build_forced_shadow_transform_old():
    """Same 7-transform list as augmentation.py's _build_transform(), but
    with RandomShadow's p forced to 1.0. Used ONLY when --force-shadow is
    passed with --mode old. Does not touch the real augmentation.py file."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, fill=255, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), num_shadows_limit=(1, 2),
                        shadow_dimension=4, p=1.0),   # forced from 0.3 -> 1.0
    ])


def _build_forced_shadow_transform_new():
    """Same as augmentation_mask_aware_addition.py's transform, but with
    RandomShadow's p forced to 1.0. Used ONLY when --force-shadow is
    passed with --mode new."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, fill=255, fill_mask=0,
                  mask_interpolation=cv2.INTER_NEAREST, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.RandomShadow(shadow_roi=(0.0, 0.0, 1.0, 1.0), num_shadows_limit=(1, 2),
                        shadow_dimension=4, p=1.0),   # forced from 0.3 -> 1.0
    ])


def _mask_overlay(img_resized: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = img_resized.copy()
    red = np.zeros_like(img_resized)
    red[:, :, 2] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(img_resized, 1 - alpha, red, alpha, 0)[mask_bool]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return overlay


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def check_one_image_old(img_path: Path, species: str, view: str, out_dir: Path,
                          seed: int, force_shadow: bool):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[SKIP] Could not read {img_path}")
        return

    np.random.seed(seed)
    if force_shadow:
        transform = _build_forced_shadow_transform_old()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        variants = [img_bgr] + [
            cv2.cvtColor(transform(image=img_rgb)["image"], cv2.COLOR_RGB2BGR)
            for _ in range(N_AUGMENTATIONS)
        ]
    else:
        variants = augment_raw_with_original(img_bgr, n=N_AUGMENTATIONS)

    labels = ["original"] + [f"aug_{i:02d}" for i in range(1, N_AUGMENTATIONS + 1)]
    rows, diag_rows = [], []

    print(f"\n=== [OLD mode] {img_path.name} ({species}/{view}) — force_shadow={force_shadow} ===")
    print(f"{'variant':<10} {'qc_pass':<8} {'coverage%':<10} {'mask_choice':<8}")

    for lbl, variant_bgr in zip(labels, variants):
        feats, info = run_pipeline(img_path, species, view, img_bgr_override=variant_bgr)
        diag = info.get("mask_diag", {})
        qc_passed = info.get("qc_passed", False)
        coverage = diag.get("coverage_pct", float("nan"))
        mask_choice = info.get("mask_choice", "-")

        print(f"{lbl:<10} {str(qc_passed):<8} {coverage:<10.2f} {str(mask_choice):<8}")
        diag_rows.append({"variant": lbl, "qc_passed": qc_passed, "coverage_pct": coverage})

        img_resized = info.get("img_resized")
        mask_final = info.get("mask_final")
        masked = info.get("img_masked")
        if img_resized is None or mask_final is None:
            placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(placeholder, "NO MASK", (140, 260), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)
            rows.append(_label(placeholder, f"{lbl} [FAIL]"))
            continue

        overlay = _mask_overlay(img_resized, mask_final)
        masked = masked if masked is not None else np.zeros_like(img_resized)
        tag = f"{lbl}" + ("" if qc_passed else " [QC FAIL]")
        strip = np.hstack([_label(img_resized, "resized"),
                            _label(overlay, f"{tag} | cov={coverage:.1f}%"),
                            _label(masked, "masked")])
        rows.append(strip)

    grid = np.vstack(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_{species}_{view}_OLD_augmask_check.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[SAVED] {out_path}")
    return diag_rows


def check_one_image_new(img_path: Path, species: str, view: str, out_dir: Path,
                          seed: int, force_shadow: bool):
    if not _NEW_MODE_AVAILABLE:
        raise ImportError(
            "--mode new requires augment_resized_with_mask_and_original() in "
            "augmentation.py and run_pipeline_from_resized() in pipeline.py. "
            "Add those first (see earlier fix), or use --mode old for now."
        )

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[SKIP] Could not read {img_path}")
        return

    # Mask ONCE on the clean image
    img_resized_orig, _ = letterbox_resize(img_bgr, TARGET_LONG)
    mask_orig, mask_choice_orig, mask_diag_orig = select_mask(img_resized_orig)
    qc_passed_orig, qc_reason_orig = qc_check(mask_diag_orig)

    print(f"\n=== [NEW mode] {img_path.name} ({species}/{view}) — force_shadow={force_shadow} ===")
    if not qc_passed_orig:
        print(f"[FAIL] Original failed QC: {qc_reason_orig} — cannot proceed")
        return

    np.random.seed(seed)
    if force_shadow:
        transform = _build_forced_shadow_transform_new()
        img_rgb = cv2.cvtColor(img_resized_orig, cv2.COLOR_BGR2RGB)
        variant_pairs = [(img_resized_orig, mask_orig)]
        for _ in range(N_AUGMENTATIONS):
            out = transform(image=img_rgb, mask=mask_orig)
            variant_pairs.append((cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR), out["mask"]))
    else:
        variant_pairs = augment_resized_with_mask_and_original(
            img_resized_orig, mask_orig, n=N_AUGMENTATIONS
        )

    labels = ["original"] + [f"aug_{i:02d}" for i in range(1, N_AUGMENTATIONS + 1)]
    rows, diag_rows = [], []

    print(f"{'variant':<10} {'qc_pass':<8} {'coverage%':<10} {'mask_choice':<8}")

    for lbl, (img_resized_v, mask_v) in zip(labels, variant_pairs):
        feats, info = run_pipeline_from_resized(img_path, species, view, img_resized_v, mask_v)
        diag = info.get("mask_diag", {})
        qc_passed = info.get("qc_passed", False)
        coverage = diag.get("coverage_pct", float("nan"))
        mask_choice = info.get("mask_choice", "-")

        print(f"{lbl:<10} {str(qc_passed):<8} {coverage:<10.2f} {str(mask_choice):<8}")
        diag_rows.append({"variant": lbl, "qc_passed": qc_passed, "coverage_pct": coverage})

        overlay = _mask_overlay(img_resized_v, mask_v)
        masked = info.get("img_masked")
        masked = masked if masked is not None else np.zeros_like(img_resized_v)
        tag = f"{lbl}" + ("" if qc_passed else " [QC FAIL]")
        strip = np.hstack([_label(img_resized_v, "resized"),
                            _label(overlay, f"{tag} | cov={coverage:.1f}%"),
                            _label(masked, "masked")])
        rows.append(strip)

    grid = np.vstack(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_{species}_{view}_NEW_augmask_check.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[SAVED] {out_path}")

    covs = [r["coverage_pct"] for r in diag_rows if isinstance(r["coverage_pct"], (int, float))]
    if covs:
        spread = max(covs) - min(covs)
        print(f"[INFO] Coverage spread across 7 variants: {spread:.2f}pp "
              f"(should now be small/rounding-only even with shadow forced on)")
    return diag_rows


def main():
    ap = argparse.ArgumentParser(description="Visually QC masking on augmented image variants")
    ap.add_argument("--image", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--view", required=True, choices=["top", "bottom"])
    ap.add_argument("--n-images", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="diagnostics/aug_mask_check")
    ap.add_argument("--mode", choices=["old", "new"], default="new",
                     help="old = pre-fix behaviour (re-mask by colour per variant); "
                          "new = mask-warping fix (mask once, warp geometrically)")
    ap.add_argument("--force-shadow", action="store_true",
                     help="Force RandomShadow to p=1.0 for this run only, "
                          "so you don't have to wait for a random 30% draw")
    args = ap.parse_args()

    img_arg = Path(args.image)
    out_dir = Path(args.out)

    if img_arg.is_dir():
        candidates = sorted([p for p in img_arg.iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                              and not p.stem.lower().startswith("test_")])[:args.n_images]
    else:
        candidates = [img_arg]

    if not candidates:
        print(f"[ERROR] No images found at {img_arg}")
        return

    fn = check_one_image_old if args.mode == "old" else check_one_image_new
    for img_path in candidates:
        fn(img_path, args.species, args.view, out_dir, args.seed, args.force_shadow)


if __name__ == "__main__":
    main()
"""
VedaVision — Species-ID Preprocessing Pipeline
================================================
Orchestrates the full pipeline for ONE image → feature dict.

Usage (smoke test):
    python -m preprocessing.species_id.pipeline --image <path> --species <name> --view top

This module is also imported by batch_processor.py.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from preprocessing.shared.resize  import letterbox_resize
from preprocessing.shared.masking import select_mask, qc_check
from preprocessing.species_id.enhance import enhance_for_species_id
from feature_extraction.species_id.shape     import extract_shape_features
from feature_extraction.species_id.colour    import extract_colour_features
from feature_extraction.species_id.texture   import extract_texture_features
from feature_extraction.species_id.vein      import extract_vein_features
from feature_extraction.species_id.whole_leaf import extract_whole_leaf_features
from preprocessing.config import TARGET_LONG


def run_pipeline(img_path: str | Path,
                 species: str,
                 view_side: str
                 ) -> tuple[Optional[dict], Optional[dict]]:
    """
    Run the full species-ID preprocessing pipeline on a single image.

    Parameters
    ----------
    img_path  : path to the raw input image
    species   : species label string (used in CSV metadata)
    view_side : "top" | "bottom"

    Returns
    -------
    feats : flat feature dict ready for CSV row, or None on failure
    info  : dict with intermediate arrays and diagnostics (for visualisation/QC)
            keys: img_orig, img_resized, mask_final, mask_choice, mask_diag,
                  img_masked, img_sharp, qc_passed, qc_reason
    """
    img_path = Path(img_path)

    # ── Load ──────────────────────────────────────────────────────────────────
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        return None, {"qc_passed": False, "qc_reason": "cv2.imread failed"}

    # ── Step 1: Letterbox resize ───────────────────────────────────────────────
    img_resized, resize_meta = letterbox_resize(img_orig, TARGET_LONG)

    # ── Step 2: Background removal (shadow-aware v4.1) ─────────────────────────
    mask_final, mask_choice, mask_diag = select_mask(img_resized)

    # ── QC check ──────────────────────────────────────────────────────────────
    qc_passed, qc_reason = qc_check(mask_diag)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask_final)

    info = {
        "img_orig"    : img_orig,
        "img_resized" : img_resized,
        "mask_final"  : mask_final,
        "mask_choice" : mask_choice,
        "mask_diag"   : mask_diag,
        "img_masked"  : img_masked,
        "img_sharp"   : None,
        "qc_passed"   : qc_passed,
        "qc_reason"   : qc_reason,
    }

    if not qc_passed:
        return None, info

    # ── Step 3: Enhancement (species-ID branch only) ───────────────────────────
    img_sharp = enhance_for_species_id(img_masked, mask_final)
    info["img_sharp"] = img_sharp

    # ── Step 4: Feature extraction ────────────────────────────────────────────
    shape_f          = extract_shape_features(mask_final)
    colour_f         = extract_colour_features(img_resized, mask_final)   # RAW image
    texture_f        = extract_texture_features(img_sharp, mask_final)    # Enhanced
    vein_f, skel, _  = extract_vein_features(img_sharp, mask_final)
    whole_f          = extract_whole_leaf_features(mask_final)

    info["vein_skel"] = skel

    # ── Assemble flat feature dict ────────────────────────────────────────────
    feats = {}
    feats.update({f"shape_{k}":   v for k, v in shape_f.items()})
    feats.update({f"colour_{k}":  v for k, v in colour_f.items()})
    feats.update({f"texture_{k}": v for k, v in texture_f.items()})
    feats.update({f"vein_{k}":    v for k, v in vein_f.items()})
    feats.update({f"whole_{k}":   v for k, v in whole_f.items()})

    # Metadata columns (not features — excluded from classifier input)
    feats["species"]     = species
    feats["view_side"]   = view_side
    feats["image_path"]  = str(img_path)
    feats["mask_choice"] = mask_choice
    feats["coverage_pct"] = mask_diag["coverage_pct"]

    return feats, info


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="VedaVision — species-ID pipeline smoke test")
    parser.add_argument("--image",   required=True, help="Path to input image")
    parser.add_argument("--species", required=True, help="Species name")
    parser.add_argument("--view",    required=True, choices=["top", "bottom"])
    args = parser.parse_args()

    feats, info = run_pipeline(args.image, args.species, args.view)

    if feats is None:
        print(f"[FAIL] {info['qc_reason']}")
    else:
        n = sum(1 for k in feats if k not in ("species", "view_side", "image_path",
                                               "mask_choice", "coverage_pct"))
        print(f"[OK] {n} features extracted")
        print(f"  mask_choice  : {info['mask_choice']}")
        print(f"  coverage_pct : {info['mask_diag']['coverage_pct']:.1f}%")
        print(f"  seed_relaxed : {info['mask_diag']['seed_relaxed']}")

"""
VedaVision — Species-ID Preprocessing Pipeline
================================================
Orchestrates the full pipeline for ONE image → feature dict.

Usage (smoke test):
    python -m preprocessing.species_id.pipeline --image <path> --species <name> --view top

This module is also imported by batch_processor.py.

Masking change (this session) — wired to select_mask_guarded()
-----------------------------------------------------------------
Previously this pipeline ALWAYS ran select_mask() on an unconditionally
illumination-flattened copy of the image:

    img_flattened = flatten_illumination(img_resized)
    mask_final, mask_choice, mask_diag = select_mask(img_flattened)

That has two problems, both fixed by routing through mask_guard's
select_mask_guarded() instead:
  1. It never compared against the unflattened baseline, so any image
     where flattening made things worse (confirmed on siymbala: bleed
     3.14% -> 6.69% on one image from a tight->loose Stage-5 flip) had no
     fallback — the pipeline just accepted the worse mask.
  2. Flattening was computed on every single image regardless of whether
     baseline was already clean, which is wasted compute at batch scale.

select_mask_guarded() computes baseline always, only computes the
flattened variant when baseline bleed is above a floor
(skip_flatten_if_baseline_below), picks whichever has lower shadow-bleed,
and — critically — refuses to accept the flattened variant if doing so
would drop foreground area by more than max_coverage_drop_ratio (default
8%), since shadow-bleed is a ratio that's trivially gameable by shrinking
the mask (confirmed on ranawara_bottom_PXL_20260506_054942264: flattening
dropped an entire leaflet, 29% of baseline area, and still scored a lower
bleed fraction than baseline). See mask_guard.py for full detail.

mask_diag now carries the winning variant's normal select_mask() fields
(coverage_pct, seed_relaxed, n_final_components, rachis_pct, ...) PLUS
guard_* audit fields (guard_variant_used, guard_baseline_bleed,
guard_flattened_bleed, guard_coverage_drop_ratio,
guard_leaflet_loss_rejected, ...). qc_check() only reads the original
fields, so it needs no changes. return_all is left at its default
(False) here — the extra baseline/flattened mask arrays it would add to
diag are QC-tooling-only and have no reason to be carried through the
training/inference pipeline or serialized to the features CSV.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from preprocessing.shared.resize      import letterbox_resize
from preprocessing.shared.masking     import qc_check
from preprocessing.shared.mask_guard  import select_mask_guarded
from preprocessing.species_id.enhance import enhance_for_species_id
from feature_extraction.species_id.shape     import extract_shape_features
from feature_extraction.species_id.colour    import extract_colour_features
from feature_extraction.species_id.texture   import extract_texture_features
from feature_extraction.species_id.vein      import extract_vein_features
from feature_extraction.species_id.whole_leaf import extract_whole_leaf_features
from preprocessing.config import TARGET_LONG


def _namespace_features(prefix: str, features: dict) -> dict:
    """Prefix feature keys once, preserving keys that are already namespaced."""
    namespaced = {}
    for key, value in features.items():
        if key.startswith(f"{prefix}_"):
            namespaced[key] = value
        else:
            namespaced[f"{prefix}_{key}"] = value
    return namespaced


def run_pipeline(img_path: str | Path,
                 species: str,
                 view_side: str,
                 img_bgr_override: "Optional[np.ndarray]" = None,
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
    img_orig = img_bgr_override if img_bgr_override is not None \
              else cv2.imread(str(img_path))
    if img_orig is None:
        return None, {"qc_passed": False, "qc_reason": "cv2.imread failed"}

    # ── Step 1: Letterbox resize ───────────────────────────────────────────────
    img_resized, resize_meta = letterbox_resize(img_orig, TARGET_LONG)

    # ── Step 2: Background removal (guarded: baseline vs illumination-
    #    flattened, whichever has lower shadow-bleed WITHOUT losing a
    #    leaflet's worth of foreground area — see module docstring) ───────────
    mask_final, mask_choice, mask_diag = select_mask_guarded(img_resized)

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
    feats.update(_namespace_features("shape", shape_f))
    feats.update(_namespace_features("colour", colour_f))
    feats.update(_namespace_features("texture", texture_f))
    feats.update(_namespace_features("vein", vein_f))
    feats.update(_namespace_features("whole", whole_f))

    # Metadata columns (not features — excluded from classifier input)
    # vein_coverage_pct and vein_roi_scale are already in feats via
    # the f"vein_{k}" loop above (they come from extract_vein_features).
    # Do NOT re-assign them here — that would overwrite with mask_diag
    # keys that do not exist, causing a KeyError at runtime.
    feats["species"]      = species
    feats["view_side"]    = view_side
    feats["image_path"]   = str(img_path)
    feats["mask_choice"]  = mask_choice
    feats["coverage_pct"] = mask_diag["coverage_pct"]

    # Guard audit trail — cheap to carry, useful later if you need to ask
    # "which images used the flattened variant" or "which images tripped
    # the leaflet-loss veto" without re-running the whole batch. Excluded
    # from classifier input the same way image_path/mask_choice are (all
    # non-numeric / non-feature columns should already be dropped before
    # training — same convention as the existing metadata columns above).
    feats["guard_variant_used"]        = mask_diag.get("guard_variant_used")
    feats["guard_leaflet_loss_rejected"] = mask_diag.get("guard_leaflet_loss_rejected")

    return feats, info


def run_pipeline_from_resized(img_path: str | Path,
                               species: str,
                               view_side: str,
                               img_resized: "np.ndarray",
                               mask_final: "np.ndarray",
                               ) -> tuple[Optional[dict], Optional[dict]]:
    """
    Run enhancement + feature extraction on an already-resized image with
    an already-known mask (skips letterbox resize + select_mask stages).

    Use this for augmented variants where the mask has already been
    computed on the CLEAN original and geometrically warped alongside the
    augmented image via augment_resized_with_mask() -- never re-derive the
    mask by colour-thresholding a photometrically-augmented (shadowed /
    brightness-shifted / noisy) image, since that reintroduces the exact
    shadow-vs-leaf ambiguity select_mask_guarded() is designed to resolve
    on the clean original. (No guard call happens here at all, by design
    — the mask for this row was already decided upstream, once, on the
    unaugmented photo.)

    Parameters mirror run_pipeline()'s return contract exactly, so
    downstream code (batch_processor.py) doesn't need to branch on which
    variant produced a given feature row.
    """
    img_path = Path(img_path)
    img_area = img_resized.shape[0] * img_resized.shape[1]

    # Lightweight QC recompute -- coverage should barely move from the
    # original's already-passed QC (see empirical test: <0.2pp typical
    # drift from flip/rotate rounding only), but keep the check as a
    # cheap safety net against any edge case (e.g. severe rotate cropping
    # a corner of a leaf that was already close to the frame edge).
    coverage_pct = round(float((mask_final > 0).sum()) / img_area * 100, 2)
    mask_diag = {"coverage_pct": coverage_pct}
    qc_passed, qc_reason = qc_check(mask_diag)

    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask_final)

    info = {
        "img_orig"    : img_resized,   # no separate raw copy at this stage
        "img_resized" : img_resized,
        "mask_final"  : mask_final,
        "mask_choice" : "warped",      # flags this row came from mask-warping, not fresh select_mask()
        "mask_diag"   : mask_diag,
        "img_masked"  : img_masked,
        "img_sharp"   : None,
        "qc_passed"   : qc_passed,
        "qc_reason"   : qc_reason,
    }

    if not qc_passed:
        return None, info

    img_sharp = enhance_for_species_id(img_masked, mask_final)
    info["img_sharp"] = img_sharp

    shape_f          = extract_shape_features(mask_final)
    colour_f         = extract_colour_features(img_resized, mask_final)
    texture_f        = extract_texture_features(img_sharp, mask_final)
    vein_f, skel, _  = extract_vein_features(img_sharp, mask_final)
    whole_f          = extract_whole_leaf_features(mask_final)

    info["vein_skel"] = skel

    feats = {}
    feats.update(_namespace_features("shape", shape_f))
    feats.update(_namespace_features("colour", colour_f))
    feats.update(_namespace_features("texture", texture_f))
    feats.update(_namespace_features("vein", vein_f))
    feats.update(_namespace_features("whole", whole_f))

    feats["species"]      = species
    feats["view_side"]    = view_side
    feats["image_path"]   = str(img_path)
    feats["mask_choice"]  = "warped"
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
                                               "mask_choice", "coverage_pct",
                                               "vein_coverage_pct", "vein_roi_scale",
                                               "guard_variant_used",
                                               "guard_leaflet_loss_rejected"))
        print(f"[OK] {n} features extracted")
        print(f"  mask_choice          : {info['mask_choice']}")
        print(f"  coverage_pct         : {info['mask_diag']['coverage_pct']:.1f}%")
        print(f"  seed_relaxed         : {info['mask_diag']['seed_relaxed']}")
        print(f"  guard_variant_used   : {info['mask_diag'].get('guard_variant_used')}")
        print(f"  guard_baseline_bleed : {info['mask_diag'].get('guard_baseline_bleed')}")
        print(f"  guard_flattened_bleed: {info['mask_diag'].get('guard_flattened_bleed')}")
        print(f"  leaflet_loss_rejected: {info['mask_diag'].get('guard_leaflet_loss_rejected')}")
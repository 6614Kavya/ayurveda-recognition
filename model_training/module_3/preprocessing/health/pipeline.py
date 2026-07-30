"""
Health-assessment single-image pipeline orchestrator.

Mirrors species_id/pipeline.py's structure but:
  - never calls enhance.py (health branch must stay on raw, un-enhanced colour)
  - reads mask_before_holefill AND rachis_mask out of masking.select_mask()'s
    diag dict so holes.py/scar.py and boundary.py/scar.py (respectively)
    can use them
  - computes the full feature row: boundary_* + hole_* + colour_* + scar_*
    + miner_trail_* + texture_h_* + ldsi_* (severity_index.py)

--- WIRING CHANGE (this session) ---
5. extract_deformation_features() (deformation.py) added as a seventh
   feature group -- specular-highlight fragmentation, width-profile
   roughness, and luminance spread, targeting leaf curling/non-planarity,
   a symptom none of the other six groups measure at all. Added directly
   to health_index.py's SUBSCORE_RAW_COLUMNS as new candidates (same
   treatment texture_health.py's features got when first added) -- not
   yet validated against real severity labels, so don't assume any of the
   three sub-features works before checking per_subscore_correlation()
   and the binary-target Ridge weights on your own data.

--- WIRING CHANGE (this session) ---
4. extract_texture_health_features() (texture_health.py) added as a sixth
   feature group -- GLCM (shadow/median-fill robust, mirroring
   species_id/texture.py) + LBP, computed on masked_raw. Added to close
   the biggest remaining feature-bank gap: none of boundary/hole/colour/
   scar/miner directly measure surface texture, and those five plateaued
   around Spearman rho ~0.19-0.23 against severity order. NOT yet fed
   into compute_ldsi()/severity_index.py's sub-score aggregation -- added
   directly to health_index.py's SUBSCORE_RAW_COLUMNS instead, same
   treatment as the raw colour_pct_* columns, so Ridge can weight it on
   its own merits rather than being pre-blended into an existing sub-score.

--- WIRING CHANGES (this session) ---
1. rachis_mask is now pulled from diag["rachis_mask"] (added to masking.py's
   diag dict this session -- it was computed all along but only ever
   exposed as a pixel count, rachis_px, never as the actual mask array) and
   passed into extract_boundary_features() and extract_scar_features() so
   their rachis-proximity gating (see those modules' docstrings) actually
   activates. Without this, both modules silently fall back to
   rachis_mask=None and behave exactly as before -- so if this wiring is
   ever accidentally dropped, nothing crashes, but the leaflet-junction
   false-positive bug comes right back. Don't let that regress silently.
2. extract_miner_trail_features() added as a fifth feature group, feeding
   into compute_ldsi(..., miner_feats=...) as the new ldsi_miner_sub.
3. run_health_pipeline_from_resized() gained a required rachis_mask
   parameter (4th positional) so augmented rows get the same gating as
   originals -- see augmentation.py's matching update, which now carries
   rachis_mask through the same geometric warp as mask_final/
   mask_before_holefill.

Open issue carried over from memory: augmented rows produced via
run_pipeline_from_resized() (species-ID branch) still can't access
mask_before_holefill on THAT branch -- unrelated to this file, which is
health-only and now has both masks + rachis_mask flowing through
correctly on both the original and augmented paths.

ADAPT THE IMPORT PATHS below to match your actual module locations if
they differ (these assume the layout documented in VedaVision memory).
"""
import cv2
import numpy as np

from preprocessing.shared.resize import letterbox_resize
from preprocessing.shared.masking import select_mask, qc_check

from feature_extraction.health.boundary import extract_boundary_features
from feature_extraction.health.holes import extract_hole_features
from feature_extraction.health.colour_health import extract_colour_health_features
from feature_extraction.health.scar import extract_scar_features
from feature_extraction.health.miner_trail import extract_miner_trail_features
from feature_extraction.health.texture_health import extract_texture_health_features
from feature_extraction.health.deformation import extract_deformation_features
from feature_extraction.health.spots import extract_spot_features
from feature_extraction.health.severity_index import compute_ldsi


def _extract_all_features(masked_raw, mask_final, mask_before_holefill, rachis_mask):
    """Shared by both entry points below so the two paths can't drift apart."""
    boundary_feats = extract_boundary_features(mask_final, rachis_mask=rachis_mask)
    hole_feats = extract_hole_features(mask_final, mask_before_holefill)
    colour_feats = extract_colour_health_features(masked_raw, mask_final)
    scar_feats = extract_scar_features(masked_raw, mask_final, mask_before_holefill, rachis_mask=rachis_mask)
    miner_feats = extract_miner_trail_features(masked_raw, mask_final)
    texture_feats = extract_texture_health_features(masked_raw, mask_final)
    deform_feats = extract_deformation_features(masked_raw, mask_final, rachis_mask=rachis_mask)
    # WIRING CHANGE (this session): extract_spot_features() (spots.py)
    # added as an eighth feature group -- discrete lesion/spot count,
    # size distribution, and density, targeting the "dilution" gap where
    # colour_pct_* stays flat because scattered small lesions barely move
    # a whole-leaf percentage (see diagnose_feature_gaps.py's per-species
    # effect-size report). NOT yet fed into compute_ldsi()'s sub-score
    # aggregation -- same treatment texture_health.py and deformation.py
    # got when first added: goes straight into health_index.py's
    # SUBSCORE_RAW_COLUMNS as new candidates so the binary-target Ridge
    # fit weighs it on its own merits, rather than being pre-blended into
    # an existing sub-score before it's been validated.
    spot_feats = extract_spot_features(
        masked_raw, mask_final,
        mask_before_holefill=mask_before_holefill, rachis_mask=rachis_mask,
    )
    ldsi_feats = compute_ldsi(boundary_feats, hole_feats, colour_feats, scar_feats, miner_feats=miner_feats)

    row = {}
    row.update(boundary_feats)
    row.update(hole_feats)
    row.update(colour_feats)
    row.update(scar_feats)
    row.update(miner_feats)
    row.update(texture_feats)
    row.update(deform_feats)
    row.update(spot_feats)
    row.update(ldsi_feats)
    return row


def run_health_pipeline(img_bgr_raw: np.ndarray, image_path: str = "") -> dict:
    """
    Full pipeline for one raw image (top OR bottom view of one leaf), from
    raw camera frame to a flat feature-row dict ready for CSV export /
    the health-branch classifier.

    Returns a dict. On QC reject, returns a row with qc_pass=False and no
    feature columns, rather than raising, so a batch run can log the
    failure and continue instead of crashing mid-batch.
    """
    resized, _ = letterbox_resize(img_bgr_raw)

    mask_final, mask_choice, diag = select_mask(resized)
    mask_before_holefill = diag.get("mask_before_holefill")  # None for paths that don't expose it
    rachis_mask = diag.get("rachis_mask")                    # None for paths that don't expose it

    qc_ok, qc_reason = qc_check(diag)
    if not qc_ok:
        return {"image_path": image_path, "qc_pass": False, "qc_reason": qc_reason}

    # health branch: masked_raw only, NO enhancement, ever
    masked_raw = cv2.bitwise_and(resized, resized, mask=mask_final.astype(np.uint8))

    row = {"image_path": image_path, "qc_pass": True, "mask_choice": mask_choice}
    row.update(_extract_all_features(masked_raw, mask_final, mask_before_holefill, rachis_mask))
    return row


def run_health_pipeline_from_resized(img_resized: np.ndarray,
                                      mask_final: np.ndarray,
                                      mask_before_holefill: np.ndarray,
                                      rachis_mask: np.ndarray,
                                      image_path: str = "") -> dict:
    """
    Health pipeline for an already-resized image with an already-known
    set of masks (all geometrically warped together upstream by
    augment_health_resized_with_masks()). Mirrors run_health_pipeline()'s
    return contract exactly.

    rachis_mask may be None (e.g. the photometric-comparison path in
    check_augmentation_safety.py, which doesn't carry it) -- boundary.py/
    scar.py both handle rachis_mask=None by falling back to their
    pre-gating behaviour, same as before this session's fix.

    Never re-derives masks by colour-thresholding here -- geo-only
    augmentation means there's no shadow/brightness contamination to
    re-mask against in the first place, but the point still stands:
    the mask for this row was decided once, upstream, on the clean image.
    """
    qc_ok, qc_reason = qc_check({"coverage_pct":
                                  round(float((mask_final > 0).sum()) /
                                        (mask_final.shape[0] * mask_final.shape[1]) * 100, 2)})
    if not qc_ok:
        return {"image_path": image_path, "qc_pass": False, "qc_reason": qc_reason}

    masked_raw = cv2.bitwise_and(img_resized, img_resized, mask=mask_final.astype(np.uint8))

    row = {"image_path": image_path, "qc_pass": True, "mask_choice": "warped"}
    row.update(_extract_all_features(masked_raw, mask_final, mask_before_holefill, rachis_mask))
    return row
import cv2
import numpy as np
from typing import Optional

from app.module3_compound_leaves.preprocessing.shared.resize  import letterbox_resize
from app.module3_compound_leaves.preprocessing.shared.masking import select_mask, qc_check

from app.module3_compound_leaves.feature_extraction.health.boundary       import extract_boundary_features
from app.module3_compound_leaves.feature_extraction.health.holes         import extract_hole_features
from app.module3_compound_leaves.feature_extraction.health.colour_health import extract_colour_health_features
from app.module3_compound_leaves.feature_extraction.health.scar          import extract_scar_features
from app.module3_compound_leaves.feature_extraction.health.miner_trail   import extract_miner_trail_features
from app.module3_compound_leaves.feature_extraction.health.texture_health import extract_texture_health_features
from app.module3_compound_leaves.feature_extraction.health.deformation   import extract_deformation_features
from app.module3_compound_leaves.feature_extraction.health.spots         import extract_spot_features
from app.module3_compound_leaves.feature_extraction.health.severity_index import compute_ldsi


def _extract_all_health_features(masked_raw, mask_final, mask_before_holefill, rachis_mask) -> dict:
    """Mirrors preprocessing/health/pipeline.py's _extract_all_features()
    exactly — same 8 feature groups + LDSI, same source image
    (masked_raw, NEVER enhanced — the health branch's colour signal is
    load-bearing and enhancement corrupts it, see project memory)."""
    boundary_feats = extract_boundary_features(mask_final, rachis_mask=rachis_mask)
    hole_feats = extract_hole_features(mask_final, mask_before_holefill)
    colour_feats = extract_colour_health_features(masked_raw, mask_final)
    scar_feats = extract_scar_features(masked_raw, mask_final, mask_before_holefill, rachis_mask=rachis_mask)
    miner_feats = extract_miner_trail_features(masked_raw, mask_final)
    texture_feats = extract_texture_health_features(masked_raw, mask_final)
    deform_feats = extract_deformation_features(masked_raw, mask_final, rachis_mask=rachis_mask)
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


def extract_health_features(img_bgr: np.ndarray) -> tuple[Optional[dict], dict]:
    """
    Full health-branch pipeline for ONE view (top OR bottom) of one leaf.

    Note this deliberately uses select_mask() directly (the baseline
    masking pipeline), NOT select_mask_guarded() — the species-ID branch
    uses the guarded/illumination-flattened variant, but the health
    branch never did (see preprocessing/health/pipeline.py); keeping
    that distinction here rather than "harmonising" it, since it would
    silently change every health feature value relative to what the
    model was trained on.

    Returns
    -------
    feats : flat dict of raw health features (unprefixed — top_/bottom_/
            worst_ fusion happens one level up, in predictor.py, via
            models.health.classifier.fuse_top_bottom), or None on QC fail
    info  : dict with mask_choice, coverage_pct, qc_passed, qc_reason
    """
    if img_bgr is None:
        return None, {"qc_passed": False, "qc_reason": "empty image"}

    img_resized, _resize_meta = letterbox_resize(img_bgr)

    mask_final, mask_choice, diag = select_mask(img_resized)
    mask_before_holefill = diag.get("mask_before_holefill")
    rachis_mask = diag.get("rachis_mask")

    qc_passed, qc_reason = qc_check(diag)
    info = {
        "mask_choice": mask_choice,
        "coverage_pct": diag.get("coverage_pct"),
        "qc_passed": qc_passed,
        "qc_reason": qc_reason,
    }
    if not qc_passed:
        return None, info

    # Health branch: masked_raw only, NEVER enhanced.
    masked_raw = cv2.bitwise_and(img_resized, img_resized, mask=mask_final.astype(np.uint8))

    feats = _extract_all_health_features(masked_raw, mask_final, mask_before_holefill, rachis_mask)
    return feats, info

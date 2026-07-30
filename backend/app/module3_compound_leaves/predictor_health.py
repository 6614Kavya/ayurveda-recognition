from typing import Optional

import numpy as np
import pandas as pd

from app.module3_compound_leaves.feature_pipeline import decode_image
from app.module3_compound_leaves.feature_pipeline_health import extract_health_features
from app.module3_compound_leaves.predictor import predict_species

from app.module3_compound_leaves.model_loader_health import get_health_index_model, get_stage1_model

from models.health.classifier import fuse_top_bottom
from models.health.model_training import NON_FEATURE_COLS
from models.health.train_stage1_binary import add_species_relative_features, _finalize_X


class InvalidImageError(Exception):
    """Raised when either uploaded image can't be decoded."""
    def __init__(self, which: str):
        self.which = which
        super().__init__(f"Could not decode {which} image — check file format")


class LeafNotDetectedError(Exception):
    """Raised when either view fails QC / leaf detection."""
    def __init__(self, which: str, reason: str):
        self.which = which
        self.reason = reason
        super().__init__(f"{which}: {reason}")


class HealthFeatureMismatchError(Exception):
    """Raised when the fused feature row is missing columns the Stage-1
    model or the Health Index model were trained on."""
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Missing {len(missing)} expected feature column(s): {missing[:5]}...")


# Mirrors predict_health.py's _META_KEYS — metadata keys that can appear
# in a raw feature dict but are never features to fuse.
_META_KEYS = {"species"} | NON_FEATURE_COLS


def assess_leaf_health(top_image_bytes: bytes, bottom_image_bytes: bytes,
                        species: Optional[str] = None) -> dict:
    """
    Full health-branch inference: raw top + bottom image bytes -> a
    healthy/unhealthy decision (Stage 1) plus an explainable 0-100
    Health Index score with a per-symptom breakdown.

    This intentionally mirrors models/health/predict_health.py's
    assess_leaf() exactly (same fusion-once, score-twice structure,
    same species-relative feature construction) — see that file's
    docstring for the full design rationale. The only behavioural
    addition is: `species` is optional here. If not supplied, it is
    auto-predicted from the top-view image using the already-loaded
    species-ID model, since the health branch's z-scored features are
    species-relative and cannot be computed without a species label.

    Returns
    -------
    dict with keys: species, decision, decision_confidence,
    health_value (0-100, 100=healthiest), severity_score_raw,
    breakdown (per-subscore % contribution to deviation),
    view_diagnostics (per-view mask_choice/coverage_pct, for debugging
    a QC-borderline upload).
    """
    top_bgr = decode_image(top_image_bytes)
    if top_bgr is None:
        raise InvalidImageError("top")
    bottom_bgr = decode_image(bottom_image_bytes)
    if bottom_bgr is None:
        raise InvalidImageError("bottom")

    top_feats, top_info = extract_health_features(top_bgr)
    if top_feats is None:
        raise LeafNotDetectedError("top", top_info.get("qc_reason", "leaf not detected"))
    bottom_feats, bottom_info = extract_health_features(bottom_bgr)
    if bottom_feats is None:
        raise LeafNotDetectedError("bottom", bottom_info.get("qc_reason", "leaf not detected"))

    if species is None:
        species_result = predict_species(top_image_bytes)
        species = species_result["species"]

    top_feats["species"] = species
    bottom_feats["species"] = species

    # Fuse ONCE over the full feature set (worst-side-wins), then score
    # with both models against that same fused row — matches
    # predict_health.py's assess_leaf() exactly. Do NOT pre-drop
    # Stage-1's dead_features here: the Health Index still needs two raw
    # columns Stage-1 drops (worst_ldsi_miner_sub,
    # worst_deform_luminance_std) — see predict_health.py's comment.
    feature_keys = sorted((set(top_feats.keys()) | set(bottom_feats.keys())) - _META_KEYS)
    fused = fuse_top_bottom(top_feats, bottom_feats, feature_keys)
    fused["species"] = species
    row_df = pd.DataFrame([fused])

    # --- Health Index (explainability) ---
    index_bundle = get_health_index_model()
    missing_subscores = [c for c in index_bundle.subscore_columns if c not in fused]
    if missing_subscores:
        raise HealthFeatureMismatchError(missing_subscores)
    severity_score = float(index_bundle.model.score(row_df)[0])
    breakdown = index_bundle.model.score_breakdown(row_df.iloc[0])

    # --- Stage 1 (the actual healthy/unhealthy decision) ---
    stage1_bundle = get_stage1_model()
    z_feature_cols = stage1_bundle.z_feature_cols

    stage1_row = row_df.copy()
    stage1_row["level"] = "unknown"  # placeholder _finalize_X needs something to drop; never read
    stage1_row = add_species_relative_features(stage1_row, z_feature_cols, stage1_bundle.species_baselines)
    X_stage1 = _finalize_X(stage1_row)
    # Only a genuinely unseen species dummy should ever be fill_value=0
    # here — every real feature/z-column was just computed above.
    X_stage1 = X_stage1.reindex(columns=stage1_bundle.feature_columns, fill_value=0.0)

    proba = stage1_bundle.model.predict_proba(X_stage1.values)
    classes = list(stage1_bundle.model.classes_)
    unhealthy_col = classes.index("unhealthy")
    p_unhealthy = float(proba[0, unhealthy_col])
    decision = "unhealthy" if p_unhealthy >= stage1_bundle.threshold else "healthy"
    decision_confidence = p_unhealthy if decision == "unhealthy" else (1.0 - p_unhealthy)

    return {
        "species": species,
        "decision": decision,
        "decision_confidence": round(decision_confidence, 3),
        "health_value": round(100.0 - severity_score, 2),
        "severity_score_raw": round(severity_score, 2),
        "breakdown": breakdown,
        "view_diagnostics": {
            "top": {"mask_choice": top_info.get("mask_choice"), "coverage_pct": top_info.get("coverage_pct")},
            "bottom": {"mask_choice": bottom_info.get("mask_choice"), "coverage_pct": bottom_info.get("coverage_pct")},
        },
    }

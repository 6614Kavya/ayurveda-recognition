from typing import Optional

import numpy as np
import pandas as pd

from app.module3_compound_leaves.feature_pipeline import decode_image
from app.module3_compound_leaves.feature_pipeline_health import extract_health_features
from app.module3_compound_leaves.predictor import predict_species
from app.module3_compound_leaves.predictor import InvalidImageError as SpeciesInvalidImageError
from app.module3_compound_leaves.predictor import LeafNotDetectedError as SpeciesLeafNotDetectedError
from app.module3_compound_leaves.predictor import FeatureMismatchError as SpeciesFeatureMismatchError

from app.module3_compound_leaves.model_loader_health import get_health_index_model, get_stage1_model
from app.module3_compound_leaves.species_metadata import SPECIES_METADATA

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


class InvalidSpeciesError(Exception):
    """Raised when an explicitly-supplied `species` isn't one of the
    known codes — e.g. a Swagger "Try it out" call left on its default
    placeholder text ("string") instead of a real code or an empty
    value. Silently accepting an unrecognized code would feed it
    straight into the species-relative feature lookup as if it were
    real, so this is rejected rather than passed through."""
    def __init__(self, species: str):
        self.species = species
        valid = ", ".join(sorted(SPECIES_METADATA.keys()))
        super().__init__(
            f"'{species}' is not a recognized species code. Either omit "
            f"the `species` field to auto-detect it from the images, or "
            f"supply one of: {valid}"
        )


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
    addition is: `species` is optional here. The public HTTP endpoint
    (router_health.py) never accepts a species from the caller — it's
    always auto-predicted using the already-loaded species-ID model,
    tried against the top-view image first and falling back to the
    bottom-view image if the top view fails species-ID's own QC/feature
    checks (a different, stricter pipeline than the health branch's —
    see the inline comment below). This parameter exists only so
    internal scripts/tests can pin a known species and skip that step;
    it is validated against the known code list either way. The health
    branch's z-scored features are species-relative and cannot be
    computed without a species label, so if species-ID fails on both
    views, the whole assessment fails with a clear error rather than
    proceeding with a bad or missing species.

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

    # An explicitly empty/whitespace value (some clients send "" rather
    # than omitting the field) is treated the same as not supplying one.
    if species is not None and species.strip() == "":
        species = None

    if species is None:
        # Species-ID uses its own masking/QC pipeline (select_mask_guarded),
        # which is deliberately different from the health branch's
        # (select_mask) — see feature_pipeline.py / feature_pipeline_health.py.
        # That means an image can clear health QC above and still fail
        # species QC. predict_species() raises ITS OWN exception classes
        # (defined in predictor.py, not the ones defined in this file), so
        # they must be caught explicitly here or they crash the endpoint
        # with an unhandled 500 instead of a clean error. Try the top view
        # first (it's the primary species-ID view); if that fails for any
        # of these reasons, retry against the bottom view before giving up.
        species_errors = {}
        species_result = None
        for which, img_bytes in (("top", top_image_bytes), ("bottom", bottom_image_bytes)):
            try:
                species_result = predict_species(img_bytes)
                break
            except SpeciesInvalidImageError as e:
                species_errors[which] = f"invalid image ({e})"
            except SpeciesLeafNotDetectedError as e:
                species_errors[which] = f"leaf not detected ({e.reason})"
            except SpeciesFeatureMismatchError as e:
                species_errors[which] = f"feature mismatch ({e})"

        if species_result is None:
            reason = "; ".join(f"{k}: {v}" for k, v in species_errors.items())
            raise LeafNotDetectedError(
                "top+bottom (species-ID)",
                f"could not identify species from either view — {reason}",
            )
        species = species_result["species"]
    elif species not in SPECIES_METADATA:
        raise InvalidSpeciesError(species)

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
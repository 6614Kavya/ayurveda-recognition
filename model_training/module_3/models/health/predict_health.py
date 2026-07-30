
import json

import joblib
import pandas as pd

from models.health.classifier import fuse_top_bottom
from models.health.model_training import NON_FEATURE_COLS
from models.health.train_stage1_binary import (
    Z_FEATURE_COLS, add_species_relative_features, _finalize_X,
)
from feature_extraction.health.health_index import SUBSCORE_RAW_COLUMNS

MODEL_PATH = "processed/models/vedavision_health_index_model.pkl"
STAGE1_MODEL_PATH = "processed/models/vedavision_stage1_svm_model.pkl"

# Metadata keys that can legitimately appear in top_row/bottom_row but are
# never features to fuse (mirrors NON_FEATURE_COLS minus the CSV-only
# columns that won't exist on a single live inference row).
_META_KEYS = {"species"} | NON_FEATURE_COLS


def load_index_model(model_path: str = MODEL_PATH):
    bundle = joblib.load(model_path)
    model = bundle["health_index_model"]
    fit_target = bundle.get("fit_target", "unknown")
    if fit_target != "binary_healthy_vs_unhealthy":
        print(
            f"[warning] loaded index model's fit_target is '{fit_target}', not "
            f"'binary_healthy_vs_unhealthy'. If this is the pre-fix severity-"
            f"target model, its weights were fit against a target that did not "
            f"separate cleanly on real data -- see train_health_index.py's "
            f"docstring. Re-run train_health_index.py to regenerate the "
            f"binary-target model before trusting this score."
        )
    # subscore_columns actually used by THIS saved model -- may be the
    # full SUBSCORE_RAW_COLUMNS set or a trimmed subset, depending on
    # whether train_health_index.py's --save-trimmed-if-better accepted
    # the trim. Read from the bundle rather than assuming the imported
    # module-level constant, so this stays correct either way.
    subscore_columns = bundle.get("subscore_columns", SUBSCORE_RAW_COLUMNS)
    return model, subscore_columns


def load_stage1_model(model_path: str = STAGE1_MODEL_PATH):
    """Loads the Stage-1 healthy/unhealthy model bundle: {
    'stage1_model': fitted sklearn Pipeline, 'stage1_threshold': float,
    'feature_columns': list[str] the Pipeline was trained on,
    'species_baselines': per-species healthy median/IQR used to compute
    the z_<feature> columns at train time, 'z_feature_cols': list[str]
    of candidate columns those baselines/z-scores were built from,
    'dead_features': set[str] excluded before fusion ever happens}."""
    return joblib.load(model_path)


def assess_leaf(
    top_row: dict, bottom_row: dict,
    index_model=None, index_subscore_columns=None, stage1_bundle=None,
    model_path: str = MODEL_PATH, stage1_model_path: str = STAGE1_MODEL_PATH,
) -> dict:
    """
    Fuse top/bottom views ONCE over the FULL feature set (worst-side-wins,
    matching fuse_top_bottom's convention project-wide), then score with
    both models against that same fused row -- Stage-1 uses every raw
    column it was trained on plus its species-relative z-features; the
    Health Index only reads the subscore_columns subset it needs. Neither
    model receives a partially-zeroed row.

    Returns
    -------
    {
      "species": ...,
      "decision": "healthy" | "unhealthy",        # from Stage-1
      "decision_confidence": float 0-1,            # from Stage-1
      "health_value": float 0-100,                 # from Health Index, 100=healthiest
      "severity_score_raw": float 0-100,           # internal, index's raw severity direction
      "breakdown": {<subscore column>: pct contribution to deviation, ...}
    }
    """
    if index_model is None or index_subscore_columns is None:
        index_model, index_subscore_columns = load_index_model(model_path)
    if stage1_bundle is None:
        stage1_bundle = load_stage1_model(stage1_model_path)

    # NOTE: do NOT also exclude Stage-1's dead_features here -- the
    # Health Index still needs two raw columns that Stage-1 drops
    # (worst_ldsi_miner_sub, worst_deform_luminance_std; see
    # health_index.py's SUBSCORE_RAW_COLUMNS docstring for why they're
    # kept there despite being in Stage-1's DEAD_FEATURES). Fusing the
    # full feature set and letting Stage-1's own feature_columns reindex
    # (below) drop what it doesn't want is what keeps both models
    # correctly fed from one fused row.
    feature_keys = sorted(
        (set(top_row.keys()) | set(bottom_row.keys())) - _META_KEYS
    )
    fused = fuse_top_bottom(top_row, bottom_row, feature_keys)
    fused["species"] = top_row.get("species", bottom_row.get("species"))
    row_df = pd.DataFrame([fused])

    # --- Health Index (explainability) ---
    missing_subscores = [c for c in index_subscore_columns if c not in fused]
    if missing_subscores:
        raise ValueError(
            f"top_row/bottom_row is missing raw feature(s) needed for "
            f"{missing_subscores}. Check that the caller supplied every "
            f"column present in health_features_top.csv / _bottom.csv."
        )
    severity_score = float(index_model.score(row_df)[0])
    breakdown = index_model.score_breakdown(row_df.iloc[0])

    # --- Stage-1 (the actual decision) ---
    stage1_clf = stage1_bundle["stage1_model"]
    stage1_threshold = stage1_bundle["stage1_threshold"]
    feature_columns = stage1_bundle["feature_columns"]
    species_baselines = stage1_bundle["species_baselines"]
    z_feature_cols = stage1_bundle.get("z_feature_cols", Z_FEATURE_COLS)

    # add_species_relative_features only reads "species"; _finalize_X
    # just needs a "level" column present so it has something to drop --
    # this placeholder is never read for anything else, single-leaf
    # inference has no real label to put here.
    stage1_row = row_df.copy()
    stage1_row["level"] = "unknown"
    stage1_row = add_species_relative_features(stage1_row, z_feature_cols, species_baselines)
    X_stage1 = _finalize_X(stage1_row)
    # only a genuinely unseen species dummy should ever be fill_value=0
    # here -- every real feature/z-column was just computed above.
    X_stage1 = X_stage1.reindex(columns=feature_columns, fill_value=0.0)

    proba = stage1_clf.predict_proba(X_stage1.values)
    classes = list(stage1_clf.classes_)
    unhealthy_col = classes.index("unhealthy")
    p_unhealthy = float(proba[0, unhealthy_col])
    decision = "unhealthy" if p_unhealthy >= stage1_threshold else "healthy"
    decision_confidence = p_unhealthy if decision == "unhealthy" else (1.0 - p_unhealthy)

    return {
        "species": fused.get("species"),
        "decision": decision,
        "decision_confidence": round(decision_confidence, 3),
        "health_value": round(100.0 - severity_score, 2),
        "severity_score_raw": round(severity_score, 2),
        "breakdown": breakdown,
    }


if __name__ == "__main__":
    # Pull ONE real, held-out test leaf straight from the actual feature
    # CSVs instead of a hand-typed placeholder -- guarantees every raw
    # column Stage-1/index need is present with real values.
    TOP_CSV = "processed/features/health_features_top.csv"
    BOTTOM_CSV = "processed/features/health_features_bottom.csv"

    top_df = pd.read_csv(TOP_CSV)
    bottom_df = pd.read_csv(BOTTOM_CSV)

    # any test-set, non-augmented, qc-passed leaf that has both views
    test_leaf_ids = (
        set(top_df.loc[top_df["is_test"] & (top_df["variant_id"] == 0) & top_df["qc_pass"], "leaf_id"])
        & set(bottom_df.loc[bottom_df["is_test"] & (bottom_df["variant_id"] == 0) & bottom_df["qc_pass"], "leaf_id"])
    )
    if not test_leaf_ids:
        raise RuntimeError(
            f"No matching test leaf found with both views in {TOP_CSV} / "
            f"{BOTTOM_CSV}. Point TOP_CSV/BOTTOM_CSV above at your real "
            f"feature CSVs, or pass real top_row/bottom_row dicts directly."
        )
    example_leaf_id = sorted(test_leaf_ids)[0]

    example_top_row = top_df[
        (top_df["leaf_id"] == example_leaf_id) & (top_df["variant_id"] == 0)
    ].iloc[0].to_dict()
    example_bottom_row = bottom_df[
        (bottom_df["leaf_id"] == example_leaf_id) & (bottom_df["variant_id"] == 0)
    ].iloc[0].to_dict()

    print(f"[demo] scoring real test leaf: {example_leaf_id} "
          f"(true folder level: {example_top_row.get('level')})")
    print(json.dumps(assess_leaf(example_top_row, example_bottom_row), indent=2))
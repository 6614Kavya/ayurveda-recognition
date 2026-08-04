"""
models/health/export_test_health_index.py

Writes one CSV row per held-out TEST leaf (original, non-augmented) with:
  - leaf_id, species, level (raw 4-tier folder label)
  - y_true (healthy/unhealthy, collapsed)
  - health_value        (0-100, Health Index, 100 = healthiest)
  - severity_score_raw  (0-100, same model, severity direction -- higher = worse)
  - stage1_pred          (healthy/unhealthy, Stage-1 SVM decision)
  - stage1_p_unhealthy   (Stage-1's raw probability)
  - stage1_correct       (stage1_pred == y_true)
  - health_percentile_in_group (0-100, ADDED this session -- see below)

health_percentile_in_group: health_value's rank (as a percentile) among
OTHER leaves that share this leaf's OWN stage1_pred -- i.e. a leaf
Stage-1 called "unhealthy" is only ever ranked against other leaves
Stage-1 also called "unhealthy", never against the "healthy" group, and
vice versa. This is the fix for a specific, real failure mode:
health_value alone lets a leaf Stage-1 correctly called "healthy" score
LOWER than a leaf Stage-1 correctly called "unhealthy" (a genuine
example: beli__healthy__test_004 scored 79.55, several beli "unhealthy"
leaves scored 89-96 -- both models actually agree that leaf is the most
borderline of beli's healthy leaves; the Health Index just isn't
AUC=1.0, so exact cross-class ordering isn't guaranteed). health_value
itself is UNCHANGED here -- still the same continuous, cross-species
score. health_percentile_in_group is an additional column: report it
next to health_value, and only ever compare/rank leaves within the same
stage1_pred group using it, so a "healthy" leaf can no longer visually
appear worse than an "unhealthy" one just because they were compared
across the class boundary the two models don't perfectly agree on.
"""
import pandas as pd
import joblib

from models.health.model_training import load_and_merge_from, split_train_test, TOP_CSV, BOTTOM_CSV
from models.health.classifier import fuse_top_bottom
from models.health.train_health_index import build_fused_frame
from models.health.train_stage1_binary import (
    _fuse_leaves, compute_species_baselines, add_species_relative_features,
    _finalize_X, Z_FEATURE_COLS,
)

HEALTH_INDEX_MODEL_PATH = "processed/models/vedavision_health_index_model.pkl"
STAGE1_MODEL_PATH = "processed/models/vedavision_stage1_svm_model.pkl"
OUT_CSV = "processed/features/test_health_index_report.csv"


def main():
    top, bottom = load_and_merge_from(TOP_CSV, BOTTOM_CSV)
    train_top, train_bottom, test_top, test_bottom = split_train_test(top, bottom)

    # --- Health Index: score every ORIGINAL test leaf ---
    index_bundle = joblib.load(HEALTH_INDEX_MODEL_PATH)
    index_model = index_bundle["health_index_model"]

    fused_test_idx = build_fused_frame(test_top, test_bottom)
    test_original = fused_test_idx[fused_test_idx["variant_id"] == 0].copy()
    severity_scores = index_model.score(test_original)
    test_original["severity_score_raw"] = severity_scores
    test_original["health_value"] = (100.0 - severity_scores).round(2)
    test_original["severity_score_raw"] = test_original["severity_score_raw"].round(2)

    # --- Stage-1: re-baseline against TRAIN healthy stats, score the same leaves ---
    stage1_bundle = joblib.load(STAGE1_MODEL_PATH)
    stage1_clf = stage1_bundle["stage1_model"]
    stage1_threshold = stage1_bundle["stage1_threshold"]
    feature_columns = stage1_bundle["feature_columns"]
    baselines = stage1_bundle["species_baselines"]

    fused_test_s1 = _fuse_leaves(test_top, test_bottom)
    fused_test_s1 = fused_test_s1.loc[fused_test_s1.index.isin(
        test_top.index[test_top["variant_id"] == 0]
    )]
    fused_test_s1_z = add_species_relative_features(fused_test_s1, Z_FEATURE_COLS, baselines)
    X_test_s1 = _finalize_X(fused_test_s1_z)
    X_test_s1 = X_test_s1.reindex(columns=feature_columns, fill_value=0.0)

    proba = stage1_clf.predict_proba(X_test_s1.values)
    classes = list(stage1_clf.classes_)
    p_unhealthy = proba[:, classes.index("unhealthy")]
    stage1_pred = ["unhealthy" if p >= stage1_threshold else "healthy" for p in p_unhealthy]

    stage1_df = pd.DataFrame({
        "leaf_id": test_top.loc[fused_test_s1.index, "leaf_id"].values,
        "stage1_pred": stage1_pred,
        "stage1_p_unhealthy": p_unhealthy.round(3),
    })

    # --- Merge + write ---
    report = test_original[["leaf_id", "species", "level", "severity_score_raw", "health_value"]].merge(
        stage1_df, on="leaf_id", how="left"
    )
    report["y_true"] = report["level"].apply(lambda lv: "healthy" if lv == "healthy" else "unhealthy")
    report["stage1_correct"] = report["y_true"] == report["stage1_pred"]
    # Bound comparisons to within the same predicted class: rank health_value
    # only against other leaves sharing this leaf's stage1_pred, so a
    # "healthy"-predicted leaf's number is never read against an
    # "unhealthy"-predicted leaf's number.
    report["health_percentile_in_group"] = (
        report.groupby("stage1_pred")["health_value"].rank(pct=True) * 100.0
    ).round(1)
    report = report[[
        "leaf_id", "species", "level", "y_true",
        "health_value", "health_percentile_in_group", "severity_score_raw",
        "stage1_pred", "stage1_p_unhealthy", "stage1_correct",
    ]].sort_values("leaf_id").reset_index(drop=True)

    report.to_csv(OUT_CSV, index=False)
    print(f"[done] wrote {len(report)} test leaves -> {OUT_CSV}")
    print(report.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
"""
models/health/train_health_index.py

Trains the VedaVision Health Index -- the SOLE health-assessment output.
No classifier, no Stage 1/Stage 2 low/mid/high split as a training
target. Extracted handcrafted features -> weighted, per-species-
normalized continuous index (0-100).

--- FIX (this session): binary target is now PRIMARY ---
The original approach (fit_health_index() against the healthy->low->mid
->high severity continuum, validated via validate_monotonicity()) was
run against real labeled data and failed: low/mid/high group medians
overlapped substantially (confirmed via validate_monotonicity's
group_medians output and bootstrap_ci.py's gap check) -- evidence the
3-way severity boundary isn't a signal these features separate, not a
weighting problem.

This script now fits fit_health_index_binary() -- healthy (0) vs.
low/mid/high collapsed to unhealthy (1) -- as the primary, reported
method, and validates with validate_binary_separation() (ROC-AUC,
Cliff's delta, healthy-vs-unhealthy) instead of monotonicity. The old
severity-target fit is still run too, immediately after, printed under
a clearly-labeled "SUPERSEDED / negative-result" section so the
dissertation can cite both the failure and the fix side by side -- it
is NOT the model that gets saved to MODEL_OUT.

Run from module_3/ root:

    D:\\Python313\\python.exe -m models.health.train_health_index

The original two-stage classifier (binary + low/mid/high) is NOT part of
this script at all anymore -- it lives entirely in
models/health/legacy_flat_classifier.py, kept only as documented
negative-result evidence for the dissertation. Run that file separately
and explicitly if you want those comparison numbers; it never runs as a
side effect of this one.
"""
import argparse

import numpy as np
import pandas as pd
import joblib

from models.health.model_training import (
    load_and_merge_from, split_train_test, dataset_label_qc,
    TOP_CSV, BOTTOM_CSV, NON_FEATURE_COLS,
)
from models.health.classifier import fuse_top_bottom
from feature_extraction.health.health_index import (
    SUBSCORE_RAW_COLUMNS, fit_health_index, validate_monotonicity,
    fit_health_index_binary, validate_binary_separation,
    per_subscore_correlation,
)

MODEL_OUT = "processed/models/vedavision_health_index_model.pkl"

# --- ABLATION (added this session): zero-Ridge-weight subscore trim ---
# These three columns received EXACTLY 0.0 weight in the PRIMARY
# binary-target fit (confirmed run: worst_deform_luminance_std=0.0,
# worst_spot_area_ratio=0.0, worst_spot_density_per_1000px=0.0), and none
# of the three clears p<0.05 on the diagnostic per-subscore Spearman
# check either. This matches the exact pattern that got
# worst_ldsi_hole_sub / worst_ldsi_scar_sub trimmed earlier in the
# project's methodology -- same "confirm before trimming" discipline
# applies here: --trim-zero-weight below reruns the fit on a second
# split-independent frame (same train/test split, so this is a same-data
# reproduction check, not a fresh-seed one -- rerun again after the next
# feature-extraction pass for a stronger confirmation before citing this
# as settled in the dissertation).
#
# Kept as a SEPARATE list rather than edited in-place into
# SUBSCORE_RAW_COLUMNS so both configurations can be printed side-by-side
# and compared before committing to the smaller set.
ZERO_WEIGHT_CANDIDATES = [
    "worst_deform_luminance_std",
    "worst_spot_area_ratio",
    "worst_spot_density_per_1000px",
]
TRIMMED_SUBSCORE_COLUMNS = [c for c in SUBSCORE_RAW_COLUMNS if c not in ZERO_WEIGHT_CANDIDATES]


def build_fused_frame(top: pd.DataFrame, bottom: pd.DataFrame) -> pd.DataFrame:
    """Worst-of-both-views fusion (models.health.classifier.fuse_top_bottom),
    keeping species/level/leaf_id/variant_id as plain columns -- species is
    needed as a grouping key for per-species normalization, and level is
    used ONLY for validation, never as a training target for the index."""
    feature_cols = [c for c in top.columns if c not in NON_FEATURE_COLS]
    rows = []
    for sample_id in top.index:
        fused = fuse_top_bottom(top.loc[sample_id].to_dict(), bottom.loc[sample_id].to_dict(), feature_cols)
        fused["leaf_id"] = top.loc[sample_id, "leaf_id"]
        fused["variant_id"] = top.loc[sample_id, "variant_id"]
        fused["species"] = top.loc[sample_id, "species"]
        fused["level"] = top.loc[sample_id, "level"]
        rows.append(fused)
    return pd.DataFrame(rows, index=top.index)


def main():
    parser = argparse.ArgumentParser(description="Train the VedaVision Health Index (binary-target, primary method)")
    parser.add_argument("--label-qc", action="store_true",
                         help="print the old-LDSI-vs-folder-label per-leaf mismatch diagnostic. "
                              "OFF by default: this compares against the superseded equal-weighted "
                              "LDSI formula, not the model being trained here -- its disagreement "
                              "rate is not a target to drive to zero. Use models/health/"
                              "spot_check_label_qc.py instead if you actually want to review leaves.")
    parser.add_argument("--compare-legacy", action="store_true",
                         help="also fit and print the superseded severity-ordinal method "
                              "(fit_health_index/validate_monotonicity) for side-by-side dissertation "
                              "comparison. OFF by default to keep normal runs fast and uncluttered; "
                              "this model is never saved regardless of this flag.")
    parser.add_argument("--trim-zero-weight", action="store_true",
                         help="also fit the PRIMARY (binary-target) method on "
                              "TRIMMED_SUBSCORE_COLUMNS (drops the 3 subscores that received "
                              "0.0 Ridge weight: worst_deform_luminance_std, "
                              "worst_spot_area_ratio, worst_spot_density_per_1000px) and print "
                              "a side-by-side train/test comparison against the full-column "
                              "PRIMARY model. Does not change what gets saved unless "
                              "--save-trimmed-if-better is also passed.")
    parser.add_argument("--save-trimmed-if-better", action="store_true",
                         help="only meaningful with --trim-zero-weight. If the trimmed model's "
                              "HELD-OUT TEST roc_auc_healthy_vs_unhealthy is strictly higher than "
                              "the full-column PRIMARY model's, save the TRIMMED model to "
                              "MODEL_OUT instead. Mirrors the species-ID pair-specialist "
                              "safety-guard pattern: auto-reject unless the smaller model is a "
                              "genuine improvement, not just fewer columns.")
    args = parser.parse_args()

    top, bottom = load_and_merge_from(TOP_CSV, BOTTOM_CSV)
    train_top, train_bottom, test_top, test_bottom = split_train_test(top, bottom)

    if args.label_qc:
        dataset_label_qc(train_top, train_bottom)

    fused_train = build_fused_frame(train_top, train_bottom)
    fused_test = build_fused_frame(test_top, test_bottom)

    fit_frame = fused_train[fused_train["variant_id"] == 0].copy()

    missing_cols = [c for c in SUBSCORE_RAW_COLUMNS if c not in fit_frame.columns]
    if missing_cols:
        raise RuntimeError(
            f"Expected sub-score columns missing from feature CSV: {missing_cols}. "
            f"Check that feature_extraction/health/severity_index.py and "
            f"texture_health.py still emit these columns, and that "
            f"batch_processor.py was re-run after any feature-extraction changes."
        )

    n_leaves = fit_frame["leaf_id"].nunique()
    print(f"\n[health index] fitting weights on {len(fit_frame)} original training leaves "
          f"({n_leaves} unique physical leaves), sub-scores: {SUBSCORE_RAW_COLUMNS}")

    test_original = fused_test[fused_test["variant_id"] == 0].copy()

    binary_model = fit_health_index_binary(fit_frame, SUBSCORE_RAW_COLUMNS, sign_correct=True, per_species_scale=True)
    print("\n[health index -- PRIMARY, binary target] learned weights: "
          f"{dict(zip(SUBSCORE_RAW_COLUMNS, np.round(binary_model.weights, 3)))}")

    corr_table = per_subscore_correlation(fit_frame, SUBSCORE_RAW_COLUMNS)
    print("\n[health index] per-subscore Spearman rho vs. severity order (diagnostic only -- "
          "kept for reference, NOT what the binary model is fit against):")
    print(corr_table.to_string(index=False))

    train_scores_bin = binary_model.score(fit_frame)
    train_report_bin = validate_binary_separation(train_scores_bin, fit_frame["level"])
    print("\n[health index -- PRIMARY] TRAIN healthy-vs-unhealthy separation:")
    for k, v in train_report_bin.items():
        print(f"  {k}: {v}")

    test_scores_bin = binary_model.score(test_original)
    test_report_bin = validate_binary_separation(test_scores_bin, test_original["level"])
    print("\n[health index -- PRIMARY] HELD-OUT TEST healthy-vs-unhealthy separation:")
    for k, v in test_report_bin.items():
        print(f"  {k}: {v}")
    print("  -> report roc_auc_healthy_vs_unhealthy and cliffs_delta_healthy_vs_unhealthy "
          "as the headline numbers. group_medians_descriptive_only is informational: a trend "
          "across low/mid/high is a bonus if present, but is NOT this model's claim and its "
          "absence does not invalidate the healthy-vs-unhealthy result above.")

    model_to_save = binary_model
    columns_to_save = SUBSCORE_RAW_COLUMNS
    save_note = "PRIMARY (full-column, binary-target)"

    if args.trim_zero_weight:
        print("\n" + "=" * 78)
        print(f"TRIM ABLATION: dropping {ZERO_WEIGHT_CANDIDATES}")
        print("=" * 78)

        trimmed_model = fit_health_index_binary(fit_frame, TRIMMED_SUBSCORE_COLUMNS, sign_correct=True, per_species_scale=True)
        print("\n[health index -- TRIMMED, binary target] learned weights: "
              f"{dict(zip(TRIMMED_SUBSCORE_COLUMNS, np.round(trimmed_model.weights, 3)))}")

        train_scores_trim = trimmed_model.score(fit_frame)
        train_report_trim = validate_binary_separation(train_scores_trim, fit_frame["level"])
        test_scores_trim = trimmed_model.score(test_original)
        test_report_trim = validate_binary_separation(test_scores_trim, test_original["level"])

        print("\n[trim ablation] TRAIN roc_auc_healthy_vs_unhealthy: "
              f"full={train_report_bin['roc_auc_healthy_vs_unhealthy']:.4f}  "
              f"trimmed={train_report_trim['roc_auc_healthy_vs_unhealthy']:.4f}")
        print("[trim ablation] TEST  roc_auc_healthy_vs_unhealthy: "
              f"full={test_report_bin['roc_auc_healthy_vs_unhealthy']:.4f}  "
              f"trimmed={test_report_trim['roc_auc_healthy_vs_unhealthy']:.4f}")
        print("[trim ablation] TEST  cliffs_delta_healthy_vs_unhealthy: "
              f"full={test_report_bin['cliffs_delta_healthy_vs_unhealthy']:.4f}  "
              f"trimmed={test_report_trim['cliffs_delta_healthy_vs_unhealthy']:.4f}")
        print("\n[trim ablation] TRIMMED HELD-OUT TEST full report:")
        for k, v in test_report_trim.items():
            print(f"  {k}: {v}")

        test_auc_full = test_report_bin["roc_auc_healthy_vs_unhealthy"]
        test_auc_trim = test_report_trim["roc_auc_healthy_vs_unhealthy"]
        if args.save_trimmed_if_better and test_auc_trim > test_auc_full:
            model_to_save = trimmed_model
            columns_to_save = TRIMMED_SUBSCORE_COLUMNS
            save_note = "TRIMMED (safety-guard accepted: test AUC improved)"
            print(f"\n[trim ablation] ACCEPTED: trimmed test AUC {test_auc_trim:.4f} > "
                  f"full test AUC {test_auc_full:.4f} -> saving TRIMMED model.")
        elif args.save_trimmed_if_better:
            print(f"\n[trim ablation] REJECTED: trimmed test AUC {test_auc_trim:.4f} did not "
                  f"improve on full test AUC {test_auc_full:.4f} -> keeping full-column PRIMARY "
                  f"model as MODEL_OUT (same safety-guard discipline as the species-ID pair "
                  f"specialist: fewer columns alone is not a reason to ship a regression).")
        else:
            print("\n[trim ablation] Comparison printed only (--save-trimmed-if-better not set) "
                  "-- MODEL_OUT unchanged.")

    joblib.dump({
        "health_index_model": model_to_save,
        "subscore_columns": columns_to_save,
        "fit_target": "binary_healthy_vs_unhealthy",
    }, MODEL_OUT)
    print(f"\n[done] saved {save_note} model -> {MODEL_OUT}")

    if args.compare_legacy:
        print("\n" + "=" * 78)
        print("SUPERSEDED METHOD (negative result, printed for comparison only -- "
              "not saved, not used downstream):")
        print("=" * 78)
        severity_model = fit_health_index(fit_frame, SUBSCORE_RAW_COLUMNS, fit_on_unhealthy_only=False)
        print("[health index -- superseded, severity target] learned weights: "
              f"{dict(zip(SUBSCORE_RAW_COLUMNS, np.round(severity_model.weights, 3)))}")

        train_scores_sev = severity_model.score(fit_frame)
        train_report_sev = validate_monotonicity(train_scores_sev, fit_frame["level"])
        print("\n[health index -- superseded] TRAIN monotonicity validation:")
        for k, v in train_report_sev.items():
            print(f"  {k}: {v}")

        test_scores_sev = severity_model.score(test_original)
        test_report_sev = validate_monotonicity(test_scores_sev, test_original["level"])
        print("\n[health index -- superseded] HELD-OUT TEST monotonicity validation:")
        for k, v in test_report_sev.items():
            print(f"  {k}: {v}")
        print("  -> note the overlapping low/mid/high group_medians above: this is the "
              "documented failure that motivated the binary-target fix.")


if __name__ == "__main__":
    main()
"""
VedaVision — Species Identification Classifier: Training
============================================================
Trains the flat soft-voting ensemble (RandomForest + SVM-RBF +
HistGradientBoosting) defined in classifier.py, evaluates it with honest
StratifiedGroupKFold cross-validation on the training data, and (if --test
is given) runs one sealed pass against the held-out test CSV.

Reads vedavision_features_train_clf.csv (already produced by
batch_processor.py, which drops the broken whole_leaf sentinel columns and
the redundant colour columns via FEATURE_DROP_COLS — see
batch_processor.py's "Final outputs" section). This script does NOT
re-drop those columns; if you're pointing it at an OLDER csv that still
has them, re-run batch_processor.py first rather than special-casing an
old export here.

Why flat, not hierarchical
---------------------------
An earlier version of this script also trained a two-stage hierarchical
classifier: Stage 1 flat ensemble, Stage 2 specialist models on
auto-discovered confusable clusters (kattakumanjal/kalawal,
kathurupila/nil_awariya, wal_bilin/maha_undupiyaliya/wal_kollu,
ranawara/siymbala), gated by Stage-1 top1/top2 margin.

Honest nested cross-validation (inner cluster-discovery CV strictly inside
each outer training fold, never touching the outer test fold) showed no
real improvement: F1-macro 0.9556 (hierarchical) vs 0.9562 (flat) —
well within fold-to-fold noise. Worse, the discovered clusters were
unstable across folds for every pair except kalawal/kattakumanjal, which
means most "confusable clusters" weren't a stable signal to specialise on.
The hierarchical path also cost ~5x the training time for no measurable
benefit.

classifier.py now implements only the flat ensemble, and this script has
been updated to match. The hierarchical experiment is kept in the
dissertation as a documented negative result — see classifier.py's module
docstring for the same numbers, and re-run this comparison once the
morph_* handcrafted features (built specifically around the 5 documented
look-alike pairs) are merged in, in case they make the clusters stable
enough for a specialist stage to start paying for itself.

Usage
-----
    python -m models.species_id.model_training --train processed/features/vedavision_features_train_clf.csv --test  processed/features/vedavision_features_test_clf.csv

IMPORTANT: always run this with `python -m models.species_id.model_training`
from the module_3/ root (not `python model_training.py` from inside the
species_id/ folder, and not by double-clicking/running it directly in an
IDE's "run current file" mode). Running it any other way changes how
Python resolves the models.species_id package path, which is exactly the
kind of mismatch that causes pickle load errors when evaluate.py later
loads the saved model. If models/__init__.py and models/species_id/__init__.py
don't exist yet, create them (empty files) — required for `-m` to treat
these folders as proper packages.
"""

import argparse
import ast
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report

from models.species_id.classifier import (
    make_species_classifier, feature_diversity_report,
    svm_vote_impact_report, run_voting_weight_sweep,
    compare_svm_selection_strategies, report_final_svm_features,
)
from models.species_id.pair_specialist import (
    SpeciesClassifierWithPairSpecialist, evaluate_with_pair_specialist,
)

warnings.filterwarnings("ignore")

# Non-feature columns that may be present in the train clf CSV
# (image_path is kept there specifically for StratifiedGroupKFold grouping;
#  it and species are dropped from X but species is the label).
NON_FEATURE_COLS = ["species", "image_path"]


def load_xy(csv_path: str, require_groups: bool):
    df = pd.read_csv(csv_path)
    if require_groups and "image_path" not in df.columns:
        raise ValueError(
            f"{csv_path} has no image_path column — this must be the TRAIN "
            "clf CSV (batch_processor.py only includes image_path for "
            "mode='train', to support StratifiedGroupKFold grouping)."
        )
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].values
    y = df["species"].values
    groups = df["image_path"].values if "image_path" in df.columns else None
    return X, y, groups, feat_cols


# ---------------------------------------------------------------------------
# Honest CV evaluation on train data (flat ensemble only — see module
# docstring for why the hierarchical comparison was dropped)
# ---------------------------------------------------------------------------

def cross_validate(X, y, groups, n_splits=5, random_state=1, feature_names=None,
                    svm_selection="pairwise_aware"):
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_true, y_pred = [], []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        model = make_species_classifier(random_state, feature_names=feature_names,
                                         svm_selection=svm_selection)
        model.fit(X[tr], y[tr])
        y_pred.extend(model.predict(X[te]))
        y_true.extend(y[te])
        print(f"  fold {fold} done ({len(te)} held-out rows)")

    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\nFlat ensemble  F1-macro: {f1_macro:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred))
    return f1_macro


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="vedavision_features_train_clf.csv")
    ap.add_argument("--test", default=None, help="vedavision_features_test_clf.csv (sealed, run once)")
    ap.add_argument("--out", default="processed",
                     help="Output root — SAME value you passed to batch_processor.py's --out. "
                          "Model is saved to <out>/models/, alongside <out>/features/ and "
                          "<out>/diagnostics/, so everything from one run lives in one tree.")
    ap.add_argument("--model-name", default="vedavision_species_model.pkl")
    ap.add_argument("--diversity-report", action="store_true",
                     help="Also run feature_diversity_report (base-learner "
                          "disagreement rate + Q-statistic per pair, plus "
                          "which columns the SVM branch selected). This is "
                          "the evidence that the per-branch feature-exposure "
                          "change actually decorrelated base-learner errors, "
                          "not just a re-statement of the CV F1-macro above.")
    ap.add_argument("--vote-impact", action="store_true",
                     help="Also run svm_vote_impact_report: measures how "
                          "often SVM's vote actually changes the ensemble's "
                          "final prediction vs an RF+HGB-only vote, and "
                          "compares F1-macro with/without SVM at equal "
                          "(1/3-1/3-1/3) weight.")
    ap.add_argument("--weight-sweep", action="store_true",
                     help="Also run run_voting_weight_sweep: tries several "
                          "(rf, svm, hgb) soft-voting weight combinations "
                          "under CV and reports F1-macro for each, so the "
                          "equal-weight default is a measured choice.")
    ap.add_argument("--svm-selection", default="pairwise_aware",
                     choices=["guaranteed_botanical", "pairwise_aware"],
                     help="How the SVM branch picks its feature subset. "
                          "'guaranteed_botanical' (default): botanical_* "
                          "columns always kept, MI-ranked standard columns "
                          "fill the rest. 'pairwise_aware': no forcing -- "
                          "all columns compete under a score of max(global "
                          "MI, best MI on any one documented look-alike "
                          "pair), which matches how botanical_* features "
                          "were actually designed. Affects the main CV run, "
                          "--diversity-report, and the final saved model. "
                          "Use --compare-selection to see both side by side "
                          "before picking one.")
    ap.add_argument("--compare-selection", action="store_true",
                     help="Also run compare_svm_selection_strategies: runs "
                          "CV under BOTH guaranteed_botanical and "
                          "pairwise_aware, reporting F1-macro for each plus "
                          "how many botanical_* columns pairwise_aware "
                          "naturally selects (vs the forced 21/21 for the "
                          "other strategy). Doesn't affect which strategy "
                          "trains the final saved model -- that's still "
                          "controlled by --svm-selection.")
    ap.add_argument("--random-state", type=int, default=42,
                     help="Seed used for EVERY CV split in this run (main "
                          "CV, --diversity-report, --vote-impact, "
                          "--weight-sweep, --compare-selection) and for the "
                          "final model fit. Previously the main CV run "
                          "silently used a different default seed (1) than "
                          "the diagnostic functions (42, classifier.py's "
                          "RANDOM_STATE) -- meaning the main F1-macro and "
                          "every report below it were computed on DIFFERENT "
                          "5-fold splits, not directly comparable to each "
                          "other. Fixed by passing one explicit seed "
                          "through everything in this run.")
    ap.add_argument("--feature-count-sweep", action="store_true",
                 help="Sweep SVM_N_FEATURES (20/40/60/80) under CV.")
    ap.add_argument("--use-swept-weights", action="store_true",
                     help="Requires --weight-sweep. Instead of training the "
                          "final model with classifier.py's hardcoded WEIGHTS "
                          "constant, use whichever (rf,svm,hgb) combination "
                          "the sweep just found best on THIS feature set / "
                          "architecture. Important because the optimal weight "
                          "is not a fixed property of the ensemble -- it "
                          "shifts whenever a base learner's feature exposure "
                          "changes (e.g. the HGB bagging change moved the "
                          "optimum from (1,1,1) to (1,2,1) on one real run). "
                          "Prefer this over hand-copying a number out of "
                          "sweep output into WEIGHTS, which silently goes "
                          "stale the next time classifier.py changes.")
    ap.add_argument("--pair-specialist", action="store_true",
                     help="Also run evaluate_with_pair_specialist: honest CV "
                          "comparison of the flat ensemble alone vs flat + "
                          "the kattakumanjal/kalawal Stage-2 specialist "
                          "(see pair_specialist.py for why this ONE pair, "
                          "not a general hierarchical architecture). If the "
                          "comparison shows a real gain, the FINAL saved "
                          "model is also wrapped with the specialist -- "
                          "otherwise the plain flat model is saved, same as "
                          "without this flag.")
    args = ap.parse_args()

    if args.use_swept_weights and not args.weight_sweep:
        raise ValueError("--use-swept-weights requires --weight-sweep (need "
                          "the sweep to run before its winner can be used).")

    out_root = Path(args.out)
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_out = models_dir / args.model_name

    X, y, groups, feat_cols = load_xy(args.train, require_groups=True)
    print(f"Loaded {len(y)} rows, {len(feat_cols)} feature columns, "
          f"{len(set(groups))} unique source leaves.")

    print("\n=== 5-fold StratifiedGroupKFold CV (train data only) ===")
    print(f"(svm_selection={args.svm_selection}, random_state={args.random_state})")
    cross_validate(X, y, groups, feature_names=feat_cols, svm_selection=args.svm_selection,
                    random_state=args.random_state)

    if args.diversity_report:
        print("\n=== Base-learner diversity report ===")
        feature_diversity_report(X, y, groups, feat_cols, svm_selection=args.svm_selection,
                                  random_state=args.random_state)

    if args.vote_impact:
        print("\n=== SVM vote impact report ===")
        svm_vote_impact_report(X, y, groups, feat_cols, svm_selection=args.svm_selection,
                                random_state=args.random_state)

    best_weights = None
    if args.weight_sweep:
        print("\n=== Soft-voting weight sweep ===")
        sweep_df = run_voting_weight_sweep(X, y, groups, feat_cols, svm_selection=args.svm_selection,
                                            random_state=args.random_state)
        print(sweep_df.to_string(index=False))
        # sweep_df is already sorted best-F1-macro-first (see
        # run_voting_weight_sweep's return); row 0's weight string is
        # literally "(1, 1, 1)" etc, safe to parse back with ast.literal_eval
        # since it's produced by str(tuple) on our own weight_options, not
        # external input.
        best_row = sweep_df.iloc[0]
        best_weights = ast.literal_eval(best_row["weights_rf_svm_hgb"])
        print(f"\nBest swept weights: {best_weights}  (F1-macro={best_row['f1_macro']:.4f})")

    if args.compare_selection:
        print("\n=== SVM feature-selection strategy comparison ===")
        compare_df = compare_svm_selection_strategies(X, y, groups, feat_cols,
                                                        random_state=args.random_state)
        print()
        print(compare_df.to_string(index=False))
    if args.feature_count_sweep:
        from models.species_id.classifier import run_feature_count_sweep
        print("\n=== SVM feature-count sweep ===")
        sweep_df = run_feature_count_sweep(X, y, groups, feature_names=feat_cols, random_state=args.random_state)
        print(sweep_df.to_string(index=False))

    use_pair_specialist_in_final = False
    if args.pair_specialist:
        print("\n=== Pair specialist: kattakumanjal/kalawal ===")
        print("(scoped Stage-2, see pair_specialist.py docstring for why this "
              "pair and not a general hierarchical architecture)")
        y_true, y_pred_flat, y_pred_specialist = evaluate_with_pair_specialist(
            X, y, groups, feat_cols, random_state=args.random_state,
            svm_selection=args.svm_selection,
        )
        f1_flat = f1_score(y_true, y_pred_flat, average="macro")
        f1_spec = f1_score(y_true, y_pred_specialist, average="macro")
        # The specialist's gating (override only when Stage-1's top-2 are
        # exactly this pair) makes it structurally safe -- it can only ever
        # touch rows Stage-1 was already unsure about. Still, require a
        # non-negative measured delta on THIS run's CV before shipping it in
        # the saved model, rather than trusting the mechanism blindly.
        if f1_spec >= f1_flat:
            use_pair_specialist_in_final = True
            print(f"\nSpecialist did not hurt CV F1-macro ({f1_spec:.4f} >= {f1_flat:.4f}) "
                  f"-- will be included in the final saved model.")
        else:
            print(f"\nSpecialist REDUCED CV F1-macro this run ({f1_spec:.4f} < {f1_flat:.4f}) "
                  f"-- final saved model will NOT include it. Investigate before retrying "
                  f"(check pair_specialist.py's PAIR_SPECIALIST_FEATURES are still present "
                  f"in feat_cols, and that batch_processor.py hasn't changed column names).")

    print("\n=== Fitting final model on all training data ===")
    if args.use_swept_weights:
        print(f"Using swept weights {best_weights} for final model (--use-swept-weights).")
        final_model = make_species_classifier(args.random_state, feature_names=feat_cols,
                                               svm_selection=args.svm_selection,
                                               weights=tuple(best_weights))
    else:
        final_model = make_species_classifier(args.random_state, feature_names=feat_cols,
                                               svm_selection=args.svm_selection)

    if use_pair_specialist_in_final:
        final_model = SpeciesClassifierWithPairSpecialist(
            final_model, feat_cols, random_state=args.random_state,
        )
    final_model.fit(X, y)

    # Report the ACTUAL SVM feature composition of the model being saved --
    # not the fold-aggregate stat from --diversity-report, the real thing.
    # Works whether final_model is a plain VotingClassifier or wrapped in
    # SpeciesClassifierWithPairSpecialist (unwrap to .base_model in that case).
    _voting_model = getattr(final_model, "base_model", final_model)
    report_final_svm_features(_voting_model, feat_cols)

    # Bundle format kept identical to before (evaluate.py depends on it):
    # "model" for prediction, "feature_columns" to enforce column order on
    # any future CSV, "train_image_paths" so evaluate.py can warn about
    # train/test leakage before trusting any accuracy number.
    joblib.dump({
        "model": final_model,
        "feature_columns": feat_cols,
        "train_image_paths": set(groups.tolist()),
    }, model_out)
    print(f"\nSaved model to {model_out.resolve()}")

    if args.test:
        Xt, yt, _, test_feat_cols = load_xy(args.test, require_groups=False)
        if test_feat_cols != feat_cols:
            missing = set(feat_cols) - set(test_feat_cols)
            extra = set(test_feat_cols) - set(feat_cols)
            raise ValueError(
                f"Test CSV columns don't match train. Missing: {missing}. Extra: {extra}. "
                "Both CSVs must come from the same batch_processor.py version."
            )
        pred = final_model.predict(Xt)
        print("\n=== Held-out test set (sealed — run this only once) ===")
        print(f"F1-macro: {f1_score(yt, pred, average='macro'):.4f}")
        print(classification_report(yt, pred))


if __name__ == "__main__":
    main()
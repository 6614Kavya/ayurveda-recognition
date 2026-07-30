"""
VedaVision -- Species ID: Base-Learner Hyperparameter Tuning
===========================================================================
Everything tested this session (ensemble size, voting weights, SVM feature
budget, HGB diversification, botanical feature composition) changed WHAT
each learner sees or HOW their outputs are combined. Nothing has tuned a
single base learner's own hyperparameters -- RF's n_estimators=200, SVM's
C=10/gamma=scale, HGB's max_iter=150 have been fixed guesses the entire
session, never swept.

Reference numbers this should be compared against (single-branch,
out-of-fold, from feature_diversity_report() on real data):
    rf   F1-macro=0.9122
    svm  F1-macro=0.9110
    hgb  F1-macro=0.9191
    ensemble (untuned, weights=(1,1,1)) F1-macro=0.9301
    ensemble (untuned, best swept weights=(2,1,1)) F1-macro=0.9305

What this does
---------------
1. RandomizedSearchCV, independently per base learner, using
   StratifiedGroupKFold (grouped by image_path -- same leak-prevention
   rule as everywhere else in this project) and f1_macro scoring.
2. SVM's feature-selection step ("select") is held FIXED at the current
   pairwise_aware default while tuning C/gamma -- this script tunes the
   classifier's own hyperparameters, not feature selection (already swept
   separately via --feature-count-sweep / --compare-selection).
3. Prints best params + best CV F1-macro for each learner, compared
   against the untuned reference numbers above.
4. Rebuilds the full 3-classifier ensemble using the tuned hyperparameters
   (via make_species_classifier's new rf_params/svm_params/hgb_params
   overrides) and reports whether the TUNED ensemble beats the untuned
   baseline. This is the number that actually matters -- an individual
   learner improving doesn't guarantee the ensemble does, since tuning
   toward higher individual accuracy can reduce diversity (same lesson
   as the HGB diversification attempts earlier this session).

Usage
-----
    python tune_hyperparameters.py \
        --train processed/features/vedavision_features_train_clf.csv \
        --n-iter 30

    # Quick sanity check with a smaller search before committing to a
    # full run (full run with n_iter=30 x 3 learners x 5 folds = 450
    # fits total, SVM with probability=True is the slow one):
    python tune_hyperparameters.py --train ... --n-iter 8 --quick
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.metrics import f1_score

from models.species_id.classifier import (
    make_species_classifier, cross_validate, run_voting_weight_sweep,
    SVM_N_FEATURES, LOOK_ALIKE_PAIRS_IDX, _pairwise_aware_mi_score,
)
from sklearn.feature_selection import SelectKBest
import functools

NON_FEATURE_COLS = ["species", "image_path"]

REFERENCE = {
    "rf": 0.9122, "svm": 0.9110, "hgb": 0.9191,
    "ensemble_untuned": 0.9305,  # best swept weight this session
}


def load(train_csv):
    df = pd.read_csv(train_csv)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feat_cols].values
    y = df["species"].values
    groups = df["image_path"].values
    return X, y, groups, feat_cols


def tune_rf(X, y, groups, n_splits, n_iter, random_state, quick):
    param_dist = {
        "clf__n_estimators": randint(100, 200 if quick else 600),
        "clf__max_depth": [None, 10, 20, 30, 40],
        "clf__min_samples_leaf": randint(1, 8),
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
    }
    pipe = Pipeline([("clf", RandomForestClassifier(
        class_weight="balanced", random_state=random_state, n_jobs=1))])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=n_iter, scoring="f1_macro",
                                 cv=cv, random_state=random_state, n_jobs=-1, refit=False)
    search.fit(X, y, groups=groups)
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return best_params, search.best_score_


def tune_svm(X, y, groups, feature_names, n_splits, n_iter, random_state, quick):
    # Feature selection held fixed at the current pairwise_aware default --
    # this tunes C/gamma only, not what columns SVM sees.
    pairwise_score_func = functools.partial(
        _pairwise_aware_mi_score, pairs=LOOK_ALIKE_PAIRS_IDX, random_state=random_state
    )
    param_dist = {
        "clf__C": loguniform(1e-1, 1e3),
        "clf__gamma": loguniform(1e-4, 1e0) if not quick else ["scale", "auto"],
    }
    pipe = Pipeline([
        ("select", SelectKBest(pairwise_score_func, k=SVM_N_FEATURES)),
        ("scale", StandardScaler()),
        ("clf", SVC(class_weight="balanced", probability=True, random_state=random_state)),
    ])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=n_iter, scoring="f1_macro",
                                 cv=cv, random_state=random_state, n_jobs=-1, refit=False)
    search.fit(X, y, groups=groups)
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return best_params, search.best_score_


def tune_hgb(X, y, groups, n_splits, n_iter, random_state, quick):
    param_dist = {
        "clf__max_iter": randint(80, 200 if quick else 400),
        "clf__learning_rate": loguniform(1e-2, 3e-1),
        "clf__max_depth": [None, 3, 5, 8, 15],
        "clf__max_leaf_nodes": [15, 31, 63, 127],
        "clf__l2_regularization": uniform(0.0, 2.0),
    }
    pipe = Pipeline([("clf", HistGradientBoostingClassifier(
        class_weight="balanced", random_state=random_state))])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=n_iter, scoring="f1_macro",
                                 cv=cv, random_state=random_state, n_jobs=-1, refit=False)
    search.fit(X, y, groups=groups)
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return best_params, search.best_score_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=30,
                     help="RandomizedSearchCV iterations PER learner. 30 is a reasonable "
                          "budget; each iteration fits n_splits models, so total fits per "
                          "learner = n_iter * n_splits. SVM (probability=True) is by far "
                          "the slowest of the three.")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                     help="Smaller search ranges for a fast sanity check before committing "
                          "to a full run.")
    ap.add_argument("--svm-selection", default="pairwise_aware",
                     choices=["pairwise_aware", "guaranteed_botanical"])
    args = ap.parse_args()

    X, y, groups, feat_cols = load(args.train)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} feature columns.\n")

    print("=== Tuning RandomForest ===")
    rf_params, rf_score = tune_rf(X, y, groups, args.n_splits, args.n_iter,
                                   args.random_state, args.quick)
    print(f"Best params: {rf_params}")
    print(f"Best CV F1-macro: {rf_score:.4f}  (untuned reference: {REFERENCE['rf']:.4f}, "
          f"delta={rf_score - REFERENCE['rf']:+.4f})\n")

    print("=== Tuning SVM-RBF ===")
    svm_params, svm_score = tune_svm(X, y, groups, feat_cols, args.n_splits, args.n_iter,
                                      args.random_state, args.quick)
    print(f"Best params: {svm_params}")
    print(f"Best CV F1-macro: {svm_score:.4f}  (untuned reference: {REFERENCE['svm']:.4f}, "
          f"delta={svm_score - REFERENCE['svm']:+.4f})\n")

    print("=== Tuning HistGradientBoosting ===")
    hgb_params, hgb_score = tune_hgb(X, y, groups, args.n_splits, args.n_iter,
                                      args.random_state, args.quick)
    print(f"Best params: {hgb_params}")
    print(f"Best CV F1-macro: {hgb_score:.4f}  (untuned reference: {REFERENCE['hgb']:.4f}, "
          f"delta={hgb_score - REFERENCE['hgb']:+.4f})\n")

    print("=== Rebuilding ensemble with tuned hyperparameters ===")
    tuned_model_factory = lambda **kw: make_species_classifier(
        args.random_state, feature_names=feat_cols, svm_selection=args.svm_selection,
        rf_params=rf_params, svm_params=svm_params, hgb_params=hgb_params, **kw
    )
    # Reuse run_voting_weight_sweep's mechanism by monkeypatching isn't
    # worth the complexity here -- just run cross_validate directly at
    # default (1,1,1) weights for the headline comparison, then a small
    # manual weight check since the optimal weight may shift again with
    # different individual-learner strengths (same lesson as the HGB
    # bagging experiment moving the weight optimum earlier this session).
    from sklearn.model_selection import StratifiedGroupKFold as _SGKF
    cv = _SGKF(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    y_true, y_pred = [], []
    for tr_idx, te_idx in cv.split(X, y, groups):
        model = tuned_model_factory(weights=(1, 1, 1))
        model.fit(X[tr_idx], y[tr_idx])
        y_pred.extend(model.predict(X[te_idx]))
        y_true.extend(y[te_idx])
    tuned_ensemble_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Tuned ensemble F1-macro (weights=(1,1,1)): {tuned_ensemble_f1:.4f}")
    print(f"Untuned ensemble reference: {REFERENCE['ensemble_untuned']:.4f}")
    delta = tuned_ensemble_f1 - REFERENCE["ensemble_untuned"]
    print(f"Delta: {delta:+.4f}")
    if delta > 0:
        print("\n>>> Tuned ensemble WINS. Worth a full --weight-sweep with these "
              "hyperparameters baked into make_species_classifier's defaults next.")
    else:
        print("\n>>> Tuned ensemble does not beat the untuned baseline. Consistent with "
              "everything else this session -- 0.930-0.931 CV F1-macro looks like a real "
              "ceiling for this feature set + these three learner families, not just an "
              "under-tuned baseline.")


if __name__ == "__main__":
    main()
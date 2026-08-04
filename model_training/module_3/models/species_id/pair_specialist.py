"""
VedaVision -- Kattakumanjal/Kalawal Pair Specialist (scoped Stage 2)
=======================================================================
NOT a re-introduction of the general hierarchical architecture that was
tried and rejected (see model_training.py's module docstring: F1-macro
0.9556 hierarchical vs 0.9562 flat, within noise, AND cluster discovery
was unstable across folds for every pair except kalawal/kattakumanjal).

This is different in scope: ONE hand-picked, individually-validated pair,
not "specialize on whatever clusters got auto-discovered this run."
Justification chain:
  1. model_training.py's own negative-result writeup already singled out
     kalawal/kattakumanjal as the one STABLE cluster across folds.
  2. analyze_pair_features.py confirmed moderate, non-redundant signal
     exists for this pair specifically (texture LBP, vein density/
     branching, colour lab_b/hsv_s, 2 botanical hits) that does NOT
     overlap with the shape_* features dominating SVM's global selection.
  3. pair_ceiling_check.py confirmed a small dedicated model on just these
     15 features clears ~0.85 F1 for both classes -- +6pts over the full
     ensemble's ~0.79 sealed-test F1 on this pair. Signal exists AND
     combines well; the full ensemble just isn't routing it here.

Routing rule (deliberately conservative): Stage 2 only overrides Stage 1
when Stage 1's top-2 predicted classes for a row are EXACTLY this pair,
regardless of which one is top-1. Every other row -- including rows where
Stage 1 is confident about a third species, or uncertain between this
pair and something else -- is left untouched. This means the specialist
can only ever help pair-level accuracy; it cannot introduce new confusion
with unrelated species, because it never fires for them.

Pickle note: per project convention (classifier.py is the single canonical
import path baked into joblib pickles at training time), if you save a
model wrapped in SpeciesClassifierWithPairSpecialist, both training and
inference (evaluate.py, the FastAPI backend) must import this class from
THIS exact module path. Keep pair_specialist.py at the same package depth
as classifier.py (models/species_id/) for that reason.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, f1_score

from models.species_id.classifier import make_species_classifier

SENTINEL = -1.0

# Features validated by pair_ceiling_check.py against
# kattakumanjal_kalawal_audit.csv (separating_power ranked, sentinel-heavy
# columns already excluded). Re-run the audit + ceiling check before
# editing this list -- it's the evidence trail, not a guess.
PAIR_SPECIALIST_FEATURES = {
    frozenset({"kattakumanjal", "kalawal"}): [
        "texture_lbp_08", "texture_lbp_09", "texture_lbp_07", "colour_lab_b_q75",
        "vein_branch_density", "texture_lbp_06", "colour_hsv_s_skew", "vein_length_ratio",
        "colour_lab_b_median", "whole_aspect", "colour_hsv_s_kurt", "vein_density",
        "colour_botanical_gloss_v_p95_median_ratio", "colour_botanical_oil_gland_density",
        "colour_exg_q75",
    ],
}

MIN_ROWS_FOR_SPECIALIST = 20  # below this, skip rather than fit on too little data


def _make_specialist_pipe(random_state: int) -> VotingClassifier:
    # RF + SVM soft-voting -- both hit ~0.85 F1 independently in the
    # ceiling check (0.8512 / 0.8564), close enough to combine rather than
    # pick one; HGB deliberately left out here, it wasn't tested in the
    # ceiling check and adding an untested third learner to a 2-class
    # specialist on already-thin per-pair data isn't worth the risk.
    rf = Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=random_state, n_jobs=-1)),
    ])
    svm = Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(C=10, gamma="scale", class_weight="balanced",
                    probability=True, random_state=random_state)),
    ])
    return VotingClassifier(estimators=[("rf", rf), ("svm", svm)], voting="soft")


def _impute_sentinel_median(X_sub: np.ndarray) -> np.ndarray:
    X_sub = X_sub.astype(float).copy()
    for j in range(X_sub.shape[1]):
        col = X_sub[:, j]
        valid = col != SENTINEL
        fill = np.median(col[valid]) if valid.any() else 0.0
        col[~valid] = fill
    return X_sub


class SpeciesClassifierWithPairSpecialist:
    """Wraps a fitted flat ensemble (Stage 1) with small dedicated binary
    specialists (Stage 2) for individually-validated confusable pairs.
    See module docstring for why this is scoped, not the general
    hierarchical architecture that was tried and rejected."""

    def __init__(self, base_model, feature_names,
                 pair_features: dict = None, random_state: int = 42):
        self.base_model = base_model
        self.feature_names = list(feature_names)
        self.pair_features = pair_features if pair_features is not None else PAIR_SPECIALIST_FEATURES
        self.random_state = random_state
        self._name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        self.specialists_ = {}   # frozenset(pair) -> (fitted VotingClassifier, col_idx list)

    def fit(self, X, y):
        self.base_model.fit(X, y)
        for pair, feats in self.pair_features.items():
            idx = [self._name_to_idx[f] for f in feats if f in self._name_to_idx]
            missing = [f for f in feats if f not in self._name_to_idx]
            if missing:
                print(f"WARNING: pair specialist for {sorted(pair)} missing columns "
                      f"{missing} -- trained on {len(idx)}/{len(feats)} intended features.")
            row_mask = np.isin(y, list(pair))
            if row_mask.sum() < MIN_ROWS_FOR_SPECIALIST:
                print(f"WARNING: only {int(row_mask.sum())} rows for pair {sorted(pair)} -- "
                      f"skipping specialist (need >= {MIN_ROWS_FOR_SPECIALIST}).")
                continue
            Xp = _impute_sentinel_median(X[row_mask][:, idx])
            yp = y[row_mask]
            specialist = _make_specialist_pipe(self.random_state)
            specialist.fit(Xp, yp)
            self.specialists_[pair] = (specialist, idx)
        return self

    def predict(self, X):
        base_proba = self.base_model.predict_proba(X)
        base_classes = self.base_model.classes_
        preds = base_classes[np.argmax(base_proba, axis=1)]

        if not self.specialists_:
            return preds

        top2_idx = np.argsort(-base_proba, axis=1)[:, :2]
        top2_classes = base_classes[top2_idx]  # (n_rows, 2)

        for pair, (specialist, idx) in self.specialists_.items():
            is_ambiguous = np.array([frozenset(row) == pair for row in top2_classes])
            if not is_ambiguous.any():
                continue
            Xp = _impute_sentinel_median(X[is_ambiguous][:, idx])
            preds[is_ambiguous] = specialist.predict(Xp)

        return preds

    def predict_proba(self, X):
        # Deliberately NOT overridden by the specialist -- keeps base_model's
        # probability calibration intact for anything downstream that reads
        # predict_proba (e.g. the confidence-threshold review triage).
        # Specialist-routed rows are, by construction, already the base
        # model's genuinely-uncertain top-2 cases, so this is a reasonable
        # approximation rather than a correctness issue.
        return self.base_model.predict_proba(X)


def evaluate_with_pair_specialist(X, y, groups, feature_names, n_splits=5,
                                   random_state=42, svm_selection="pairwise_aware"):
    """Honest CV comparison: SAME outer StratifiedGroupKFold splits as the
    flat baseline (same random_state), Stage 1 AND the specialist both
    fit strictly inside each outer training fold -- no leakage of outer
    test rows into specialist training. Directly comparable to your
    existing cross_validate() / --diversity-report numbers."""
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_true, y_pred_flat, y_pred_specialist = [], [], []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        base = make_species_classifier(random_state, feature_names=feature_names,
                                        svm_selection=svm_selection)
        wrapped = SpeciesClassifierWithPairSpecialist(base, feature_names, random_state=random_state)
        wrapped.fit(X[tr], y[tr])

        y_true.extend(y[te])
        y_pred_flat.extend(wrapped.base_model.predict(X[te]))
        y_pred_specialist.extend(wrapped.predict(X[te]))
        print(f"  fold {fold} done ({len(te)} held-out rows)")

    print("\n--- Stage 1 only (flat ensemble) ---")
    f1_flat = f1_score(y_true, y_pred_flat, average="macro")
    print(f"F1-macro: {f1_flat:.4f}")
    print(classification_report(y_true, y_pred_flat, digits=3))

    print("\n--- Stage 1 + kattakumanjal/kalawal specialist ---")
    f1_spec = f1_score(y_true, y_pred_specialist, average="macro")
    print(f"F1-macro: {f1_spec:.4f}")
    print(classification_report(y_true, y_pred_specialist, digits=3))

    print(f"\nDelta F1-macro (specialist - flat): {f1_spec - f1_flat:+.4f}")
    return y_true, y_pred_flat, y_pred_specialist
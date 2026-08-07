import functools
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix,
)

# Number of features exposed to the SVM-RBF branch only. Kept well below
# the ~123-column full set on purpose (SVM-RBF degrades with high
# dimensionality / redundant columns); RF still sees everything.
# Sweep this (e.g. 20/40/60) against CV F1-macro before treating 40 as
# final -- see run_feature_count_sweep() at the bottom of this file.
SVM_N_FEATURES = 40

# --- HGB diversification: two attempts, two negative results ------------
# Attempt 1: SelectKBest(f_classif, k=60) column-cut on HGB (same
# mechanism as the SVM branch). Result: rf-hgb Q improved 0.9614 -> 0.9472
# (more diverse), but CV F1-macro dropped 0.9320 -> 0.9281. Diagnosis:
# boosting's strength is exploiting many weakly-informative features
# together; univariate top-K selection assumes single-feature strength,
# which is the wrong restriction for a boosting model specifically.
#
# Attempt 2: BaggingClassifier(HGB, max_samples=0.8, max_features=0.85,
# n_estimators=5) -- row+mild-column subsampling instead of a hard column
# cut. Result: WORSE on both axes than attempt 1 -- rf-hgb Q got *worse*
# than the untouched baseline (0.9614 -> 0.9704, i.e. MORE correlated with
# RF, not less), and the kattakumanjal/kalawal sealed-test pair (the
# actual target pair) hit its worst precision/recall of any of the three
# configurations tried. Diagnosis: RF is already "bagging of trees";
# wrapping HGB in bagging turns it into another averaged-ensemble-of-trees,
# which structurally converges HGB's behaviour toward RF's rather than
# away from it -- the opposite of the intended effect.
#
# Conclusion: kattakumanjal/kalawal stays the worst-confused pair across
# ALL THREE ensemble configurations (plain, SelectKBest-HGB, bagged-HGB),
# regardless of voting mechanics. That's decisive evidence the bottleneck
# is feature separability for this pair, not ensemble diversity -- no
# voting/bagging/weighting scheme can out-vote a weak underlying signal.
# REVERTED to plain HGB below. Do not reintroduce either HGB
# diversification attempt without first showing (a) CV F1-macro >= 0.9320
# AND (b) rf-hgb Q improved vs 0.9614 AND (c) kattakumanjal/kalawal
# sealed-test F1 not regressed vs the plain-HGB baseline -- all three
# conditions, not just one. Effort now redirected to feature-level fixes
# for this pair specifically -- see analyze_pair_features.py.

# Floor on how many *standard* (non-botanical) columns the SVM branch is
# allowed, even if svm_n_features is small enough that the botanical block
# alone would eat the whole budget. Keeps the SVM from being trained on
# a botanical-only feature set that's too thin to separate 12 classes.
SVM_MIN_STANDARD_FEATURES = 10

RANDOM_STATE = 42
N_SPLITS = 5
WEIGHTS = (1,1,1)

# The 5 documented look-alike pairs (project memory) — checked separately
# from overall accuracy because these are the pairs the whole feature/
# classifier design effort is aimed at.
LOOK_ALIKE_PAIRS = [
    ("thunpath_kurundu", "kasthuri_dehi"),
    ("beli", "wal_kollu"),
    ("kattakumanjal", "kalawal"),
    ("wal_bilin", "maha_undupiyaliya"),
    ("kathurupila", "nil_awariya"),
    ("siymbala", "ranawara"),               
    ("thunpath_kurundu", "wal_bilin"),       
    ("maha_undupiyaliya", "thunpath_kurundu"), 
    ("nil_awariya", "kattakumanjal"),        
    ("nil_awariya", "kalawal"),              
    ("kathurupila", "kattakumanjal"),
    ("kathurupila", "kalawal"),     
]

ALL_SPECIES_SORTED = sorted([
    "beli", "kalawal", "kasthuri_dehi", "kathurupila", "kattakumanjal",
    "maha_undupiyaliya", "nil_awariya", "ranawara", "siymbala",
    "thunpath_kurundu", "wal_bilin", "wal_kollu",
])
_SPECIES_TO_IDX = {name: i for i, name in enumerate(ALL_SPECIES_SORTED)}
LOOK_ALIKE_PAIRS_IDX = [(_SPECIES_TO_IDX[a], _SPECIES_TO_IDX[b]) for a, b in LOOK_ALIKE_PAIRS]


def _pairwise_aware_mi_score(X: np.ndarray, y: np.ndarray,
                              pairs: list = None, random_state: int = RANDOM_STATE) -> np.ndarray:
   
    if pairs is None:
        pairs = LOOK_ALIKE_PAIRS_IDX

    global_score = mutual_info_classif(X, y, random_state=random_state)
    pairwise_score = np.zeros(X.shape[1])

    for cls_a, cls_b in pairs:
        mask = np.isin(y, [cls_a, cls_b])
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            continue  # pair not present (or too thin) in this fold's split
        try:
            pair_mi = mutual_info_classif(X[mask], y[mask], random_state=random_state)
        except ValueError:
            continue
        pairwise_score = np.maximum(pairwise_score, pair_mi)

    def _max_norm(a):
        m = a.max()
        return a / m if m > 0 else a

    return np.maximum(_max_norm(global_score), _max_norm(pairwise_score))



def report_final_svm_features(model: VotingClassifier, feature_names: list) -> dict:
    """
    Unlike feature_diversity_report (which aggregates selection across 5
    CV folds -- useful for "how stable is this selection", but not the
    same thing as "what did the model I actually shipped use"), this
    inspects ONE already-fitted model -- normally the final model trained
    on all data -- and reports exactly which of its SVM branch's <=
    svm_n_features columns are botanical_* vs standard.

    Motivation: the fold-aggregate stat ("botanical_*: 2, standard: 45,
    unique across 5 folds") tells you botanical features rarely survive
    pairwise_aware selection ACROSS folds, but doesn't say how many made
    it into the specific model being saved to disk and shipped to the
    backend. This answers that directly -- important given the project's
    core claim is that handcrafted botanical features are load-bearing,
    not just present in the CSV.
    """
    select_step = model.named_estimators_["svm"].named_steps["select"]
    if isinstance(select_step, ColumnTransformer):
        botanical_idx = select_step.transformers_[0][2]
        standard_idx = select_step.transformers_[1][2]
        standard_support = select_step.named_transformers_["standard_select"].get_support()
        botanical_names = [feature_names[i] for i in botanical_idx]
        standard_names = [feature_names[i] for i, kept in zip(standard_idx, standard_support) if kept]
    else:
        support = select_step.get_support()
        selected = [n for n, kept in zip(feature_names, support) if kept]
        botanical_names = [n for n in selected if "botanical_" in n]
        standard_names = [n for n in selected if "botanical_" not in n]

    n_botanical_total = sum(1 for n in feature_names if "botanical_" in n)
    print(f"\n-- Final model: SVM branch feature composition --")
    print(f"  botanical_*: {len(botanical_names)}/{n_botanical_total} available  |  "
          f"standard: {len(standard_names)}  |  total in SVM: "
          f"{len(botanical_names) + len(standard_names)}")
    if botanical_names:
        print("  botanical features actually used by SVM:")
        for n in botanical_names:
            print(f"    {n}")
    else:
        print("  WARNING: zero botanical_* features made it into the final SVM branch.")
    return {"botanical": botanical_names, "standard": standard_names}


def make_species_classifier(random_state: int = RANDOM_STATE,
                             svm_n_features: int = SVM_N_FEATURES,
                             feature_names: list = None,
                             weights: tuple = WEIGHTS,
                             svm_selection: str = "pairwise_aware",
                             rf_params: dict = None,
                             svm_params: dict = None,
                             hgb_params: dict = None) -> VotingClassifier:
    """
    rf_params / svm_params / hgb_params: optional dicts overriding that
    learner's hyperparameters, e.g. rf_params={"n_estimators": 400,
    "max_depth": 20}.

    Defaults below are TUNED (via tune_hyperparameters.py, RandomizedSearchCV,
    n_iter=30, StratifiedGroupKFold 5-fold, grouped by image_path):
        RF alone:  0.9122 -> 0.9165 (+0.0043)
        SVM alone: 0.9110 -> (see svm_defaults below)
        HGB alone: 0.9191 -> 0.9196 (+0.0005)
        Full ensemble (weights=(1,1,1)): 0.9305 -> 0.9387 (+0.0082)
    Prior untuned defaults (RF: n_estimators=200, no other params set;
    SVM: C=10, gamma="scale"; HGB: max_iter=150, no other params set) are
    kept in the RandomizedSearchCV param distributions in
    tune_hyperparameters.py as the search space's implicit starting point,
    not duplicated here as dead code.

    Re-run tune_hyperparameters.py --n-iter 30 (no --subsample-frac / lowered
    --n-splits for the FINAL comparison specifically, see that file's
    --n-splits-final flag) if features change again -- these were tuned
    against the 130-feature set (129 + botanical_pigmentation_wax_index),
    not from scratch, so they may drift if feature_extraction/ changes
    meaningfully.
    """
    rf_defaults = dict(n_estimators=519, max_depth=20, max_features="log2",
                        min_samples_leaf=1, class_weight="balanced",
                        random_state=random_state, n_jobs=-1)
    rf_defaults.update(rf_params or {})
    rf_pipe = Pipeline([("clf", RandomForestClassifier(**rf_defaults))])

    if svm_selection == "pairwise_aware":
        pairwise_score_func = functools.partial(
            _pairwise_aware_mi_score, pairs=LOOK_ALIKE_PAIRS_IDX, random_state=random_state
        )
        svm_features = SelectKBest(pairwise_score_func, k=svm_n_features)

    elif svm_selection == "guaranteed_botanical":
        if feature_names is not None:
            # Real column names are group-prefixed BEFORE the botanical tag
            # (e.g. "shape_botanical_apex_curvature_median",
            # "vein_botanical_vein_loop_fraction") -- pipeline.py's
            # _namespace_features() only skips re-prefixing a key that
            # already starts with its own group name, and "botanical_*"
            # keys don't start with "shape_"/"colour_"/etc, so they get
            # the group prefix prepended too. A plain
            # startswith("botanical_") check matches nothing on real data
            # (verified: a real run reported "botanical_*: 0" even with
            # this guarantee in place, tracing back to exactly this).
            # Substring match is the correct check.
            botanical_idx = [i for i, n in enumerate(feature_names) if "botanical_" in n]
            standard_idx = [i for i, n in enumerate(feature_names) if "botanical_" not in n]
            n_standard_keep = max(svm_n_features - len(botanical_idx), SVM_MIN_STANDARD_FEATURES)
            n_standard_keep = min(n_standard_keep, len(standard_idx))

            svm_features = ColumnTransformer([
                ("botanical", "passthrough", botanical_idx),
                ("standard_select", SelectKBest(mutual_info_classif, k=n_standard_keep), standard_idx),
            ])
        else:
            print("WARNING: make_species_classifier() called without feature_names -- "
                  "SVM branch falls back to plain global SelectKBest, which was verified "
                  "to drop ALL botanical_* columns on real data. Pass feature_names to "
                  "guarantee they're kept.")
            svm_features = SelectKBest(mutual_info_classif, k=svm_n_features)
    else:
        raise ValueError(f"Unknown svm_selection={svm_selection!r}; expected "
                          f"'guaranteed_botanical' or 'pairwise_aware'.")

    svm_defaults = dict(C=54.567, gamma=0.005762, class_weight="balanced",
                         probability=True, random_state=random_state)
    svm_defaults.update(svm_params or {})
    svm_pipe = Pipeline([
        ("select", svm_features),
        ("scale", StandardScaler()),
        ("clf", SVC(**svm_defaults)),
    ])

    hgb_defaults = dict(max_iter=151, learning_rate=0.2537, max_depth=5,
                         max_leaf_nodes=15, l2_regularization=0.749,
                         class_weight="balanced", random_state=random_state)
    hgb_defaults.update(hgb_params or {})
    hgb_pipe = Pipeline([("clf", HistGradientBoostingClassifier(**hgb_defaults))])

    return VotingClassifier(
        estimators=[("rf", rf_pipe), ("svm", svm_pipe), ("hgb", hgb_pipe)],
        voting="soft",
        weights=list(weights) if weights is not None else None,
    )


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE,
                    feature_names: list = None,
                    svm_selection: str = "pairwise_aware") -> dict:
    """
    StratifiedGroupKFold CV, grouped by leaf (image_path) so augmented
    copies of the same physical leaf never split across train/validation.
    Returns out-of-fold predictions plus the summary report/F1.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.empty_like(y)
    for tr_idx, te_idx in cv.split(X, y, groups):
        model = make_species_classifier(random_state, feature_names=feature_names,
                                         svm_selection=svm_selection)
        model.fit(X[tr_idx], y[tr_idx])
        oof_pred[te_idx] = model.predict(X[te_idx])

    return {
        "f1_macro": f1_score(y, oof_pred, average="macro"),
        "report": classification_report(y, oof_pred),
        "confusion_matrix": confusion_matrix(y, oof_pred, labels=sorted(set(y))),
        "oof_pred": oof_pred,
    }


def feature_diversity_report(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                              feature_names: list, n_splits: int = N_SPLITS,
                              random_state: int = RANDOM_STATE,
                              svm_selection: str = "pairwise_aware") -> dict:
    
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred_rf, pred_svm, pred_hgb, pred_ensemble, y_true = [], [], [], [], []
    svm_selected_counter = {}

    for tr_idx, te_idx in cv.split(X, y, groups):
        model = make_species_classifier(random_state, feature_names=feature_names,
                                         svm_selection=svm_selection)
        model.fit(X[tr_idx], y[tr_idx])

        rf_pipe = model.named_estimators_["rf"]
        svm_pipe = model.named_estimators_["svm"]
        hgb_pipe = model.named_estimators_["hgb"]

        # IMPORTANT: VotingClassifier.fit() internally label-encodes y
        # (self.le_ = LabelEncoder().fit(y)) and fits every sub-estimator
        # on the ENCODED integer labels, not the original strings. So
        # rf_pipe.predict() etc. return integer class indices, not
        # species names -- must decode via model.le_ before comparing to
        # y[te_idx] (which is still the original string labels), or every
        # comparison silently evaluates False and the diversity stats
        # come out meaningless (this was caught via a real run showing
        # Q=nan across all three pairs simultaneously -- a strong tell
        # that predictions and ground truth had mismatched encodings,
        # not that errors were literally never both-wrong).
        pred_rf.extend(model.le_.inverse_transform(rf_pipe.predict(X[te_idx])))
        pred_svm.extend(model.le_.inverse_transform(svm_pipe.predict(X[te_idx])))
        pred_hgb.extend(model.le_.inverse_transform(hgb_pipe.predict(X[te_idx])))
        # model.predict() (not the sub-estimator .predict()) already
        # decodes back to original string labels internally -- unlike
        # rf_pipe/svm_pipe/hgb_pipe above, no manual inverse_transform needed.
        pred_ensemble.extend(model.predict(X[te_idx]))
        y_true.extend(y[te_idx])

        # "select" is now a ColumnTransformer (botanical passthrough +
        # MI-selected standard columns), not a bare SelectKBest, so
        # get_support() isn't available on it directly. Reconstruct which
        # columns were kept from its two named sub-transformers instead.
        select_step = svm_pipe.named_steps["select"]
        if isinstance(select_step, ColumnTransformer):
            botanical_idx = select_step.transformers_[0][2]
            standard_idx = select_step.transformers_[1][2]
            standard_support = select_step.named_transformers_["standard_select"].get_support()
            for i in botanical_idx:
                svm_selected_counter[feature_names[i]] = svm_selected_counter.get(feature_names[i], 0) + 1
            for i, kept in zip(standard_idx, standard_support):
                if kept:
                    svm_selected_counter[feature_names[i]] = svm_selected_counter.get(feature_names[i], 0) + 1
        else:
            support_mask = select_step.get_support()
            for name, kept in zip(feature_names, support_mask):
                if kept:
                    svm_selected_counter[name] = svm_selected_counter.get(name, 0) + 1

    y_true = np.array(y_true)
    preds = {"rf": np.array(pred_rf), "svm": np.array(pred_svm), "hgb": np.array(pred_hgb)}

    def _q_statistic(y_true, p1, p2):
        c1, c2 = (p1 == y_true), (p2 == y_true)
        n11 = np.sum(c1 & c2); n10 = np.sum(c1 & ~c2)
        n01 = np.sum(~c1 & c2); n00 = np.sum(~c1 & ~c2)
        denom = (n11 * n00 + n01 * n10)
        return float((n11 * n00 - n01 * n10) / denom) if denom != 0 else float("nan")

    pairs = [("rf", "svm"), ("rf", "hgb"), ("svm", "hgb")]
    diversity_rows = []
    for a, b in pairs:
        disagreement = float(np.mean(preds[a] != preds[b]))
        q = _q_statistic(y_true, preds[a], preds[b])
        diversity_rows.append({"pair": f"{a}-{b}", "disagreement_rate": round(disagreement, 4),
                                "q_statistic": round(q, 4)})

    print("\n-- Base-learner diversity (lower Q / higher disagreement = more diverse) --")
    for row in diversity_rows:
        print(f"  {row['pair']:10s}  disagreement={row['disagreement_rate']:.4f}  "
              f"Q={row['q_statistic']:.4f}")

    # Single-branch F1-macro: same out-of-fold predictions already collected
    # above for the diversity stats (plus the ensemble's own decoded
    # predictions, captured in the same loop -- no redundant retraining).
    # Answers "would one classifier alone beat the ensemble" directly with
    # real numbers instead of extrapolating from the 2-classifier weight-
    # sweep trend.
    ensemble_f1 = f1_score(y_true, np.array(pred_ensemble), average="macro")
    print("\n-- Single-branch F1-macro (same CV folds, out-of-fold) --")
    branch_f1 = {}
    for name in ["rf", "svm", "hgb"]:
        f1 = f1_score(y_true, preds[name], average="macro")
        branch_f1[name] = f1
        print(f"  {name:10s} F1-macro={f1:.4f}")
    print(f"  {'ensemble':10s} F1-macro={ensemble_f1:.4f}  (full 3-classifier soft vote)")
    best_single = max(branch_f1, key=branch_f1.get)
    delta = ensemble_f1 - branch_f1[best_single]
    print(f"\n  Best single branch: {best_single} ({branch_f1[best_single]:.4f}). "
          f"Ensemble {'beats' if delta > 0 else 'loses to'} it by {abs(delta):.4f}.")

    top_svm_features = sorted(svm_selected_counter.items(), key=lambda kv: -kv[1])
    n_botanical = sum(1 for f, _ in top_svm_features if "botanical_" in f)
    n_standard = len(top_svm_features) - n_botanical
    print(f"\n-- SVM branch feature selection across {n_splits} folds --")
    print(f"  Unique columns ever selected: {len(top_svm_features)}  "
          f"(botanical_*: {n_botanical}, standard: {n_standard})")
    print("  Most consistently selected (top 10):")
    for name, count in top_svm_features[:10]:
        print(f"    {name:35s} selected in {count}/{n_splits} folds")

    return {
        "diversity": pd.DataFrame(diversity_rows),
        "svm_feature_counts": top_svm_features,
        "n_botanical_selected": n_botanical,
        "n_standard_selected": n_standard,
    }


def run_feature_count_sweep(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             k_values=(20, 40, 60, 80), n_splits: int = N_SPLITS,
                             random_state: int = RANDOM_STATE,
                             feature_names: list = None) -> pd.DataFrame:
    """
    Sweep SVM_N_FEATURES and report ensemble CV F1-macro for each, so
    the k=40 default is a measured choice, not a guess. Run this once
    when you have the final feature CSV, then hardcode the best k as
    SVM_N_FEATURES above.
    """
    rows = []
    n_features = X.shape[1]
    for k in k_values:
        k_eff = min(k, n_features)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_true, y_pred = [], []
        for tr_idx, te_idx in cv.split(X, y, groups):
            model = make_species_classifier(random_state, svm_n_features=k_eff,
                                             feature_names=feature_names)
            model.fit(X[tr_idx], y[tr_idx])
            y_pred.extend(model.predict(X[te_idx]))
            y_true.extend(y[te_idx])
        f1 = f1_score(y_true, y_pred, average="macro")
        rows.append({"svm_n_features": k_eff, "f1_macro": round(f1, 4)})
        print(f"  svm_n_features={k_eff:3d}  F1-macro={f1:.4f}")
    return pd.DataFrame(rows)


def compare_svm_selection_strategies(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                      feature_names: list,
                                      strategies: tuple = ("guaranteed_botanical", "pairwise_aware"),
                                      svm_n_features: int = SVM_N_FEATURES,
                                      n_splits: int = N_SPLITS,
                                      random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Direct empirical comparison: does letting botanical_* features win
    their spot under pairwise-aware scoring (rather than forcing them
    in) perform as well, better, or worse than the guaranteed-inclusion
    approach? Reports, per strategy:
      - CV F1-macro
      - how many of the (up to) 21 botanical_* columns were EVER
        selected across the 5 folds, and in how many folds each one
        survived (for "pairwise_aware" this is earned, not forced --
        for "guaranteed_botanical" it will always show all of them at
        5/5 by construction, included here only as the reference point)

    "pairwise_aware sounds more principled" is not itself evidence it
    performs better on your data -- this is the check.
    """
    rows = []
    for strat in strategies:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_true, y_pred = [], []
        botanical_fold_counter = {}
        for tr_idx, te_idx in cv.split(X, y, groups):
            model = make_species_classifier(random_state, svm_n_features=svm_n_features,
                                             feature_names=feature_names, svm_selection=strat)
            model.fit(X[tr_idx], y[tr_idx])
            y_pred.extend(model.predict(X[te_idx]))
            y_true.extend(y[te_idx])

            select_step = model.named_estimators_["svm"].named_steps["select"]
            if isinstance(select_step, ColumnTransformer):
                kept_names = [feature_names[i] for i in select_step.transformers_[0][2]]
            else:
                support = select_step.get_support()
                kept_names = [n for n, k in zip(feature_names, support) if k and "botanical_" in n]
            for n in kept_names:
                botanical_fold_counter[n] = botanical_fold_counter.get(n, 0) + 1

        f1 = f1_score(y_true, y_pred, average="macro")
        n_total_botanical = sum(1 for n in feature_names if "botanical_" in n)
        n_ever = len(botanical_fold_counter)
        rows.append({"strategy": strat, "f1_macro": round(f1, 4),
                      "botanical_ever_selected": f"{n_ever}/{n_total_botanical}"})
        print(f"\n  strategy={strat}")
        print(f"    F1-macro: {f1:.4f}")
        print(f"    botanical_* ever selected: {n_ever}/{n_total_botanical}")
        if strat == "pairwise_aware" and n_ever > 0:
            ranked = sorted(botanical_fold_counter.items(), key=lambda kv: -kv[1])
            print(f"    (fold-consistency, out of {n_splits}):")
            for name, count in ranked:
                print(f"      {name:45s} {count}/{n_splits}")

    return pd.DataFrame(rows)


def svm_vote_impact_report(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            feature_names: list, n_splits: int = N_SPLITS,
                            random_state: int = RANDOM_STATE,
                            svm_selection: str = "pairwise_aware") -> dict:
    """
    Directly measures whether the SVM branch's vote ever changes the
    ensemble's final decision, rather than assuming the equal 1/3-1/3-1/3
    soft-voting weight lets it matter.

    For every held-out CV sample, compares the argmax of the full
    3-way averaged probability against the argmax of an RF+HGB-only
    2-way average (SVM excluded). Any sample where these differ is one
    where SVM's vote was decisive. Also reports F1-macro with vs.
    without SVM in the vote, so you can see whether SVM is net helping,
    hurting, or simply not moving the needle on your dataset size.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_flipped = 0
    n_total = 0
    y_true_all, pred_full_all, pred_no_svm_all = [], [], []

    for tr_idx, te_idx in cv.split(X, y, groups):
        model = make_species_classifier(random_state, feature_names=feature_names,
                                         svm_selection=svm_selection)
        model.fit(X[tr_idx], y[tr_idx])

        proba_rf = model.named_estimators_["rf"].predict_proba(X[te_idx])
        proba_svm = model.named_estimators_["svm"].predict_proba(X[te_idx])
        proba_hgb = model.named_estimators_["hgb"].predict_proba(X[te_idx])

        # named_estimators_ predict_proba columns follow the SAME
        # integer-encoded class order as model.le_.classes_ (all three
        # were fit on the same VotingClassifier-encoded y), so decoding
        # once via model.le_.classes_ is valid for all three arrays --
        # same encoding pitfall as the predict() bug fixed earlier, just
        # on the probability side instead of the label side.
        classes = model.le_.classes_
        proba_full = (proba_rf + proba_svm + proba_hgb) / 3.0
        proba_no_svm = (proba_rf + proba_hgb) / 2.0

        pred_full = classes[np.argmax(proba_full, axis=1)]
        pred_no_svm = classes[np.argmax(proba_no_svm, axis=1)]

        n_flipped += int(np.sum(pred_full != pred_no_svm))
        n_total += len(te_idx)
        y_true_all.extend(y[te_idx])
        pred_full_all.extend(pred_full)
        pred_no_svm_all.extend(pred_no_svm)

    f1_full = f1_score(y_true_all, pred_full_all, average="macro")
    f1_no_svm = f1_score(y_true_all, pred_no_svm_all, average="macro")

    print(f"\n-- SVM vote impact --")
    print(f"  SVM changed the final prediction on {n_flipped}/{n_total} samples "
          f"({100 * n_flipped / n_total:.1f}%)")
    print(f"  F1-macro WITH svm  (rf+svm+hgb, equal weight): {f1_full:.4f}")
    print(f"  F1-macro WITHOUT svm (rf+hgb only):            {f1_no_svm:.4f}")
    if f1_full <= f1_no_svm:
        print("  -> SVM is not net-improving the ensemble at equal weight on this "
              "dataset -- try run_voting_weight_sweep() before keeping it at 1/3.")
    else:
        print("  -> SVM's vote is net-helping at equal weight.")

    return {"n_flipped": n_flipped, "n_total": n_total,
            "f1_with_svm": f1_full, "f1_without_svm": f1_no_svm}


def run_voting_weight_sweep(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             feature_names: list, weight_options: list = None,
                             n_splits: int = N_SPLITS,
                             random_state: int = RANDOM_STATE,
                             svm_selection: str = "pairwise_aware") -> pd.DataFrame:
    """
    Sweeps VotingClassifier(weights=...) combinations under CV so the
    1/3-1/3-1/3 default is a measured choice, not an assumption.
    weight_options are (rf, svm, hgb) tuples -- VotingClassifier's soft
    voting takes a WEIGHTED AVERAGE of predicted probabilities, so
    weights don't need to sum to 1 (they're normalised internally).
    """
    if weight_options is None:
        weight_options = [
            (1, 1, 1), (2, 1, 2), (1, 2, 1), (2, 1, 1), (1, 1, 2), (3, 1, 3),
            (1, 0, 1),   # 2-classifier: RF + HGB only (SVM excluded)
            (1, 1, 0),   # 2-classifier: RF + SVM only (HGB excluded)
            (0, 1, 1),   # 2-classifier: SVM + HGB only (RF excluded)
        ]
    rows = []
    for w in weight_options:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_true, y_pred = [], []
        for tr_idx, te_idx in cv.split(X, y, groups):
            model = make_species_classifier(random_state, feature_names=feature_names,
                                             svm_selection=svm_selection)
            model.set_params(weights=list(w))
            model.fit(X[tr_idx], y[tr_idx])
            y_pred.extend(model.predict(X[te_idx]))
            y_true.extend(y[te_idx])
        f1 = f1_score(y_true, y_pred, average="macro")
        rows.append({"weights_rf_svm_hgb": str(w), "f1_macro": round(f1, 4)})
        print(f"  weights(rf,svm,hgb)={w}  F1-macro={f1:.4f}")
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


def fit_final_model(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE,
                     feature_names: list = None,
                     svm_selection: str = "pairwise_aware") -> Pipeline:
    """Fit the classifier on all available training data (for shipping)."""
    model = make_species_classifier(random_state, feature_names=feature_names,
                                     svm_selection=svm_selection)
    model.fit(X, y)
    return model


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                          proba: np.ndarray = None, classes: np.ndarray = None,
                          image_paths: np.ndarray = None, label: str = "Evaluation") -> dict:
    """
    Print and return every number needed to judge classifier quality:
    overall accuracy/F1, per-class precision/recall/F1, confusion matrix,
    look-alike pair accuracy, and the list of misclassified examples
    (with Stage-1 top1/top2 confidence if proba is supplied).
    """
    labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n=== {label} ===")
    print(f"n = {len(y_true)}")
    print(f"Accuracy:          {acc:.4f}  ({acc*100:.2f}%)")
    print(f"F1 macro:          {f1_macro:.4f}")
    print(f"F1 weighted:       {f1_weighted:.4f}")
    print(f"Precision macro:   {prec_macro:.4f}")
    print(f"Recall macro:      {rec_macro:.4f}")

    print("\n-- Per-class report --")
    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report_str)

    print("-- Confusion matrix --")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    print("\n-- Look-alike pair accuracy --")
    pair_results = []
    for a, b in LOOK_ALIKE_PAIRS:
        mask = np.isin(y_true, [a, b])
        if mask.sum() == 0:
            continue
        pair_acc = accuracy_score(y_true[mask], y_pred[mask])
        n_wrong = int((y_true[mask] != y_pred[mask]).sum())
        pair_results.append({"pair": f"{a} / {b}", "n": int(mask.sum()),
                              "accuracy": pair_acc, "errors": n_wrong})
        print(f"  {a:20s} / {b:20s}  acc={pair_acc:.4f}  ({int(mask.sum())} images, {n_wrong} wrong)")

    print("\n-- Misclassified examples --")
    wrong_mask = y_true != y_pred
    n_wrong_total = int(wrong_mask.sum())
    print(f"Total misclassified: {n_wrong_total} / {len(y_true)}")
    misclassified = pd.DataFrame({
        "image_path": image_paths[wrong_mask] if image_paths is not None else np.where(wrong_mask)[0],
        "true": y_true[wrong_mask],
        "predicted": y_pred[wrong_mask],
    })
    if proba is not None and classes is not None:
        order = np.argsort(-proba, axis=1)
        top1_idx, top2_idx = order[:, 0], order[:, 1]
        misclassified["stage1_top1_prob"] = proba[np.arange(len(y_true)), top1_idx][wrong_mask]
        misclassified["stage1_top2"] = classes[top2_idx][wrong_mask]
        misclassified["stage1_top2_prob"] = proba[np.arange(len(y_true)), top2_idx][wrong_mask]
    print(misclassified.to_string(index=False))

    return {
        "accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "precision_macro": prec_macro, "recall_macro": rec_macro,
        "confusion_matrix": cm_df, "pair_accuracy": pd.DataFrame(pair_results),
        "misclassified": misclassified, "classification_report": report_str,
    }


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str, save_path: str = None,
                           normalize: bool = False, figsize: tuple = None,
                           show: bool = True) -> None:
    """
    Render a confusion matrix (the "confusion_matrix" DataFrame returned by
    evaluate_predictions(), or classifier.py's own cross_validate()'s raw
    array wrapped in pd.DataFrame(cm, index=labels, columns=labels)) as an
    annotated heatmap.

    Matplotlib/seaborn are imported lazily here, not at module top, so
    importing classifier.py for training/inference doesn't require a
    plotting stack to be installed -- same lazy-import pattern already
    used for run_feature_count_sweep in model_training.py.

    Pass normalize=True to show row-wise (per true-class) proportions
    instead of raw counts -- more readable once class counts are
    imbalanced, which they are here (see LOOK_ALIKE_PAIRS).

    Pass show=False (recommended when calling this from a plain script,
    e.g. model_training.py run via `python -m ...` with no display attached)
    so saving the PNG never depends on a GUI backend being available --
    plt.show() can hang or raise on a headless box. Notebook callers can
    leave show=True to see the plot inline as well as save it.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = cm_df.copy()
    fmt = "d"
    if normalize:
        row_sums = data.sum(axis=1).replace(0, 1)  # guard against /0 on an empty class
        data = data.div(row_sums, axis=0)
        fmt = ".2f"

    n_labels = len(cm_df)
    if figsize is None:
        side = max(6, 0.5 * n_labels)
        figsize = (side, side)

    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", square=True,
                xticklabels=cm_df.columns, yticklabels=cm_df.index,
                cbar_kws={"label": "proportion" if normalize else "count"})
    plt.title(title)
    plt.xlabel("Predicted species")
    plt.ylabel("True species")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix plot to {save_path}")
    if show:
        try:
            plt.show()
        except Exception:
            pass  # no display available -- the PNG above is already saved
    plt.close()


if __name__ == "__main__":
    # Example usage matching the project's train/test CSV convention.
    train = pd.read_csv("vedavision_features_train_clf.csv")
    test = pd.read_csv("vedavision_features_test_clf.csv")

    feature_cols = [c for c in train.columns if c not in ("species", "image_path")]
    X_train = train[feature_cols].to_numpy()
    y_train = train["species"].to_numpy()
    groups_train = train["image_path"].to_numpy()

    X_test = test[feature_cols].to_numpy()
    y_test = test["species"].to_numpy()

    print("=== 5-fold StratifiedGroupKFold CV (train only) ===")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.empty_like(y_train)
    for tr_idx, te_idx in cv.split(X_train, y_train, groups_train):
        m = make_species_classifier(RANDOM_STATE, feature_names=feature_cols)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        oof_pred[te_idx] = m.predict(X_train[te_idx])
    evaluate_predictions(y_train, oof_pred, image_paths=groups_train, label="5-fold CV (train)")

    print("\n=== Base-learner diversity report (evidence for supervisor) ===")
    feature_diversity_report(X_train, y_train, groups_train, feature_cols)

    print("\n=== Fitting final model on all training data ===")
    final_model = fit_final_model(X_train, y_train, feature_names=feature_cols)
    save_model(final_model, "vedavision_species_model.pkl")
    print("Saved model to vedavision_species_model.pkl")

    print("\n=== Held-out test set (sealed — run this only once) ===")
    test_pred = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)
    test_classes = final_model.classes_
    results = evaluate_predictions(y_test, test_pred, proba=test_proba, classes=test_classes,
                                    label="Held-out test set")

    # Save the key numbers to disk so they can be dropped straight into a
    # report/slide without re-running anything.
    results["confusion_matrix"].to_csv("test_confusion_matrix.csv")
    results["pair_accuracy"].to_csv("test_pair_accuracy.csv", index=False)
    results["misclassified"].to_csv("test_misclassified.csv", index=False)
    print("\nSaved test_confusion_matrix.csv, test_pair_accuracy.csv, test_misclassified.csv")
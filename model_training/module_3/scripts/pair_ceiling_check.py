"""
VedaVision -- Kattakumanjal vs Kalawal: Feature-Sufficiency Ceiling Check
===========================================================================
The pairwise feature audit (analyze_pair_features.py) found moderate
per-feature separation (best AUC 0.787, texture_lbp_08) but no single
strong feature, clustered into ~3 groups: texture (LBP bins), vein
structure (density/branching/length), and colour (lab_b, hsv_s), plus 2
botanical hits (gloss ratio, oil gland density). None of these appear in
the SVM branch's consistently-selected columns, which are all shape_*.

That leaves two very different possible explanations for the sealed-test
confusion on this pair, needing opposite fixes:
  (a) The classifier isn't actually using the texture/vein/colour signal
      that exists (shape_* crowding it out of SVM's 40-column budget,
      or RF/HGB just not weighting it heavily for this pair even though
      they technically see all columns) -- fixable via feature selection
      / LOOK_ALIKE_PAIRS targeting, no new data or features needed.
  (b) Even combined, these moderate signals genuinely don't separate the
      two species well enough -- needs a new feature or more data, not a
      classifier change.

This script settles which one it is: train a SMALL, DEDICATED 2-class
model using ONLY kattakumanjal/kalawal rows and ONLY the current top-N
audit features (default: top 15, i.e. roughly the texture+vein+colour+
botanical cluster, deliberately excluding the shape_* features that
dominate the full 12-class SVM selection but rank weakly for this pair).

If this dedicated model's CV accuracy is clearly better than the full
12-class ensemble's current sealed-test performance on this pair
(kattakumanjal: 0.83 precision / 0.75 recall; kalawal: 0.74 / 0.85 --
i.e. an F1 around 0.79-0.81 for each), that's explanation (a): the signal
was there, the full ensemble just wasn't using it well for this pair.
If the dedicated model tops out around the same ~0.80 F1, that's
explanation (b): genuinely at the ceiling of current features, new
feature design (or more training data for this pair) is the only path.

Usage
-----
    python pair_ceiling_check.py \
        --train processed/features/vedavision_features_train_clf.csv \
        --species-a kattakumanjal --species-b kalawal \
        --audit-csv kattakumanjal_kalawal_audit.csv \
        --top-n 15

--audit-csv is optional: if given, feature ranking comes straight from
analyze_pair_features.py's output (already-computed separating_power, no
need to recompute). If omitted, this script re-derives the ranking itself.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score

SENTINEL = -1.0
NON_FEATURE_COLS = ["species", "image_path"]

# Reference numbers from the current full 12-class ensemble's sealed test
# (run this script's numbers against THESE, not against overall accuracy --
# the full ensemble solves a harder 12-way problem, so a dedicated 2-class
# model should do at least this well just from having an easier task,
# REGARDLESS of whether explanation (a) or (b) holds. The real question is
# how much BETTER than this the dedicated model gets.)
CURRENT_SEALED_TEST_F1 = {"kattakumanjal": 0.79, "kalawal": 0.79}  # ~ from precision/recall given


def rank_features_from_audit(audit_csv: str, top_n: int) -> list:
    audit = pd.read_csv(audit_csv)
    audit = audit.sort_values("separating_power", ascending=False)
    audit = audit[~audit["unreliable_high_sentinel"]]
    return audit["feature"].head(top_n).tolist()


def rank_features_inline(df: pd.DataFrame, species_a: str, species_b: str, top_n: int) -> list:
    # Minimal re-derivation (Cohen's d only) if no audit CSV is passed --
    # for full AUC + sentinel handling, prefer analyze_pair_features.py.
    sub = df[df["species"].isin([species_a, species_b])]
    y = (sub["species"] == species_a).astype(int).values
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    scores = []
    for col in feat_cols:
        vals = sub[col].values.astype(float)
        a, b = vals[y == 1], vals[y == 0]
        a, b = a[a != SENTINEL], b[b != SENTINEL]
        if len(a) < 2 or len(b) < 2:
            continue
        pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = abs((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0
        scores.append((col, d))
    scores.sort(key=lambda x: -x[1])
    return [c for c, _ in scores[:top_n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--species-a", required=True)
    ap.add_argument("--species-b", required=True)
    ap.add_argument("--audit-csv", default=None,
                     help="Output of analyze_pair_features.py --out-csv. If given, feature "
                          "ranking (incl. sentinel filtering) comes from there. Recommended.")
    ap.add_argument("--top-n", type=int, default=15,
                     help="How many top-separating features the dedicated model gets. Kept "
                          "small deliberately -- this is a signal-sufficiency check, not a "
                          "'give the model everything' run.")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    if "image_path" not in df.columns:
        raise ValueError("Need image_path column for GroupKFold -- pass the TRAIN clf CSV.")

    if args.audit_csv:
        top_features = rank_features_from_audit(args.audit_csv, args.top_n)
    else:
        top_features = rank_features_inline(df, args.species_a, args.species_b, args.top_n)

    print(f"Using top {len(top_features)} features:")
    for f in top_features:
        print(f"  {f}")

    sub = df[df["species"].isin([args.species_a, args.species_b])].copy()
    # Replace sentinels with per-column median (of non-sentinel values in this
    # subset) -- simple, defensible imputation for a diagnostic check; NOT
    # meant to be the production sentinel-handling strategy.
    X = sub[top_features].copy()
    for col in top_features:
        non_sentinel = X.loc[X[col] != SENTINEL, col]
        fill_val = non_sentinel.median() if len(non_sentinel) else 0.0
        X[col] = X[col].replace(SENTINEL, fill_val)
    X = X.values
    y = sub["species"].values
    groups = sub["image_path"].values

    n_groups = len(set(groups))
    n_splits = min(args.n_splits, n_groups)
    if n_splits < 2:
        raise ValueError(f"Only {n_groups} unique image_path groups for this pair -- "
                          "not enough for cross-validation.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)

    models = {
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=args.random_state, n_jobs=-1)),
        ]),
        "SVM-RBF": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=10, gamma="scale", class_weight="balanced",
                        random_state=args.random_state)),
        ]),
    }

    print(f"\n=== Dedicated 2-class ceiling check: {args.species_a} vs {args.species_b} ===")
    print(f"(GroupKFold n_splits={n_splits}, using ONLY the {len(top_features)} features above)\n")

    for name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        f1_macro = f1_score(y, y_pred, average="macro")
        print(f"--- {name} ---")
        print(f"F1-macro: {f1_macro:.4f}")
        print(classification_report(y, y_pred, digits=3))
        for species in [args.species_a, args.species_b]:
            f1_this = f1_score(y == species, y_pred == species)
            ref = CURRENT_SEALED_TEST_F1.get(species)
            if ref is not None:
                delta = f1_this - ref
                verdict = ("BETTER -- signal exists, full ensemble underusing it" if delta > 0.05
                            else "SIMILAR -- close to current ceiling, features likely insufficient" if abs(delta) <= 0.05
                            else "WORSE -- current model already extracting more than this feature subset alone")
                print(f"  {species}: dedicated F1={f1_this:.3f} vs full-ensemble sealed-test F1~{ref:.3f} "
                      f"(delta={delta:+.3f}) -> {verdict}")
        print()

    print("Interpretation guide:")
    print("  - If BOTH models clear ~0.85-0.90+ F1 here: explanation (a) -- the moderate")
    print("    per-feature signal COMBINES well, the full 12-class ensemble just isn't")
    print("    routing it to this pair. Fix: force these columns into SVM's selected set")
    print("    (e.g. lower SVM_N_FEATURES budget spent on shape_*, or add a guaranteed-slot")
    print("    mechanism like guaranteed_botanical but for LOOK_ALIKE_PAIRS-specific columns),")
    print("    and check RF/HGB feature_importances_ for this pair specifically.")
    print("  - If both models stay around ~0.75-0.82 F1, close to the current ensemble: ")
    print("    explanation (b) -- genuinely at the ceiling of current features for this pair.")
    print("    New feature design (grounded in what kattakumanjal/kalawal actually differ in")
    print("    botanically -- pigmentation/oil-gland/texture, per the audit) or more training")
    print("    images for these two species is the needed next step, not classifier tuning.")


if __name__ == "__main__":
    main()
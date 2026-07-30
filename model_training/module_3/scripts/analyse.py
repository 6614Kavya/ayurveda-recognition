"""
VedaVision -- Species Identification: Pairwise Feature Separability Audit
===========================================================================
Answers one question directly: of the 129 features currently extracted,
which ones actually separate kattakumanjal from kalawal, and how well?

Why this script, now
---------------------
Three different ensemble configurations (plain flat ensemble, HGB with a
SelectKBest column-cut, HGB wrapped in BaggingClassifier) were tried in
sequence to fix the kattakumanjal/kalawal confusion (dominant error in the
sealed test set -- 7/15 = 47% of all mistakes). All three still show this
pair as the worst-confused of the 12 species, regardless of voting/
weighting/diversity mechanics. That's decisive: no ensemble trick can
out-vote a signal that isn't in the features to begin with. This script
stops guessing and measures per-feature separability directly, so any new
handcrafted feature work is aimed at a confirmed gap instead of a hunch.

What it does
------------
1. Loads the TRAIN clf CSV (use train, not the sealed test -- this is
   diagnostic work, not evaluation; keep the sealed test untouched).
2. Filters to just the two classes given via --species-a/--species-b.
3. For every feature column, computes:
     - Cohen's d (standardised mean difference) -- effect size, robust to
       scale differences across features.
     - ROC-AUC treating the feature as a single-feature classifier for
       "is this species-a vs species-b" -- 0.5 = no separation at all,
       1.0 (or 0.0) = perfect separation. AUC is used alongside Cohen's d
       because it's monotonic-invariant (catches features where the two
       classes' distributions barely overlap but aren't simply mean-shifted,
       e.g. one class is much more variable than the other).
   Sentinel values (-1.0, the project's documented placeholder for failed
   extraction) are excluded from BOTH classes' stats for a feature before
   scoring it, so a feature that failed to extract for many leaves doesn't
   look artificially separating.
4. Prints a ranked table, best-separating features first, split into
   botanical_* vs standard so you can see at a glance whether the
   botanical set is pulling its weight on this specific pair or not.
5. Flags any feature where BOTH classes have >30% sentinel rows -- these
   are unreliable for this pair regardless of their apparent AUC and
   should not be trusted as a fix.

Usage
-----
    python analyze_pair_features.py \
        --train processed/features/vedavision_features_train_clf.csv \
        --species-a kattakumanjal --species-b kalawal

Interpreting the output
------------------------
- If the top features show AUC well above ~0.75-0.80, there IS usable
  signal already in the feature set -- the problem may be more about how
  the classifier weighs/combines it (e.g. worth re-checking
  SVM_N_FEATURES / LOOK_ALIKE_PAIRS coverage for this pair specifically)
  than about missing features.
- If the top features top out around AUC ~0.55-0.65 (barely above chance),
  that confirms none of the 129 columns meaningfully separate this pair --
  the fix has to be a genuinely new feature, not a reweighting of existing
  ones. In that case, go look at what the two species actually differ in
  botanically (leaflet margin texture? rachis pigmentation -- recall
  kattakumanjal needed the Tier-2 dark-pigment seed fallback in masking.py?
  leaflet arrangement angle?) and design a feature around THAT specific
  trait, rather than adding another generic descriptor.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SENTINEL = -1.0
NON_FEATURE_COLS = ["species", "image_path"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_std == 0 or np.isnan(pooled_std):
        return np.nan
    return (a.mean() - b.mean()) / pooled_std


def audit_pair(df: pd.DataFrame, species_a: str, species_b: str, sentinel_frac_flag: float = 0.30):
    sub = df[df["species"].isin([species_a, species_b])].copy()
    if sub.empty:
        raise ValueError(f"No rows found for {species_a!r} or {species_b!r} -- check spelling "
                          f"against df['species'].unique(): {sorted(df['species'].unique())}")
    y = (sub["species"] == species_a).astype(int).values  # 1 = species_a, 0 = species_b
    n_a, n_b = int(y.sum()), int((1 - y).sum())
    print(f"{species_a}: {n_a} rows   {species_b}: {n_b} rows\n")

    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    rows = []
    for col in feat_cols:
        vals = sub[col].values.astype(float)
        a_vals_raw = vals[y == 1]
        b_vals_raw = vals[y == 0]

        a_sentinel_frac = float(np.mean(a_vals_raw == SENTINEL)) if len(a_vals_raw) else np.nan
        b_sentinel_frac = float(np.mean(b_vals_raw == SENTINEL)) if len(b_vals_raw) else np.nan

        a_vals = a_vals_raw[a_vals_raw != SENTINEL]
        b_vals = b_vals_raw[b_vals_raw != SENTINEL]

        if len(a_vals) < 2 or len(b_vals) < 2:
            continue  # not enough non-sentinel data to score this feature for this pair

        d = cohens_d(a_vals, b_vals)

        # AUC needs one array of values + one array of binary labels, built
        # only from the non-sentinel rows (sentinel rows already dropped above).
        auc_vals = np.concatenate([a_vals, b_vals])
        auc_labels = np.concatenate([np.ones(len(a_vals)), np.zeros(len(b_vals))])
        try:
            auc = roc_auc_score(auc_labels, auc_vals)
        except ValueError:
            auc = np.nan
        # AUC < 0.5 just means the feature separates in the opposite
        # direction -- report distance from 0.5 (separating power),
        # not raw AUC, so "0.15" and "0.85" both rank as strongly separating.
        separating_power = abs(auc - 0.5) + 0.5 if not np.isnan(auc) else np.nan

        rows.append({
            "feature": col,
            "group": "botanical" if "botanical_" in col else "standard",
            "cohens_d": round(d, 3) if not np.isnan(d) else np.nan,
            "auc": round(auc, 3) if not np.isnan(auc) else np.nan,
            "separating_power": round(separating_power, 3) if not np.isnan(separating_power) else np.nan,
            f"{species_a}_sentinel_frac": round(a_sentinel_frac, 2),
            f"{species_b}_sentinel_frac": round(b_sentinel_frac, 2),
            "unreliable_high_sentinel": (a_sentinel_frac > sentinel_frac_flag
                                         or b_sentinel_frac > sentinel_frac_flag),
        })

    result = pd.DataFrame(rows).sort_values("separating_power", ascending=False).reset_index(drop=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="vedavision_features_train_clf.csv")
    ap.add_argument("--species-a", required=True)
    ap.add_argument("--species-b", required=True)
    ap.add_argument("--top", type=int, default=25, help="How many top-separating features to print")
    ap.add_argument("--sentinel-flag-threshold", type=float, default=0.30,
                     help="Flag a feature as unreliable if either class has more than this "
                          "fraction of sentinel (-1.0) rows for it.")
    ap.add_argument("--out-csv", default=None,
                     help="Optional: write the full ranked table to this path for inspection "
                          "in Excel/pandas rather than just the console top-N.")
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    result = audit_pair(df, args.species_a, args.species_b, args.sentinel_flag_threshold)

    print(f"=== Top {args.top} features separating {args.species_a} vs {args.species_b} ===")
    print("(AUC=0.5 -> no separation at all. separating_power=1.0 -> perfect separation, either direction.)\n")
    print(result.head(args.top).to_string(index=False))

    n_botanical_in_top = (result.head(args.top)["group"] == "botanical").sum()
    print(f"\n{n_botanical_in_top}/{args.top} of the top-separating features are botanical_*.")

    unreliable = result[result["unreliable_high_sentinel"]]
    if not unreliable.empty:
        print(f"\nWARNING: {len(unreliable)} feature(s) in the full ranking have >"
              f"{args.sentinel_flag_threshold:.0%} sentinel rows for one or both species -- "
              "treat their separating_power as unreliable even if it looks high:")
        print(unreliable[["feature", "separating_power",
                           f"{args.species_a}_sentinel_frac",
                           f"{args.species_b}_sentinel_frac"]].to_string(index=False))

    best_auc = result["separating_power"].max()
    if best_auc < 0.65:
        print(f"\n>>> Best separating_power across ALL {len(result)} features is only {best_auc:.3f} "
              f"(chance = 0.5). None of the current features meaningfully separate this pair -- "
              f"this points to a genuinely missing feature, not a classifier/weighting issue.")
    elif best_auc < 0.80:
        print(f"\n>>> Best separating_power is {best_auc:.3f} -- weak-to-moderate signal exists "
              f"but no single feature is strongly discriminative on its own. Worth checking "
              f"whether the classifier is actually using these columns for this pair "
              f"(SVM_N_FEATURES budget, LOOK_ALIKE_PAIRS coverage) before assuming new features "
              f"are required.")
    else:
        print(f"\n>>> Best separating_power is {best_auc:.3f} -- strong signal already exists in "
              f"the feature set for this pair. If the classifier still confuses them, the issue "
              f"is more likely how that signal is being weighted/combined, not a missing feature.")

    if args.out_csv:
        result.to_csv(args.out_csv, index=False)
        print(f"\nFull ranked table written to {args.out_csv}")


if __name__ == "__main__":
    main()
"""
models/health/train_stage1_binary.py

Trains and saves ONLY the Stage-1 healthy/unhealthy classifier -- the
active binary decision-maker for the health branch, run alongside (not
replaced by) the continuous VedaVision Health Index from
train_health_index.py.

Split out from legacy_flat_classifier.py (kept as a documented negative-
result record for stage2/severity, F1-macro ~0.43-0.58). Stage 1 alone
was never the problem -- F1=0.854 on held-out test -- so it's promoted
here into its own script.

--- FIX (this session): do NOT call TwoStageHealthClassifier.fit() ---
That wrapper's .fit() ALWAYS fits Stage 2 internally regardless of
whether the caller wants it, by design (it's meant to be called with a
real y_severity, e.g. "moderate"/"high", only on unhealthy rows). This
script has no severity target -- passing y_binary in as a stand-in
y_severity meant every "unhealthy" row had the identical label
"unhealthy", i.e. Stage 2's SVM saw exactly one class and crashed
("number of classes has to be greater than one"). Fixed by building and
fitting the Stage-1 pipeline directly (see MODEL CHOICE below), and
reproducing the same OOF threshold-tuning logic classifier.py uses,
without ever touching Stage 2.

--- MODEL CHOICE (restored this session, via compare_stage1_models.py) ---
RF vs HistGradientBoosting vs SVM-RBF vs ExtraTrees, run on this exact
feature set across the identical 5 leaf-grouped folds. SVM-RBF won on
both axes:
    CV mean f1_macro:  svm_rbf 0.859 (+/-0.017)  vs  random_forest 0.810 (+/-0.019)
    held-out test f1:  svm_rbf 0.885             vs  random_forest 0.870
SVM hyperparameters (C=10, gamma='scale') match the species-ID
ensemble's SVM leg -- already justified there, nothing new to defend.
Built LOCALLY here (not via TwoStageHealthClassifier._build_stage1(),
which is RF and shared with Stage 2's deployment config -- Stage 2
keeps its own RF choice untouched) so this script can pick the better
Stage-1-only model without disturbing classifier.py at all.
random_forest kept available via --model for the comparison record /
in case a future larger (augmented) dataset changes the ranking.

--- FEATURE UPGRADE (this session, driven by gap_report.csv /
elevated_feature_counts.csv from diagnose_feature_gaps.py) ---
Held-out test was f1_macro=0.8491 (86 healthy / 108 unhealthy, 13 FP +
16 FN). Three changes, all sourced directly from the diagnostic run
rather than guessed:

  1. DEAD_FEATURES dropped entirely (top_/bottom_/worst_ never built for
     them). Confirmed via gap_report.csv: boundary_notch_density,
     deform_specular_pct and deform_specular_blob_density were EXACTLY
     0.0 for every species at every damage level -- not "weak", the
     detectors never fire at all under this dataset's lighting/imaging
     conditions. Pure noise columns; only add split-quality dilution to
     the RF, never signal. (miner_trail_* was ALSO weak in 10-11/12
     species but is NOT dropped -- it shows real, rising-with-severity
     signal for kattakumanjal/kalawal, i.e. genuinely species-specific
     absence of that damage mode elsewhere, not a broken detector.)

  2. Species-relative robust-z features (z_<feature>) added for every
     surviving CANDIDATE_FEATURES column, using each species' OWN
     healthy-leaf median/IQR (TRAIN-derived only, never test) as the
     reference point instead of a global raw value + species one-hot
     dummy. diagnose_feature_gaps.py already computed this exact
     statistic for its diagnostic report; this promotes it from
     diagnostic-only output into an actual classifier input.

  3. n_features_elevated_z<T> added at T in {1.0, 1.5, 2.0} -- how many
     candidate features sit >= T robust-z from that leaf's own species'
     healthy baseline. elevated_feature_counts.csv confirmed a real
     monotonic median trend (healthy=1 -> low=3 -> mid=3 -> high=5) that
     no single raw feature gives as cleanly on its own -- this directly
     encodes the "single-dominant-symptom dilution" hypothesis as a
     feature instead of leaving it as an unused diagnostic printout.

--- SPOT FEATURES ADDED (this session, follow-up) ---
spots.py's discrete lesion features (spot_count, spot_area_ratio,
spot_density_per_1000px, spot_mean_size, necrotic_spot_count) are now
registered in diagnose_feature_gaps.py's CANDIDATE_FEATURES, so they
automatically flow through Z_FEATURE_COLS below with no changes needed
in this file -- the whole z-scoring/elevated-count pipeline is column-
name driven. Before retraining:
  1. Re-run preprocessing/health/batch_processor.py to regenerate
     health_features_top.csv / _bottom.csv with the restored spots.py.
  2. Re-run this script and compare against the existing baseline
     (CV f1_macro=0.8096, held-out test f1_macro=0.87) -- check
     specifically whether thunpath_kurundu/wal_bilin misclassifications
     (both had strong spot-feature effect sizes once the bugs were
     fixed -- 3.67 / 2.05 on spot_area_ratio, per validate_spot_features.py)
     flip to correct.

--- FIX (this session): spot_rachis_guard_triggered no longer leaks in ---
It's a per-leaf QC flag (spots.py's docstring), not a health signal.
diagnose_feature_gaps.py already excluded it from CANDIDATE_FEATURES,
but NON_FEATURE_COLS in model_training.py -- the actual thing
_fuse_leaves() below filters raw columns against -- didn't know about
it, so it was silently flowing into X as
top_/bottom_/worst_spot_rachis_guard_triggered. Fixed at the source
(model_training.py's NON_FEATURE_COLS), not here, so every script that
fuses raw columns gets the same exclusion from one place.

Baseline leakage discipline: species healthy median/IQR are refit from
EACH CV fold's training partition only (not from the whole train set)
during cross-validation, and refit once more on all of train for the
final model. Test rows are z-scored against the FINAL train-fit
baselines -- test never contributes to its own normalisation.

Dataset requirement: NONE. Uses the same low/mid/high/healthy folder
structure and CSVs as everything else -- the low/mid/high -> "unhealthy"
collapse happens here in code (build_binary_dataset()), not in the
dataset. Do not delete or merge the low/mid/high folders/labels.

Run from module_3/ root:

    D:\\Python313\\python.exe -m models.health.train_stage1_binary
    D:\\Python313\\python.exe -m models.health.train_stage1_binary --model random_forest
"""
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.health.classifier import TwoStageHealthClassifier, fuse_top_bottom
from models.health.model_training import (
    load_and_merge_from, split_train_test, TOP_CSV, BOTTOM_CSV, NON_FEATURE_COLS,
)
from models.health.diagnose_feature_gaps import CANDIDATE_FEATURES

MODEL_OUT = "processed/models/vedavision_stage1_svm_model.pkl"

EPS = 1e-6

# See MODEL CHOICE note above. svm_rbf is the default because it's the
# empirically better Stage-1 model on this dataset; random_forest is
# kept for the comparison record / re-checking the ranking on future,
# larger data. Deliberately built LOCALLY (not via
# TwoStageHealthClassifier._build_stage1()) so choosing a model here
# never touches Stage 2's own RF config in classifier.py.
STAGE1_MODEL = "svm_rbf"

MODEL_BUILDERS = {
    "svm_rbf": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
    ]),
    "random_forest": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    ]),
}

# Confirmed dead in gap_report.csv (this session) -- exactly 0.0 for
# every species at every level. See module docstring point (1). If a
# future re-run of diagnose_feature_gaps.py shows these have started
# showing real variance (e.g. after a masking/lighting recalibration),
# remove the relevant name here rather than leaving it dropped forever.
DEAD_FEATURES = {
    "boundary_notch_density",
    "deform_specular_pct",
    "deform_specular_blob_density",
    # Added this session: deformation.py's own module docstring flags all
    # of its features as "principled proxies, not verified signal" --
    # width_profile_roughness and luminance_std are the two that AREN'T
    # already dead in the zero-variance sense (unlike specular_pct/
    # blob_density above), but train_health_index.py's own per-subscore
    # Spearman check (run earlier, independent of this file) already
    # showed both essentially uncorrelated with severity:
    #   worst_deform_width_profile_roughness  rho=0.000  p=0.999
    #   worst_deform_luminance_std             rho=0.049  p=0.232 (n.s.)
    # gap_report.csv additionally shows both flipping sign inconsistently
    # across species (e.g. width_profile_roughness: ranawara +0.51 vs
    # kattakumanjal -0.42) -- the signature of an incidental per-species
    # lighting/shape artifact rather than a real damage signal, not
    # something the species-relative z-scoring can rescue.
    "deform_width_profile_roughness",
    "deform_luminance_std",
    # Removed this session (deliberate trade-off, not a diagnostic
    # finding of "always dead" -- see note where this list is used):
    # miner_trail.py's colour/tortuosity gate (pale, desaturated,
    # winding interior pixels) has no requirement that distinguishes an
    # actual leaf-miner tunnel from a naturally winding, slightly-paler
    # leaf VEIN catching a highlight under the studio lighting -- the
    # module's own docstring admits the thresholds were never
    # recalibrated against confirmed trail crops. gap_report.csv showed
    # nonzero miner_trail_count on a large fraction of HEALTHY leaves
    # across several species (e.g. 8/30 ranawara, several kalawal/
    # kattakumanjal/kathurupila), consistent with vein false positives
    # rather than real damage. NOTE: kalawal (effect +1.0) and
    # kattakumanjal (+0.5) DID show a real, rising-with-severity signal
    # here -- dropping this wholesale trades away that genuine detection
    # for those two species specifically, in exchange for removing the
    # apparent false-positive noise everywhere else. If miner_trail.py's
    # detection gate is later tightened/recalibrated against real
    # confirmed trail crops, remove these four from DEAD_FEATURES again
    # rather than leaving them dropped forever.
    "miner_trail_length_norm",
    "miner_trail_coverage_pct",
    "miner_trail_mean_tortuosity",
    "miner_trail_count",
    # ldsi_miner_sub (severity_index.py) is computed DIRECTLY from
    # miner_trail_coverage_pct and miner_trail_length_norm above -- same
    # detector, repackaged as an LDSI subscore. Dropping the raw columns
    # but keeping this would silently let the same false-positive-prone
    # signal back into the model under a different name.
    "ldsi_miner_sub",
}

# Species-relative z-scores / elevated-counts are computed over the same
# candidate list diagnose_feature_gaps.py uses, minus whatever's dead.
Z_FEATURE_COLS = [c for c in CANDIDATE_FEATURES if c not in DEAD_FEATURES]

# Multiple thresholds kept (not just 1.5) since it's cheap and lets the
# model pick whichever granularity actually helps via feature importance
# (RF) or simply by being available as a feature (SVM), rather than us
# pre-committing to one z cutoff.
ELEVATED_Z_THRESHOLDS = [1.0, 1.5, 2.0]

SpeciesBaselines = Dict[str, Dict[str, Tuple[float, float]]]

# Species where a given feature's z-score/elevated-count contribution
# should be suppressed (baseline forced to NaN, same as the existing
# <3-healthy-leaves case) because that species' damage physically
# doesn't match what the feature is built to detect -- feeding it in
# anyway trains the model on backwards or contradictory signal rather
# than just weak/absent signal.
#
# siyambala + spot_*: confirmed via gap_report.csv (effect_size -1.41 to
# -1.53 on spot_count/necrotic_spot_count -- MORE spots on healthy
# leaves than high-damage ones) AND via direct visual inspection of real
# siyambala "high" leaves (this session): damage presents as diffuse
# mottled/marbled patches and marginal tip-browning spanning large
# regions, NOT discrete circular lesions. spots.py's connected-component
# blob model structurally doesn't match this damage morphology -- a
# continuous mottled region either fails to clear the absolute colour
# thresholds (low-contrast pattern, not a strong hue/lightness
# deviation) or collapses into one/few large blobs, while a genuinely
# healthy leaf's incidental tiny natural blemishes each register as
# their own small blob and inflate count on the HEALTHY side instead.
# This is a documented species-specific limitation, not a bug to chase
# further -- keep the raw worst_/top_/bottom_ spot_* columns in X
# (species one-hot still lets the model learn a species-conditional
# split on them if it wants to), only suppress the ENGINEERED z_/
# elevated-count contribution, since that aggregate column can't itself
# express "this direction is reversed for one species."
FEATURE_SPECIES_EXCLUSIONS: Dict[str, set] = {
    "spot_count": {"siyambala", "kathurupila"},
    "necrotic_spot_count": {"siyambala", "kathurupila"},
    "spot_area_ratio": {"siyambala", "kathurupila"},
    "spot_mean_size": {"siyambala", "kathurupila"},
    # spot_density_per_1000px was missing from this dict entirely even
    # for siyambala, despite showing siyambala's STRONGEST reversal of
    # all five spot metrics (effect_size -0.99 in gap_report.csv) -- an
    # oversight from when this dict was first built, not a deliberate
    # decision to keep it in. Filled in now for both species.
    #
    # kathurupila added (this session) on the same evidence standard as
    # siyambala: gap_report.csv shows a reversal (more spots on healthy
    # leaves than "high") across ALL FIVE spot metrics, not just one --
    # spot_count -0.46, necrotic_spot_count -0.47, spot_area_ratio -0.13,
    # spot_mean_size -0.38, spot_density_per_1000px -0.58. Same
    # structural explanation applies as siyambala: spots.py's connected-
    # component blob model assumes discrete circular lesions, and doesn't
    # match kathurupila's real damage morphology -- confirm with a direct
    # look at a few real kathurupila "high" leaves before citing in the
    # dissertation, same caveat as every other threshold in this file.
    #
    # kattakumanjal shows the SAME direction on all five metrics too
    # (spot_count -0.17, spot_area_ratio -0.32, spot_mean_size -0.44,
    # necrotic_spot_count -0.12, spot_density_per_1000px -0.32) but at
    # roughly half kathurupila's magnitude -- NOT added here yet.
    # Borderline: could be the same structural mismatch at a milder
    # degree, or could be real (if kattakumanjal's healthy leaves
    # naturally show more small blemishes than its "high" leaves do
    # large ones). Check a handful of real kattakumanjal "high" leaves
    # before deciding either way; don't add blind.
    "spot_density_per_1000px": {"siyambala", "kathurupila"},
}


def _fuse_leaves(top: pd.DataFrame, bottom: pd.DataFrame) -> pd.DataFrame:
    """One row per leaf: top_<f>/bottom_<f>/worst_<f> for every raw
    feature column EXCEPT DEAD_FEATURES (dropped before fusion ever
    happens, so those columns never exist downstream at all), plus
    species + level. level is kept as the raw folder label at this
    stage (not yet collapsed to healthy/unhealthy) because it's needed
    to select each species' healthy subset for baselining -- the
    binary target is derived from it later in build_binary_dataset().
    NON_FEATURE_COLS (model_training.py) already excludes
    spot_rachis_guard_triggered -- a QC flag, not a health signal.
    """
    feature_cols = [c for c in top.columns if c not in NON_FEATURE_COLS and c not in DEAD_FEATURES]
    rows = []
    for sample_id in top.index:
        fused = fuse_top_bottom(top.loc[sample_id].to_dict(), bottom.loc[sample_id].to_dict(), feature_cols)
        fused["species"] = top.loc[sample_id, "species"]
        fused["level"] = top.loc[sample_id, "level"]
        rows.append(fused)
    return pd.DataFrame(rows, index=top.index)


def _drop_raw_side_duplicates(fused: pd.DataFrame) -> pd.DataFrame:
    """Ablation helper for --feature-set worst_only: drops the top_<f>/
    bottom_<f> raw columns, keeping worst_<f> (= max of the two views,
    already the project's worst-side-wins convention) plus species/level.
    Tests whether fold-to-fold instability is a too-many-features-for-
    the-leaf-count problem -- keeping only worst_<f> removes 2/3 of the
    raw feature block before z_/elevated columns are even added, without
    discarding any distinct signal (top_/bottom_ are redundant with
    worst_ for any leaf where one side is undamaged)."""
    drop_cols = [c for c in fused.columns if c.startswith("top_") or c.startswith("bottom_")]
    return fused.drop(columns=drop_cols)


def compute_species_baselines(fused: pd.DataFrame, feature_cols: List[str]) -> SpeciesBaselines:
    """Per-species healthy median/IQR for each candidate feature, from
    whatever leaves are in `fused` (caller controls whether that's a
    full train set, a single CV fold's training partition, etc. --
    this function itself has no opinion about leakage, the caller does).
    Same robust-z convention as diagnose_feature_gaps.py's
    compute_elevated_feature_counts(): IQR floored at EPS to avoid
    divide-by-near-zero blowups when a species' healthy leaves show
    near-zero variance on a feature (see colour_pct_chlorotic /
    wal_kollu in gap_report.csv for what happens without the floor)."""
    baselines: SpeciesBaselines = {}
    for species, grp in fused.groupby("species"):
        healthy = grp[grp["level"] == "healthy"]
        baselines[species] = {}
        for col in feature_cols:
            fused_col = f"worst_{col}"
            if fused_col not in grp.columns:
                continue
            if species in FEATURE_SPECIES_EXCLUSIONS.get(col, ()):
                # Deliberately suppressed -- see FEATURE_SPECIES_EXCLUSIONS
                # docstring. Same (nan, nan) pathway as "too few healthy
                # leaves to baseline", so z_<col> comes out NaN and gets
                # filled to -1.0 by _finalize_X's existing convention --
                # no new missing-value handling needed anywhere else.
                baselines[species][col] = (np.nan, np.nan)
                continue
            vals = healthy[fused_col].astype(float).dropna()
            if len(vals) < 3:
                med, iqr = np.nan, np.nan
            else:
                med = float(np.median(vals))
                raw_iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                # FIX (this session): a species whose healthy leaves show
                # ZERO real variance on this feature (raw_iqr == 0, e.g.
                # kathurupila's miner-trail columns are 0 for every
                # healthy leaf) used to have that 0 floored straight to
                # EPS=1e-6, so any nonzero "high" value produced a
                # millions-magnitude z-score (confirmed in gap_report.csv:
                # kathurupila's ldsi_hole_sub effect size = 2,590,166.12).
                # That's a divide-by-near-zero artifact, not a real
                # measure of how elevated the leaf is. Treat zero-variance
                # the same as "not enough healthy leaves to baseline"
                # (NaN -> filled to -1.0 downstream) rather than fabricate
                # an unbounded number -- this still lets the RAW worst_<f>
                # column (not just its z-score) carry the signal for
                # species/features like this.
                if raw_iqr <= EPS:
                    med, iqr = np.nan, np.nan
                else:
                    iqr = raw_iqr
            baselines[species][col] = (med, iqr)
    return baselines


def add_species_relative_features(
    fused: pd.DataFrame, feature_cols: List[str], baselines: SpeciesBaselines
) -> pd.DataFrame:
    """Adds z_<feature> (per-species robust z against the supplied
    baselines) and n_features_elevated_z<T> for each T in
    ELEVATED_Z_THRESHOLDS. A leaf whose species has <3 healthy leaves in
    whatever data the baselines were fit from (or a species baselines
    has never seen) gets NaN z for that feature -- filled to -1.0 later
    by the same fillna(-1.0) convention already used for every other
    missing value in this file, rather than fabricating a 0 that would
    falsely claim "right at the healthy median"."""
    fused = fused.copy()
    n = len(fused)
    elevated_counts = {t: np.zeros(n, dtype=int) for t in ELEVATED_Z_THRESHOLDS}

    species_vals = fused["species"].values
    for col in feature_cols:
        fused_col = f"worst_{col}"
        if fused_col not in fused.columns:
            continue
        raw_vals = fused[fused_col].astype(float).values
        z_vals = np.full(n, np.nan)
        for i in range(n):
            med, iqr = baselines.get(species_vals[i], {}).get(col, (np.nan, np.nan))
            v = raw_vals[i]
            if np.isnan(med) or np.isnan(v):
                continue
            z = (v - med) / iqr
            z = float(np.clip(z, -6.0, 6.0))   
            z_vals[i] = z
            for t in ELEVATED_Z_THRESHOLDS:
                if abs(z) >= t:
                    elevated_counts[t][i] += 1
        fused[f"z_{col}"] = z_vals

    for t in ELEVATED_Z_THRESHOLDS:
        fused[f"n_features_elevated_z{str(t).replace('.', '_')}"] = elevated_counts[t]
    return fused


def _finalize_X(fused: pd.DataFrame) -> pd.DataFrame:
    """Drops the raw 'level' label (already extracted as y before this is
    called), one-hot encodes species, fills missing with -1.0."""
    X = fused.drop(columns=["level"])
    X = pd.get_dummies(X, columns=["species"], prefix="species")
    X = X.fillna(-1.0)
    return X


def build_binary_dataset(
    top: pd.DataFrame,
    bottom: pd.DataFrame,
    baselines: Optional[SpeciesBaselines] = None,
    fit_baselines: bool = False,
):
    """Same fusion/collapse convention as legacy_flat_classifier.py's
    build_dataset(), minus the severity (y_severity) target -- Stage 1
    doesn't need it -- plus the species-relative z-score / elevated-count
    features described in the module docstring.

    Exactly one of (baselines=, fit_baselines=True) must be given:
      - fit_baselines=True: this call's OWN data is used to fit the
        species baselines (use this for train / a CV fold's training
        partition). Returns (X, y_binary, groups, baselines).
      - baselines=<dict>: reuses baselines fit elsewhere (use this for
        test, or a CV fold's validation partition, so that split never
        contributes to its own normalisation). Returns (X, y_binary,
        groups).
    """
    if fit_baselines == (baselines is not None):
        raise ValueError(
            "build_binary_dataset: pass exactly one of fit_baselines=True "
            "(train / fold-train) or baselines=<dict> (test / fold-val)."
        )

    fused = _fuse_leaves(top, bottom)

    if fit_baselines:
        baselines = compute_species_baselines(fused, Z_FEATURE_COLS)

    fused = add_species_relative_features(fused, Z_FEATURE_COLS, baselines)

    y_binary = np.where(fused["level"] == "healthy", "healthy", "unhealthy")
    groups = top.loc[fused.index, "leaf_id"].values

    X = _finalize_X(fused)

    if fit_baselines:
        return X, y_binary, groups, baselines
    return X, y_binary, groups


def _fit_stage1_with_tuned_threshold(X: pd.DataFrame, y: np.ndarray, model_name: str, groups: np.ndarray, cv_splits: int = 3):
    """Fits the Stage-1 classifier on X/y, using cross-validation to
    tune the healthy/unhealthy threshold. Returns (fitted_pipeline, threshold)."""
    pipeline = MODEL_BUILDERS[model_name]()

    n_splits = min(cv_splits, int(np.min(np.unique(y, return_counts=True)[1])))
    n_splits = max(n_splits, 2)
    oof_proba = cross_val_predict(
        pipeline, X.values, y,
        cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42),
        groups=groups,
        method="predict_proba",
    )
    classes_preview = np.unique(y)
    unhealthy_col = list(sorted(classes_preview)).index("unhealthy")
    proba_unhealthy = oof_proba[:, unhealthy_col]
    threshold = TwoStageHealthClassifier._best_threshold(y, proba_unhealthy)

    pipeline.fit(X.values, y)
    return pipeline, threshold

def _print_top_feature_importances(pipeline, feature_names: List[str], top_n: int) -> Optional[pd.DataFrame]:
    """Prints the top_n RF/tree-based feature importances against their
    names, and returns the full ranked table. Run this after every
    feature-set change -- if z_*/n_features_elevated_z* aren't showing up
    near the top, they're not earning the dimensionality they cost.
    SVM-RBF (the current default Stage-1 model) has no
    feature_importances_ attribute -- skips with a note rather than
    crashing rather than silently fabricating a ranking that doesn't
    exist for that model type. Re-run with --model random_forest if you
    need this diagnostic."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        print(f"\n[stage1] {type(clf).__name__} has no feature_importances_ "
              f"(only tree-based models do) -- skipping importance printout. "
              f"Re-run with --model random_forest for this diagnostic.")
        return None
    importances = clf.feature_importances_
    ranked = pd.DataFrame({"feature": feature_names, "importance": importances})
    ranked = ranked.sort_values("importance", ascending=False).reset_index(drop=True)
    print(f"\n[stage1] top {top_n} feature importances:")
    print(ranked.head(top_n).to_string(index=False))
    n_engineered_in_top = int(
        ranked.head(top_n)["feature"].str.startswith(("z_", "n_features_elevated")).sum()
    )
    print(f"[stage1] {n_engineered_in_top}/{top_n} of the top {top_n} are engineered "
          f"(z_* / n_features_elevated_z*) features")
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Train the Stage-1 healthy/unhealthy classifier")
    parser.add_argument(
        "--model", choices=list(MODEL_BUILDERS.keys()), default=STAGE1_MODEL,
        help=f"Stage-1 classifier. Default '{STAGE1_MODEL}' per compare_stage1_models.py "
             f"(CV 0.859 vs random_forest's 0.810; test 0.885 vs 0.870).",
    )
    parser.add_argument(
        "--feature-set", choices=["full", "worst_only"], default="full",
        help="'full' keeps top_/bottom_/worst_ raw triplicate (default, current behaviour). "
             "'worst_only' drops the top_/bottom_ raw duplicates before z_/elevated features "
             "are added, as a dimensionality-reduction ablation.",
    )
    parser.add_argument("--top-n-importance", type=int, default=25)
    parser.add_argument("--predictions-out", default="processed/features/stage1_test_predictions.csv")
    args = parser.parse_args()

    top, bottom = load_and_merge_from(TOP_CSV, BOTTOM_CSV)
    train_top, train_bottom, test_top, test_bottom = split_train_test(top, bottom)

    # Fused-but-not-yet-baselined train frame, kept around so each CV
    # fold can refit its OWN species baselines from its OWN training
    # partition only (see module docstring: refit-per-fold is what keeps
    # the CV numbers leakage-free -- a global-train baseline would let a
    # validation leaf's species-mates influence its own normalisation).
    fused_train_raw = _fuse_leaves(train_top, train_bottom)
    if args.feature_set == "worst_only":
        fused_train_raw = _drop_raw_side_duplicates(fused_train_raw)
    y_train_all = np.where(fused_train_raw["level"] == "healthy", "healthy", "unhealthy")
    groups_train = train_top.loc[fused_train_raw.index, "leaf_id"].values

    print(f"[stage1] model = {args.model}")
    print(f"[stage1] feature-set = {args.feature_set}")
    print(f"[stage1] train n={len(fused_train_raw)} rows, "
          f"{len(set(groups_train))} unique leaves, "
          f"class balance: {pd.Series(y_train_all).value_counts().to_dict()}")
    print(f"[stage1] dropped dead features: {sorted(DEAD_FEATURES)}")
    print(f"[stage1] species-relative features added for {len(Z_FEATURE_COLS)} candidate columns "
          f"(z_<feature> + n_features_elevated_z at {ELEVATED_Z_THRESHOLDS})")

    # CV sanity check (grouped by leaf_id, same leakage guard as the rest
    # of the project) before the final fit on all of train.
    cv_f1s = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(
        splitter.split(np.zeros((len(fused_train_raw), 1)), y_train_all, groups_train)
    ):
        assert not (set(groups_train[tr]) & set(groups_train[va])), f"leaf leakage in fold {fold}"

        fused_tr = fused_train_raw.iloc[tr]  # already has top_/bottom_ dropped above if worst_only
        fused_va = fused_train_raw.iloc[va]
        fold_baselines = compute_species_baselines(fused_tr, Z_FEATURE_COLS)

        X_tr = _finalize_X(add_species_relative_features(fused_tr, Z_FEATURE_COLS, fold_baselines))
        X_va = _finalize_X(add_species_relative_features(fused_va, Z_FEATURE_COLS, fold_baselines))
        # Species one-hot columns can differ slightly if a species is
        # absent from one side of the fold -- align so positional
        # .values indexing into the model is always safe.
        X_tr, X_va = X_tr.align(X_va, join="outer", axis=1, fill_value=0)

        y_tr, y_va = y_train_all[tr], y_train_all[va]
        groups_tr = groups_train[tr]

        fold_pipeline, fold_threshold = _fit_stage1_with_tuned_threshold(X_tr, y_tr, args.model, groups_tr)
        proba = fold_pipeline.predict_proba(X_va.values)
        classes = list(fold_pipeline.classes_)
        p_unhealthy = proba[:, classes.index("unhealthy")]
        preds = np.where(p_unhealthy >= fold_threshold, "unhealthy", "healthy")
        f1 = f1_score(y_va, preds, average="macro")
        cv_f1s.append(f1)
        print(f"  fold {fold}: stage1 f1_macro={f1:.4f} (threshold={fold_threshold:.3f})")
    print(f"[stage1 CV] mean f1_macro={np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}")

    # Final fit on ALL training data, tuned threshold. Baselines refit
    # once more here on the full train set (this is the version that
    # also normalises the held-out test set below, and the version
    # persisted for deployment).
    final_baselines = compute_species_baselines(fused_train_raw, Z_FEATURE_COLS)
    X_train = _finalize_X(add_species_relative_features(fused_train_raw, Z_FEATURE_COLS, final_baselines))
    final_pipeline, final_threshold = _fit_stage1_with_tuned_threshold(X_train, y_train_all, args.model, groups_train)    
    print(f"[stage1] final tuned decision threshold = {final_threshold:.3f}")
    print(f"[stage1] final X_train shape = {X_train.shape} "
          f"({X_train.shape[1]} features for {X_train.shape[0]} leaves)")

    fused_test_raw = _fuse_leaves(test_top, test_bottom)
    if args.feature_set == "worst_only":
        fused_test_raw = _drop_raw_side_duplicates(fused_test_raw)
    y_test = np.where(fused_test_raw["level"] == "healthy", "healthy", "unhealthy")
    X_test = _finalize_X(add_species_relative_features(fused_test_raw, Z_FEATURE_COLS, final_baselines))
    # Same column-alignment safety as the CV folds -- also guards against
    # X_train and X_test being built via independent pd.get_dummies()
    # calls with no guarantee their columns lined up before
    # .predict_proba(X_test.values) indexed into them positionally.
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    test_proba = final_pipeline.predict_proba(X_test.values)
    classes = list(final_pipeline.classes_)
    p_unhealthy_test = test_proba[:, classes.index("unhealthy")]
    test_pred = np.where(p_unhealthy_test >= final_threshold, "unhealthy", "healthy")

    print("\n[stage1] HELD-OUT TEST report:")
    print(classification_report(y_test, test_pred))
    print(confusion_matrix(y_test, test_pred, labels=["healthy", "unhealthy"]))

    # Diagnostic 1: does the model actually use the new engineered
    # columns, or did we just add dimensionality for nothing? (RF only --
    # see _print_top_feature_importances docstring.)
    importance_table = _print_top_feature_importances(
        final_pipeline, list(X_train.columns), args.top_n_importance
    )

    # Diagnostic 2: per-leaf test predictions incl. species/4-tier level,
    # so misclassifications can be inspected by species/damage-level
    # rather than just as a raw count. Direct follow-up to
    # gap_report.csv/elevated_feature_counts.csv -- same idea, but now
    # anchored to which leaves the CLASSIFIER (not just raw features)
    # actually gets wrong.
    pred_detail = pd.DataFrame({
        "leaf_id": test_top.loc[fused_test_raw.index, "leaf_id"].values,
        "species": fused_test_raw["species"].values,
        "level": fused_test_raw["level"].values,  # raw 4-tier folder label
        "y_true": y_test,
        "y_pred": test_pred,
        "p_unhealthy": p_unhealthy_test,
    })
    pred_detail["correct"] = pred_detail["y_true"] == pred_detail["y_pred"]
    pred_detail.to_csv(args.predictions_out, index=False)
    print(f"\n[done] wrote per-leaf test predictions -> {args.predictions_out}")

    print("\n[stage1] misclassified test leaves by species x level:")
    wrong = pred_detail[~pred_detail["correct"]]
    if len(wrong):
        print(pd.crosstab(wrong["species"], wrong["level"]).to_string())
    else:
        print("  (none)")

    joblib.dump({
        "stage1_model": final_pipeline,
        "stage1_threshold": final_threshold,
        "feature_columns": list(X_train.columns),
        "species_baselines": final_baselines,
        "z_feature_cols": Z_FEATURE_COLS,
        "dead_features": sorted(DEAD_FEATURES),
        "elevated_z_thresholds": ELEVATED_Z_THRESHOLDS,
        "feature_set": args.feature_set,
        "model_name": args.model,
    }, MODEL_OUT)
    print(f"\n[done] saved Stage-1 {args.model} (binary healthy/unhealthy) -> {MODEL_OUT}")
    return importance_table, pred_detail


if __name__ == "__main__":
    main()
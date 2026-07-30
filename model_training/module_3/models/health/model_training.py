"""
models/health/model_training.py

SHARED DATA-LOADING UTILITIES for the health branch. This file does NOT
train or run any model itself anymore -- it's imported by
train_health_index.py (the current, deployed single-index approach) for:
    TOP_CSV, BOTTOM_CSV, NON_FEATURE_COLS
    load_and_merge_from(), split_train_test(), dataset_label_qc()

The ORIGINAL two-stage flat classifier (binary + moderate/high severity)
that used to live in this file's main()/run_cv()/evaluate_on_test()/
build_dataset() has been moved to models/health/legacy_flat_classifier.py
-- it is a documented negative result (see that file's docstring), kept
separately so it's not confused with, or accidentally run instead of,
the current pipeline.

Expects two feature CSVs produced by preprocessing/health/batch_processor.py
over dataset/health/<species>/<level>/top and .../bottom respectively, each
row carrying at least:
    leaf_id, variant_id, species, level (folder ground truth), qc_pass,
    ldsi_score, is_test, is_augmented, ...

WHY (leaf_id, variant_id) instead of leaf_id alone:
Once augmentation is on, one physical leaf produces 1 original + n_aug
augmented rows PER VIEW. All of them share the same leaf_id by design
(they're the same leaf). A lookup keyed on leaf_id alone can't tell which
top row pairs with which bottom row -- pandas .loc[leaf_id] would return
multiple rows instead of one. variant_id (stamped by batch_processor.py:
0=original, 1..n=augmented) disambiguates this deterministically: top's
variant 3 always pairs with bottom's variant 3 for a given leaf_id. Top
and bottom variant 3 were NOT generated from the same random transform
(each view is augmented independently) -- that's fine, because fusion is
feature-level (worst_<feature> = max), not pixel-level.

Only geometric (flip/rotate) augmentation is used for this branch --
never photometric -- because colour_health.py's necrotic/chlorotic/
pale-patch classification IS the primary health signal, and brightness/
hue/shadow augmentation could fabricate a false severity shift. See
preprocessing/shared/augmentation.py's _build_transform_geo_only()
docstring for the full rationale, and
preprocessing/health/check_augmentation_safety.py for the empirical
drift check that validated this choice.
"""
import pandas as pd
from typing import Dict

from feature_extraction.health.severity_index import calibrate_thresholds, fuse_worst_side

TOP_CSV = "processed/features/health_features_top_augmented.csv"
BOTTOM_CSV = "processed/features/health_features_bottom_augmented.csv"
# NOTE: no MODEL_OUT here. The active model's output path is
# train_health_index.py's own MODEL_OUT
# ("processed/models/vedavision_health_index_model.pkl") -- this file
# only loads/splits data, it doesn't own a model path.

NON_FEATURE_COLS = {
    "image_path", "leaf_id", "variant_id", "species", "level", "view",
    "qc_pass", "qc_reason", "mask_choice",
    "is_test", "is_augmented", "source_path",
    # spots.py's spot_rachis_guard_triggered is a per-leaf QC flag ("did
    # this image's rachis mask look contaminated"), not a health signal --
    # diagnose_feature_gaps.py already excludes it from CANDIDATE_FEATURES
    # for exactly this reason. Added here too so it's excluded at the
    # SAME single source of truth used by every script that fuses raw
    # top/bottom columns (train_stage1_binary.py's _fuse_leaves() in
    # particular) -- without this it silently leaks into the RF as
    # top_/bottom_/worst_spot_rachis_guard_triggered, a QC diagnostic
    # being fed to the model as if it were a symptom.
    "spot_rachis_guard_triggered",
}


def _read_csv_robust(path: str) -> pd.DataFrame:
    """Falls back to cp1252 if a CSV isn't clean UTF-8 (e.g. a free-text
    qc_reason field picked up a Windows smart-punctuation character)."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[warn] {path} is not UTF-8, falling back to cp1252")
        return pd.read_csv(path, encoding="cp1252")


from typing import Dict

# --- 3-tier severity collapse (this session) --------------------------------
# Empirically checked (cliffs_delta + Mann-Whitney) rather than assumed:
#   healthy vs low : delta=-0.125  (real separation)
#   low    vs mid  : delta=-0.046  p=0.403  (NOT significant -- indistinguishable)
#   mid    vs high : delta=-0.147  p=0.006  (real separation)
# This also matches a pattern that showed up independently across nearly
# every train_health_index.py run in this project's history: TRAIN group
# medians consistently had low~33.7, mid~33.2-34.2 (mid sometimes even
# BELOW low), while high consistently separated clearly (42-46). Low and
# mid were never reliably distinguishable at the feature level -- so
# rather than keep forcing a 4-way split the data doesn't support, collapse
# low+mid into one tier. This mirrors the project's standing principle:
# relabel in CODE, not by reorganising folders on disk.
LEVEL_COLLAPSE_MAP: Dict[str, str] = {
    "healthy": "healthy",
    "low": "slight_moderate",
    "mid": "slight_moderate",
    "high": "severe",
}
COLLAPSED_LEVEL_ORDER = ["healthy", "slight_moderate", "severe"]

# FIX (this session): actual dataset folder names use a "damaged_<level>"
# convention for the three non-healthy tiers (e.g. "damaged_mid",
# "damaged_high") -- NOT the bare "low"/"mid"/"high" vocabulary that
# severity_index.py's LEVEL_TO_ORDINAL, health_index.py's
# DEFAULT_LEVEL_ORDER/BINARY_PROXY_SCORE, classifier.py, bootstrap_ci.py,
# and LEVEL_COLLAPSE_MAP above all hardcode. Left unfixed, this crashes
# the very first thing that looks a folder label up in one of those dicts
# (dataset_label_qc -> fuse_worst_side -> LEVEL_TO_ORDINAL[folder_label]
# -> KeyError). Normalizing here, once, at the single load choke point
# every consumer of load_and_merge_from's output already goes through,
# means every downstream file keeps using its existing bare vocabulary
# unchanged -- only the raw folder string gets translated.
#
# If your actual folder names are ever renamed to match the bare
# vocabulary directly, this map still works (the identity entries pass
# unrecognised-but-already-correct values straight through) -- but if a
# genuinely new/unexpected level folder name shows up, this raises
# instead of silently mis-binning it, so a typo surfaces immediately
# rather than corrupting downstream training silently.
LEVEL_NORMALIZE_MAP: Dict[str, str] = {
    "healthy": "healthy",
    "damaged_low": "low",
    "damaged_mid": "mid",
    "damaged_high": "high",
    "low": "low",
    "mid": "mid",
    "high": "high",
}


def normalize_levels(df: pd.DataFrame, level_col: str = "level") -> pd.DataFrame:
    """Translate raw on-disk folder-label strings to the bare
    healthy/low/mid/high vocabulary every other health-branch file
    expects. Raises on any value not in LEVEL_NORMALIZE_MAP rather than
    dropping/mis-mapping it silently -- add a new entry above (or fix the
    folder name on disk) if this ever fires on a legitimate new label."""
    df = df.copy()
    unknown = set(df[level_col].astype(str).unique()) - set(LEVEL_NORMALIZE_MAP)
    if unknown:
        raise ValueError(
            f"Unrecognised level value(s) {unknown} not in LEVEL_NORMALIZE_MAP "
            f"{list(LEVEL_NORMALIZE_MAP)}. Add a mapping for them above before proceeding."
        )
    df[level_col] = df[level_col].astype(str).map(LEVEL_NORMALIZE_MAP)
    return df


def apply_level_collapse(df: pd.DataFrame, level_col: str = "level") -> pd.DataFrame:
    """
    Remaps the raw 4-tier folder label to the empirically-supported 3-tier
    one, IN PLACE on a copy. Call this once, immediately after loading --
    everything downstream (Stage 1's healthy/unhealthy split, dataset_label_qc,
    fit_health_index, validate_monotonicity) then automatically operates on
    the collapsed labels with no further changes needed anywhere else.

    Set apply=False at the call site (or just don't call this) to reproduce
    the original 4-tier behaviour for an ablation comparison in the
    dissertation -- this is a deliberate, documented modelling choice, not
    a silent dataset change, so keep both code paths available.
    """
    df = df.copy()
    unmapped = set(df[level_col].unique()) - set(LEVEL_COLLAPSE_MAP)
    if unmapped:
        raise ValueError(f"Unmapped level value(s) found: {unmapped}")
    df[level_col] = df[level_col].map(LEVEL_COLLAPSE_MAP)
    return df


def load_and_merge_from(top_csv: str, bottom_csv: str, collapse_levels: bool = False):
    """
    Loads both CSVs, keeps only qc_pass rows, and inner-joins top/bottom on
    the compound key (leaf_id, variant_id) -- NOT leaf_id alone -- so
    original AND augmented rows both survive and pair correctly.

    Older CSVs written before variant_id existed are handled by defaulting
    variant_id to 0 for every row (equivalent to "no augmented rows were
    ever written"). If you previously ran --augment with the OLD
    batch_processor.py, those augmented rows have no variant_id to
    disambiguate them and will collide under this fallback -- re-run
    batch_processor.py to regenerate the CSVs before training so every
    augmented row gets a real variant_id.

    collapse_levels (default False -- NOT YET ADOPTED, see notes below):
    if True, applies LEVEL_COLLAPSE_MAP so "level" becomes
    {"healthy","slight_moderate","severe"} instead of the raw 4-tier
    folder label. This was investigated (cliffs_delta showed low-vs-mid
    is NOT significantly separable, p=0.403, while mid-vs-high is,
    p=0.006) but every validated result in this project's history --
    including the currently-locked single continuous health index -- was
    produced with this OFF (4-tier labels). Leave False unless/until the
    3-tier collapse is deliberately re-adopted AND every other file that
    hardcodes "low"/"mid"/"high" (severity_index.py, classifier.py,
    colour_diagnostic.py, bootstrap_ci.py) is updated to match -- turning
    this on alone, without those, will silently break level-based
    grouping elsewhere.
    """
    top = _read_csv_robust(top_csv)
    bottom = _read_csv_robust(bottom_csv)

    for df in (top, bottom):
        if "variant_id" not in df.columns:
            df["variant_id"] = 0

    # FIX (this session): normalize raw folder-label strings
    # (e.g. "damaged_high") to the bare vocabulary
    # ("healthy"/"low"/"mid"/"high") every downstream file expects.
    # Must happen before qc filtering / collapse_levels, since both
    # (and everything after them) assume the bare vocabulary already.
    top = normalize_levels(top)
    bottom = normalize_levels(bottom)

    top = top[top["qc_pass"]].copy()
    bottom = bottom[bottom["qc_pass"]].copy()

    if collapse_levels:
        top = apply_level_collapse(top)
        bottom = apply_level_collapse(bottom)

    top["sample_id"] = top["leaf_id"].astype(str) + "__v" + top["variant_id"].astype(int).astype(str)
    bottom["sample_id"] = bottom["leaf_id"].astype(str) + "__v" + bottom["variant_id"].astype(int).astype(str)

    merged_ids = set(top["sample_id"]) & set(bottom["sample_id"])
    dropped = (set(top["sample_id"]) | set(bottom["sample_id"])) - merged_ids
    if dropped:
        print(f"[warn] {len(dropped)} sample_id(s) missing one view, dropped: {sorted(dropped)[:10]}...")

    top = top[top["sample_id"].isin(merged_ids)].set_index("sample_id")
    bottom = bottom[bottom["sample_id"].isin(merged_ids)].set_index("sample_id")

    n_aug_top = int((top["variant_id"] != 0).sum())
    n_aug_bottom = int((bottom["variant_id"] != 0).sum())
    print(f"[load] top: {len(top)} rows ({n_aug_top} augmented) | "
          f"bottom: {len(bottom)} rows ({n_aug_bottom} augmented)")

    return top, bottom


def load_and_merge():
    return load_and_merge_from(TOP_CSV, BOTTOM_CSV)


def split_train_test(top: pd.DataFrame, bottom: pd.DataFrame):
    """
    Splits on is_test (set by batch_processor.py from the test_ filename
    prefix). A sample counts as test if EITHER view is flagged is_test --
    they should always agree since both views of a leaf are renamed
    together, but this errs toward excluding from train rather than
    silently leaking a test leaf into training. (Test images are never
    augmented, so every test row has variant_id == 0 by construction.)
    """
    test_ids = set(top.index[top["is_test"]]) | set(bottom.index[bottom["is_test"]])
    train_top, train_bottom = top.drop(index=test_ids, errors="ignore"), bottom.drop(index=test_ids, errors="ignore")
    test_top = top.loc[top.index.isin(test_ids)]
    test_bottom = bottom.loc[bottom.index.isin(test_ids)]
    print(f"[split] train samples={len(train_top)}, test samples={len(test_top)}")
    return train_top, train_bottom, test_top, test_bottom


def dataset_label_qc(top: pd.DataFrame, bottom: pd.DataFrame) -> dict:
    """
    Calibrate LDSI thresholds and flag folder-label vs. worst-side-computed
    mismatches BEFORE training. Restricted to ORIGINAL rows only
    (variant_id == 0) -- calibrating thresholds/flagging on augmented
    duplicates of the same physical leaf would inflate the effective
    sample count with correlated copies and bias the median-based
    threshold fit without adding real information (geometric warps don't
    change the underlying colour/hole evidence a leaf's LDSI is built
    from). This is a defensible viva point: thresholds are fit from
    genuinely independent leaves only.
    """
    top_orig = top[top["variant_id"] == 0]
    bottom_orig = bottom.loc[bottom.index.isin(top_orig.index)]

    all_scores = pd.concat([top_orig["ldsi_score"], bottom_orig["ldsi_score"]])
    all_labels = pd.concat([top_orig["level"], bottom_orig["level"]])
    thresholds = calibrate_thresholds(all_scores.tolist(), all_labels.tolist())
    print(f"[calibration] LDSI thresholds fit from {len(top_orig)} original leaves: {thresholds}")

    n_flagged = 0
    for sample_id in top_orig.index:
        folder_label = top_orig.loc[sample_id, "level"]
        result = fuse_worst_side(
            top_orig.loc[sample_id, "ldsi_score"],
            bottom_orig.loc[sample_id, "ldsi_score"],
            thresholds=thresholds,
            folder_label=folder_label,
        )
        if result.label_mismatch:
            n_flagged += 1
            leaf_id = top_orig.loc[sample_id, "leaf_id"]
            print(
                f"[label QC] leaf_id={leaf_id}: folder='{folder_label}' vs "
                f"worst-side-computed='{result.fused_level}' "
                f"(top={result.top.score:.1f}/{result.top.computed_level}, "
                f"bottom={result.bottom.score:.1f}/{result.bottom.computed_level}) "
                f"-- review this leaf's photos/labels"
            )
    print(f"[label QC] {n_flagged}/{len(top_orig)} original leaves flagged for manual review "
          f"({n_flagged / max(len(top_orig), 1) * 100:.1f}%)")
    return thresholds
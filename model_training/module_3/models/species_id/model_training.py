"""
VedaVision — Species Identification Classifier
=================================================
This file did not exist anywhere in the codebase before this session —
model_training.py is new, not a modification of an existing file.

Reads vedavision_features_train_clf.csv (already produced by
batch_processor.py, which as of this session drops the broken whole_leaf
sentinel columns and the redundant colour columns via FEATURE_DROP_COLS —
see batch_processor.py's "Final outputs" section). This script does NOT
re-drop those columns; if you're pointing it at an OLDER csv that still
has them, re-run batch_processor.py first rather than special-casing an
old export here.

Design
------
Stage 1: soft-voting ensemble (RandomForest + SVM-RBF + HistGradientBoosting),
same hyperparameters as the original prototype baseline.

Stage 2: cluster-gated specialist. Stage 1 alone confuses a small number of
species pairs far more than the rest (verified: kattakumanjal/kalawal,
kathurupila/nil_awariya, wal_bilin/maha_undupiyaliya/wal_kollu,
ranawara/siymbala — NOT the same pairs as any older hardcoded look-alike
list; if you have one elsewhere, replace it with whatever this script
prints for your current data, since it's discovered fresh from your CSV
every run). When Stage-1's top-2 candidates are close AND both belong to
the same auto-discovered confusable cluster, a specialist trained only on
that cluster's species re-decides.

Clusters are discovered with an INNER StratifiedGroupKFold on the training
fold only, never touching the outer test fold — so the reported F1-macro
is safe to cite in the dissertation/viva.

Usage
-----
    python -m models.species_id.model_training \
        --train dataset/processed/features/vedavision_features_train_clf.csv \
        --test  dataset/processed/features/vedavision_features_test_clf.csv

IMPORTANT: always run this with `python -m models.species_id.model_training`
from the module_3/ root (not `python model_training.py` from inside the
species_id/ folder, and not by double-clicking/running it directly in an
IDE's "run current file" mode). Running it any other way changes how
Python resolves the models.species_id package path, which is exactly the
kind of mismatch that causes the pickle load error this script's sibling
classifier.py module exists to prevent. If models/__init__.py and
models/species_id/__init__.py don't exist yet, create them (empty files) —
required for `-m` to treat these folders as proper packages.
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report

from models.species_id.classifier import HierarchicalSpeciesClassifier, make_stage1_pipeline

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
# Honest nested CV evaluation on train data
# ---------------------------------------------------------------------------

def evaluate(X, y, groups, n_splits=5, random_state=1):
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_true, y_pred_flat, y_pred_hier = [], [], []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        flat = make_stage1_pipeline()
        flat.fit(X[tr], y[tr])
        y_pred_flat.extend(flat.predict(X[te]))

        hier = HierarchicalSpeciesClassifier()
        hier.fit(X[tr], y[tr], groups[tr])
        y_pred_hier.extend(hier.predict(X[te]))

        y_true.extend(y[te])
        print(f"  fold {fold} clusters: {hier.clusters_}")

    f1_flat = f1_score(y_true, y_pred_flat, average="macro")
    f1_hier = f1_score(y_true, y_pred_hier, average="macro")
    print(f"\nFlat ensemble           F1-macro: {f1_flat:.4f}")
    print(f"Hierarchical (gated)    F1-macro: {f1_hier:.4f}")
    print("\nHierarchical classification report:\n")
    print(classification_report(y_true, y_pred_hier))
    return f1_flat, f1_hier


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
    args = ap.parse_args()

    out_root = Path(args.out)
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_out = models_dir / args.model_name

    X, y, groups, feat_cols = load_xy(args.train, require_groups=True)
    print(f"Loaded {len(y)} rows, {len(feat_cols)} feature columns, "
          f"{len(set(groups))} unique source leaves.")

    print("\n=== Honest nested cross-validation (train data only) ===")
    evaluate(X, y, groups)

    print("\n=== Fitting final model on all training data ===")
    final_model = HierarchicalSpeciesClassifier()
    final_model.fit(X, y, groups)
    print("Final discovered clusters (used for the shipped model):")
    for cl in final_model.clusters_:
        print(" ", cl)

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
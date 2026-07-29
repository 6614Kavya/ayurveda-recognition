"""
Run this locally (same venv you trained in) from module_3/:

    python -m scripts.dump_predictions --model processed/models/vedavision_species_model.pkl \
        --test20 processed/vedavision_features_test_clf.csv \
        --test10 processed/features/vedavision_features_test_clf.csv \
        --out-dir processed/eval_dump

It dumps two small CSVs (predicted vs actual + confidence per sample) that you
can send back instead of the model file, plus a train/test leaf-overlap check.
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

NON_FEATURE_COLS = ["species", "view_side", "image_path", "mask_choice", "coverage_pct", "aug_label"]


def load_bundle(model_path):
    # Model was saved with joblib.dump (confirmed via NumpyArrayWrapper refs
    # in the pickle stream) -- must load with joblib.load, not pickle.load.
    obj = joblib.load(model_path)
    if not isinstance(obj, dict) or "model" not in obj:
        print("ERROR: expected a dict with a 'model' key, got:", type(obj), file=sys.stderr)
        sys.exit(1)
    return obj


def check_leakage(df, csv_path, train_image_paths):
    if "image_path" not in df.columns or not train_image_paths:
        return
    overlap = set(df["image_path"]) & set(train_image_paths)
    if overlap:
        print(f"  !! LEAKAGE WARNING: {len(overlap)}/{len(df)} images in {csv_path} "
              f"also appear in train_image_paths from the saved model bundle.")
        print(f"     e.g. {list(overlap)[:3]}")
    else:
        print(f"  OK: no overlap between {csv_path} and training images.")


def dump_predictions(bundle, csv_path, out_path):
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns")
    train_image_paths = bundle.get("train_image_paths")

    df = pd.read_csv(csv_path)
    if "species" not in df.columns:
        print(f"ERROR: no 'species' column found in {csv_path}", file=sys.stderr)
        sys.exit(1)

    check_leakage(df, csv_path, train_image_paths)

    if feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            print(f"ERROR: {csv_path} is missing expected feature columns: {missing}", file=sys.stderr)
            sys.exit(1)
        X = df[feature_columns]
    else:
        X = df[[c for c in df.columns if c not in NON_FEATURE_COLS]]

    y_true = df["species"].values

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = model.classes_
        pred_idx = np.argmax(proba, axis=1)
        y_pred = classes[pred_idx]
        confidence = proba[np.arange(len(proba)), pred_idx]
    else:
        y_pred = model.predict(X)
        confidence = np.full(len(y_pred), np.nan)

    out = pd.DataFrame({
        "image_path": df["image_path"] if "image_path" in df.columns else np.arange(len(df)),
        "actual": y_true,
        "predicted": y_pred,
        "confidence": confidence,
        "correct": (y_true == y_pred),
    })

    out.to_csv(out_path, index=False)
    n_correct = out["correct"].sum()
    print(f"[{csv_path}] {n_correct}/{len(out)} correct  ({n_correct/len(out):.4f})")
    print(f"  -> wrote {out_path}")

    summary = (
        out.groupby("actual")["correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_correct", "count": "n_total"})
    )
    summary["accuracy"] = summary["n_correct"] / summary["n_total"]
    print(summary.sort_values("accuracy"))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to vedavision_species_model.pkl")
    ap.add_argument("--test20", required=True, help="path to 20-per-species test csv")
    ap.add_argument("--test10", required=True, help="path to 10-per-species test csv")
    ap.add_argument("--out-dir", default="processed/eval_dump")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.model)

    dump_predictions(bundle, args.test20, out_dir / "predictions_test20.csv")
    dump_predictions(bundle, args.test10, out_dir / "predictions_test10.csv")

    print("Done. Send back predictions_test20.csv and predictions_test10.csv (they're tiny).")


if __name__ == "__main__":
    main()
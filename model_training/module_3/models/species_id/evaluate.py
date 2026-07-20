"""
VedaVision — Species Model Evaluation
========================================
Loads a model saved by model_training.py and evaluates it on a CSV
(held-out test, or train — see the leakage warning below), writing every
artefact needed for the dissertation into <out>/evaluation/<run-name>/.

Outputs (all under <out>/evaluation/<run-name>/):
    classification_report.csv       — per-class precision/recall/f1/support
    confusion_matrix_counts.csv      — raw counts, rows=true, cols=predicted
    confusion_matrix_normalized.csv  — row-normalized (recall view)
    confusion_matrix_counts.png      — heatmap
    confusion_matrix_normalized.png  — heatmap
    reference_pair_accuracy.csv      — accuracy restricted to the literature
                                        look-alike pairs (per Botanical
                                        Visual features.md), i.e. "did the
                                        feature engineered for this pair
                                        actually work"
    discovered_cluster_accuracy.csv  — accuracy restricted to whatever
                                        clusters THIS model auto-discovered
                                        during training (only written if the
                                        loaded model is the hierarchical
                                        classifier), i.e. "what's still
                                        actually failing"
    misclassified_examples.csv       — every wrong prediction, for manual
                                        inspection (image_path, true, pred,
                                        top-2 candidate + its probability)

Usage
-----
    python -m models.species_id.evaluate \
        --model processed/models/vedavision_species_model.pkl \
        --data  processed/features/vedavision_features_test_clf.csv \
        --run-name test \
        --out processed
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

# Required even though not referenced by name below: this import is what
# registers HierarchicalSpeciesClassifier under models.species_id.classifier
# in sys.modules BEFORE joblib.load() runs, so pickle can resolve it. Do
# not remove even if your linter flags it as unused.
from models.species_id import classifier as _classifier  # noqa: F401

# Literature/reference look-alike pairs (Botanical Visual features.md) —
# each engineered feature module names one of these as its design target.
# Kept separate from whatever the model auto-discovers, on purpose: this
# table answers "did the feature I built for this pair work", the
# discovered-cluster table answers "what's actually failing now".
REFERENCE_PAIRS = [
    ("kasthuri_dehi", "thunpath_kurundu"),
    ("kalawal", "kattakumanjal"),
    ("kathurupila", "nil_awariya"),
    ("ranawara", "siymbala"),
    ("beli", "wal_kollu"),
]


def load_eval_data(csv_path: str, feature_columns: list):
    df = pd.read_csv(csv_path)
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing {len(missing)} columns the model expects: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}. "
            "This CSV was likely produced by a different batch_processor.py "
            "version than the one used to train the model."
        )
    X = df[feature_columns].values  # reindex to training column ORDER
    y = df["species"].values
    image_paths = df["image_path"].values if "image_path" in df.columns else None
    return df, X, y, image_paths


def check_leakage(image_paths, train_image_paths):
    if image_paths is None or train_image_paths is None:
        return
    eval_set = set(image_paths.tolist())
    overlap = eval_set & train_image_paths
    if overlap:
        pct = 100 * len(overlap) / len(eval_set)
        print(f"\n{'='*70}")
        print(f"WARNING: {len(overlap)}/{len(eval_set)} ({pct:.1f}%) of the images "
              "in this evaluation set were ALSO used to train this model.")
        print("Any accuracy/F1 numbers below are optimistic and not safe to "
              "report — evaluate on the sealed test_ images instead, or "
              "retrain excluding this data.")
        print(f"{'='*70}\n")


def save_confusion_matrices(y_true, y_pred, labels, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / "confusion_matrix_counts.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    cm_norm_df = pd.DataFrame(cm_norm, index=labels, columns=labels)
    cm_norm_df.to_csv(out_dir / "confusion_matrix_normalized.csv")

    for matrix, fmt, title, fname in [
        (cm, "d", "Confusion Matrix (counts)", "confusion_matrix_counts.png"),
        (cm_norm, ".2f", "Confusion Matrix (row-normalized / recall)", "confusion_matrix_normalized.png"),
    ]:
        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
        thresh = matrix.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if val == 0:
                    continue
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                         color="white" if val > thresh else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    return cm_df


def save_classification_report(y_true, y_pred, out_dir: Path):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / "classification_report.csv")
    print(classification_report(y_true, y_pred, zero_division=0))
    return report_df


def save_pair_accuracy(y_true, y_pred, pairs, out_dir: Path, fname: str, label_col: str):
    rows = []
    for cl in pairs:
        cl = list(cl)
        mask = np.isin(y_true, cl)
        if mask.sum() == 0:
            rows.append({label_col: " / ".join(cl), "n_samples": 0,
                         "accuracy": np.nan, "n_confused_within_pair": np.nan})
            continue
        yt, yp = y_true[mask], y_pred[mask]
        acc = accuracy_score(yt, yp)
        # confused specifically WITH another member of the same pair/cluster
        confused = sum(1 for t, p in zip(yt, yp) if p != t and p in cl)
        rows.append({label_col: " / ".join(cl), "n_samples": int(mask.sum()),
                     "accuracy": round(acc, 4), "n_confused_within_pair": confused})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / fname, index=False)
    print(f"\n{fname}:\n{df.to_string(index=False)}")
    return df


def save_misclassified(df, y_true, y_pred, model, X, feature_columns, out_dir: Path):
    wrong = y_true != y_pred
    if not wrong.any():
        pd.DataFrame(columns=["image_path", "true", "predicted"]).to_csv(
            out_dir / "misclassified_examples.csv", index=False)
        print("\nNo misclassified examples.")
        return

    out_rows = {
        "image_path": df["image_path"].values[wrong] if "image_path" in df.columns else np.arange(len(y_true))[wrong],
        "true": y_true[wrong],
        "predicted": y_pred[wrong],
    }

    # If the underlying stage-1 model exposes predict_proba, record top-2
    # candidate + probability for each miss — useful to see how close it was.
    stage1 = getattr(model, "stage1_", model)
    if hasattr(stage1, "predict_proba"):
        proba = stage1.predict_proba(X[wrong])
        classes = stage1.classes_
        order = np.argsort(-proba, axis=1)
        out_rows["stage1_top1"] = classes[order[:, 0]]
        out_rows["stage1_top1_prob"] = proba[np.arange(proba.shape[0]), order[:, 0]].round(3)
        out_rows["stage1_top2"] = classes[order[:, 1]]
        out_rows["stage1_top2_prob"] = proba[np.arange(proba.shape[0]), order[:, 1]].round(3)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_dir / "misclassified_examples.csv", index=False)
    print(f"\n{wrong.sum()} misclassified examples written to misclassified_examples.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .pkl from model_training.py")
    ap.add_argument("--data", required=True, help="CSV to evaluate on (features + species [+ image_path])")
    ap.add_argument("--out", default="processed", help="SAME --out root used by model_training.py")
    ap.add_argument("--run-name", default="eval", help="subfolder name, e.g. 'test' or 'train_cv'")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    train_image_paths = bundle.get("train_image_paths")

    df, X, y_true, image_paths = load_eval_data(args.data, feature_columns)
    check_leakage(image_paths, train_image_paths)

    y_pred = model.predict(X)
    labels = sorted(set(y_true) | set(y_pred))

    out_dir = Path(args.out) / "evaluation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(y_true)} rows on {len(labels)} species.")
    print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro:         {f1_score(y_true, y_pred, average='macro'):.4f}\n")

    save_classification_report(y_true, y_pred, out_dir)
    save_confusion_matrices(y_true, y_pred, labels, out_dir)

    valid_ref_pairs = [p for p in REFERENCE_PAIRS if set(p).issubset(set(labels))]
    save_pair_accuracy(y_true, y_pred, valid_ref_pairs, out_dir,
                        "reference_pair_accuracy.csv", "reference_pair")

    clusters = getattr(model, "clusters_", None)
    if clusters:
        save_pair_accuracy(y_true, y_pred, clusters, out_dir,
                            "discovered_cluster_accuracy.csv", "discovered_cluster")
    else:
        print("\n(Loaded model has no clusters_ attribute — skipping "
              "discovered_cluster_accuracy.csv. Expected if this is a flat, "
              "non-hierarchical model.)")

    save_misclassified(df, y_true, y_pred, model, X, feature_columns, out_dir)

    print(f"\nAll evaluation artefacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
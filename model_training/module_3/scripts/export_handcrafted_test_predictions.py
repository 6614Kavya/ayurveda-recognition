"""
export_handcrafted_test_predictions.py

Generates handcrafted_test_predictions.csv — the file the DL notebook's McNemar's-test
cell (Section 8) reads to compare against the DL model on the same sealed test images.

Run from module_3/ root (file lives in scripts/, matching your other scripts):
    python -m scripts.export_handcrafted_test_predictions
All paths below are relative to module_3/ root, same as your other scripts.
"""
import joblib
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# EDIT THESE to match your actual files/column names
# ------------------------------------------------------------------
MODEL_PATH = Path("processed/models/vedavision_species_model.pkl")   # confirmed working path

# NOTE: vedavision_features_test_clf.csv (the "_clf" version) is fully trimmed to just
# the 130 model features + species — no image_path column. Using the non-_clf sibling,
# which should still carry image_path/leaf_id metadata alongside the same features.
FEATURES_CSV = Path("processed/features/vedavision_features_test.csv")

IMAGE_PATH_COL = "image_path"   # column holding the file path, in FEATURES_CSV
SPECIES_COL    = "species"      # column holding the true species label

OUTPUT_CSV = Path("handcrafted_test_predictions.csv")
# ------------------------------------------------------------------

saved = joblib.load(MODEL_PATH)
print(f"Loaded pickle: {type(saved)}")

# Your saved pickle is a dict: {'model', 'feature_columns', 'train_image_paths'} — pull
# the actual estimator/pipeline and the exact feature column list/order out of it, rather
# than re-deriving feature columns by guessing which CSV columns aren't metadata. Using
# the saved feature_columns is more reliable: it's guaranteed to match what the model was
# actually trained on, including column order.
model = saved["model"]
feature_columns = saved["feature_columns"]
print(f"Model: {type(model)} | {len(feature_columns)} feature columns")

df = pd.read_csv(FEATURES_CSV)
print(f"Loaded {FEATURES_CSV}: {df.shape[0]} rows, {df.shape[1]} columns")

if IMAGE_PATH_COL not in df.columns or SPECIES_COL not in df.columns:
    print("\nIMAGE_PATH_COL/SPECIES_COL not found — here are the actual columns:")
    print(list(df.columns))
    raise KeyError(
        f"Set IMAGE_PATH_COL/SPECIES_COL above to match one of the columns printed above "
        f"(currently IMAGE_PATH_COL={IMAGE_PATH_COL!r}, SPECIES_COL={SPECIES_COL!r})."
    )

# Every row in this file is already sealed test — no is_test filtering needed.
test_df = df.reset_index(drop=True)
print(f"Sealed test rows: {len(test_df)}")

missing = set(feature_columns) - set(test_df.columns)
assert not missing, f"FEATURES_CSV is missing columns the model expects: {missing}"
X_test = test_df[feature_columns]   # exact columns, exact order the model was trained on

y_pred = model.predict(X_test)

def extract_view(path_str):
    """Pull 'top'/'bottom' out of the path — assumes your folder convention
    (<species>/top|bottom/...) includes it as a literal path component."""
    parts_lower = [p.lower() for p in Path(path_str).parts]
    if "top" in parts_lower:
        return "top"
    if "bottom" in parts_lower:
        return "bottom"
    return "unknown"

test_df["_view"] = test_df[IMAGE_PATH_COL].apply(extract_view)
test_df["_filename"] = test_df[IMAGE_PATH_COL].apply(lambda p: Path(p).name)

# Composite key: bare filenames collide across species AND view (test_001.jpg exists once
# per species per view — 12 species x 2 views = up to 24 duplicates of the same basename),
# so species+view+filename is the minimum key that's actually unique.
image_key = (
    test_df[SPECIES_COL].astype(str) + "_" + test_df["_view"] + "_" + test_df["_filename"]
)

out = pd.DataFrame({
    "image_path": image_key,
    "y_true": test_df[SPECIES_COL],
    "y_pred": y_pred,
})

dupes = out["image_path"][out["image_path"].duplicated()]
assert dupes.empty, (
    f"image_path key still not unique after adding species+view — duplicates: "
    f"{dupes.tolist()[:5]}. Your 'top'/'bottom' folder convention may not match what "
    f"extract_view() looks for; check IMAGE_PATH_COL's actual format above."
)

out.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(out)} predictions to {OUTPUT_CSV}")
print(out.head())
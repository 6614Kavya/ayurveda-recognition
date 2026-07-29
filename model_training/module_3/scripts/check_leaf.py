"""
check_leaf.py

Quick manual check: pass in a top-view and bottom-view image of ONE leaf,
tell it the species, and get back the Stage-1 healthy/unhealthy decision
plus the Health Index explainability score.

USAGE (run from the module_3/ root, same place you run train_stage1_binary.py from):

    D:\\Python313\\python.exe check_leaf.py path\\to\\top.jpg path\\to\\bottom.jpg beli

Or just edit TOP_IMAGE / BOTTOM_IMAGE / SPECIES below and run with no args:

    D:\\Python313\\python.exe check_leaf.py

NOTE: species must be spelled exactly as it appears in your training data's
"species" column (e.g. "beli", "kalawal", "kathurupila", ...) -- this script
does not predict species, you have to supply it.
"""
import sys

import cv2

from preprocessing.health.pipeline import run_health_pipeline
from models.health.predict_health import assess_leaf, load_index_model, load_stage1_model

# --- EDIT THESE if running with no command-line args ---
TOP_IMAGE = "dataset/health_labelled_v1/ranawara/healthy/top/image_12.jpg"
BOTTOM_IMAGE = "dataset/health_labelled_v1/ranawara/healthy/bottom/image_12.jpg"
SPECIES = "ranawar"

# --- If your saved Stage-1 model isn't the RF default, point at the right file ---
# (per our chat: predict_health.py defaults to vedavision_stage1_rf_model.pkl,
#  but if train_stage1_binary.py was last run with --model svm_rbf, the file
#  on disk is actually vedavision_stage1_svm_model.pkl -- check `dir processed\models\`
#  and fix this path if needed.)
STAGE1_MODEL_PATH = "processed/models/vedavision_stage1_rf_model.pkl"


def main():
    if len(sys.argv) == 4:
        top_path, bottom_path, species = sys.argv[1], sys.argv[2], sys.argv[3]
    elif len(sys.argv) == 1:
        top_path, bottom_path, species = TOP_IMAGE, BOTTOM_IMAGE, SPECIES
    else:
        print("Usage: python check_leaf.py <top_image> <bottom_image> <species>")
        sys.exit(1)

    top_img = cv2.imread(top_path)
    bottom_img = cv2.imread(bottom_path)
    if top_img is None:
        raise FileNotFoundError(f"Could not read top image: {top_path}")
    if bottom_img is None:
        raise FileNotFoundError(f"Could not read bottom image: {bottom_path}")

    top_row = run_health_pipeline(top_img, image_path=top_path)
    bottom_row = run_health_pipeline(bottom_img, image_path=bottom_path)

    if not top_row.get("qc_pass", False) or not bottom_row.get("qc_pass", False):
        print("[qc] REJECTED -- one or both images failed quality control:")
        print(f"  top:    qc_pass={top_row.get('qc_pass')} reason={top_row.get('qc_reason')}")
        print(f"  bottom: qc_pass={bottom_row.get('qc_pass')} reason={bottom_row.get('qc_reason')}")
        sys.exit(1)

    top_row["species"] = species
    bottom_row["species"] = species

    stage1_bundle = load_stage1_model(STAGE1_MODEL_PATH)
    index_model = load_index_model()

    result = assess_leaf(top_row, bottom_row, index_model=index_model, stage1_bundle=stage1_bundle)

    print("\n=== Leaf assessment ===")
    print(f"species             : {result['species']}")
    print(f"decision             : {result['decision'].upper()}")
    print(f"decision_confidence  : {result['decision_confidence']}")
    print(f"health_value (0-100) : {result['health_value']}  (100 = healthiest)")
    print(f"severity_score_raw   : {result['severity_score_raw']}")
    print("breakdown:")
    for k, v in result["breakdown"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
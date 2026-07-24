import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

_pipeline = None
_label_encoder = None
_feature_columns = None


def _patch_scikit_learn_compat(obj):
    """Fixes cross-version unpickling compatibility for SimpleImputer."""
    if hasattr(obj, "steps"):  # For Pipeline
        for _, step in obj.steps:
            _patch_scikit_learn_compat(step)
    elif hasattr(obj, "transformers_"):  # For ColumnTransformer
        for _, trans, _ in obj.transformers_:
            _patch_scikit_learn_compat(trans)
    elif type(obj).__name__ == "SimpleImputer":
        if not hasattr(obj, "_fill_dtype") and hasattr(obj, "_fit_dtype"):
            obj._fill_dtype = obj._fit_dtype


def get_artifacts():
    global _pipeline, _label_encoder, _feature_columns
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_DIR / "species_classifier_pipeline.pkl")
        
        # Patch the imputer in memory right after unpickling
        _patch_scikit_learn_compat(_pipeline)

        _label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
        _feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

    return _pipeline, _label_encoder, _feature_columns
import json
import joblib
import os
import warnings
from pathlib import Path
import xgboost as xgb
from sklearn.pipeline import Pipeline

MODEL_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
HEALTH_MODEL_DIR = os.path.join(MODEL_DIR, "models", "health_new")
class PredictionError(Exception):
    """Custom exception for model loading or inference errors."""
    pass

# Caches for lazy loading
_pipeline = None
_label_encoder = None
_feature_columns = None
_health_artifacts_cache = None

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass


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
    """Loads and caches species classification artifacts."""
    global _pipeline, _label_encoder, _feature_columns
    if _pipeline is None:
        species_dir = MODEL_DIR / "models"/"identification"
        species_pipe_path = species_dir / "species_classifier_pipeline.pkl"
        encoder_path = species_dir / "label_encoder.pkl"
        cols_path = species_dir / "feature_columns.pkl"

        # Verify species files exist
        for path in [species_pipe_path, encoder_path, cols_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Species artifact missing at: {path.absolute()}"
                )

        _pipeline = joblib.load(species_pipe_path)
        _patch_scikit_learn_compat(_pipeline)

        _label_encoder = joblib.load(encoder_path)
        _feature_columns = joblib.load(cols_path)

    return _pipeline, _label_encoder, _feature_columns

def get_health_artifacts():
    global _health_artifacts_cache
    if _health_artifacts_cache is not None:
        return _health_artifacts_cache

    artifacts = {}

    # Stage 1: Healthy vs Damaged Classifier (two-piece load)
    stage1_preprocessing_path = os.path.join(HEALTH_MODEL_DIR, "stage1_preprocessing.pkl")
    stage1_xgb_model_path = os.path.join(HEALTH_MODEL_DIR, "stage1_xgb_model.json")
    stage1_columns_path = os.path.join(HEALTH_MODEL_DIR, "stage1_healthy_vs_damaged_feature_columns.pkl")
    stage1_encoder_path = os.path.join(HEALTH_MODEL_DIR, "stage1_healthy_vs_damaged_label_encoder.pkl")
    stage1_meta_path = os.path.join(HEALTH_MODEL_DIR, "stage1_healthy_vs_damaged_metadata.json")

    try:
        preprocessing = joblib.load(stage1_preprocessing_path)
        _patch_scikit_learn_compat(preprocessing)

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(stage1_xgb_model_path)

        artifacts["stage1_pipeline"] = Pipeline(preprocessing.steps + [("classifier", xgb_model)])
        artifacts["stage1_columns"] = joblib.load(stage1_columns_path)

        if os.path.exists(stage1_encoder_path):
            encoder = joblib.load(stage1_encoder_path)
            if hasattr(encoder, "classes_"):
                artifacts["int_to_class"] = {i: cls for i, cls in enumerate(encoder.classes_)}
            else:
                artifacts["int_to_class"] = encoder
        elif os.path.exists(stage1_meta_path):
            with open(stage1_meta_path, "r") as f:
                meta = json.load(f)
                classes = meta.get("classes", ["damaged", "healthy"])
                artifacts["int_to_class"] = {i: cls for i, cls in enumerate(classes)}
        else:
            artifacts["int_to_class"] = {0: "damaged", 1: "healthy"}

    except Exception as e:
        raise PredictionError(f"Error loading Health Stage 1 artifacts: {str(e)}")

    # Stage 2: Health Index SVR Regressor  
    stage2_pipeline_path = os.path.join(HEALTH_MODEL_DIR, "stage2_health_index_regressor_pipeline.pkl")
    stage2_columns_path = os.path.join(HEALTH_MODEL_DIR, "stage2_health_index_feature_columns.pkl")

    if os.path.exists(stage2_pipeline_path) and os.path.exists(stage2_columns_path):
        try:
            artifacts["stage2_pipeline"] = joblib.load(stage2_pipeline_path)
            _patch_scikit_learn_compat(artifacts["stage2_pipeline"])
            artifacts["stage2_columns"] = joblib.load(stage2_columns_path)
        except Exception as e:
            print(f"Warning: Could not load Stage 2 artifacts ({e}).")
    # Damage Breakdown: Ridge pipeline (category-level explanation, optional)
        ridge_pipeline_path = os.path.join(HEALTH_MODEL_DIR, "health_damage_breakdown_ridge_pipeline.pkl")
        ridge_columns_path = os.path.join(HEALTH_MODEL_DIR, "health_damage_breakdown_feature_columns.pkl")
     
        if os.path.exists(ridge_pipeline_path) and os.path.exists(ridge_columns_path):
            try:
                artifacts["ridge_pipeline"] = joblib.load(ridge_pipeline_path)
                _patch_scikit_learn_compat(artifacts["ridge_pipeline"])
                artifacts["ridge_columns"] = joblib.load(ridge_columns_path)
            except Exception as e:
                print(f"Warning: Could not load damage breakdown artifacts ({e}).")

    _health_artifacts_cache = artifacts
    return artifacts
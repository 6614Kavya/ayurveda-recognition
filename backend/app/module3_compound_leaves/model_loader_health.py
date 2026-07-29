from pathlib import Path
from functools import lru_cache
import joblib

from app.core.config import settings

# Needed so joblib can resolve the classes baked into each .pkl's pickle
# at unpickle time:
#   - vedavision_health_index_model.pkl  -> HealthIndexModel / SpeciesNormStats
#     (feature_extraction.health.health_index)
#   - vedavision_stage1_svm_model.pkl    -> a plain sklearn Pipeline, no
#     custom class, but importing classifier.py anyway keeps fuse_top_bottom
#     available from the same module path predictor.py expects.
from app.module3_compound_leaves.feature_extraction.health import health_index as _health_index  # noqa: F401
from models.health import classifier as _health_classifier  # noqa: F401
from models.health.train_stage1_binary import Z_FEATURE_COLS as _DEFAULT_Z_FEATURE_COLS


class HealthIndexBundle:
    def __init__(self, model, subscore_columns: list[str], fit_target: str):
        self.model = model
        self.subscore_columns = subscore_columns
        self.fit_target = fit_target


class Stage1Bundle:
    def __init__(self, model, threshold: float, feature_columns: list[str],
                 species_baselines: dict, z_feature_cols: list[str],
                 dead_features: set):
        self.model = model
        self.threshold = threshold
        self.feature_columns = feature_columns
        self.species_baselines = species_baselines
        self.z_feature_cols = z_feature_cols
        self.dead_features = dead_features


@lru_cache(maxsize=1)
def get_health_index_model() -> HealthIndexBundle:
    model_path = Path(settings.module3_health_index_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Health Index model not found at {model_path.resolve()}. Set "
            "MODULE3_HEALTH_INDEX_MODEL_PATH in .env to the .pkl produced "
            "by models/health/train_health_index.py."
        )
    bundle = joblib.load(model_path)
    fit_target = bundle.get("fit_target", "unknown")
    if fit_target != "binary_healthy_vs_unhealthy":
        # Not a hard failure — see predict_health.py's own warning — but
        # surfaced through normal logging so it's visible in server logs
        # rather than only when someone runs the training script by hand.
        import logging
        logging.getLogger(__name__).warning(
            "Loaded Health Index model's fit_target is %r, not "
            "'binary_healthy_vs_unhealthy' — this may be a pre-fix "
            "severity-target model. Re-run train_health_index.py to "
            "regenerate the current binary-target model.", fit_target,
        )
    return HealthIndexBundle(
        model=bundle["health_index_model"],
        subscore_columns=bundle.get("subscore_columns", _health_index.SUBSCORE_RAW_COLUMNS),
        fit_target=fit_target,
    )


@lru_cache(maxsize=1)
def get_stage1_model() -> Stage1Bundle:
    model_path = Path(settings.module3_stage1_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Stage-1 healthy/unhealthy model not found at {model_path.resolve()}. "
            "Set MODULE3_STAGE1_MODEL_PATH in .env to the .pkl produced by "
            "models/health/train_stage1_binary.py."
        )
    bundle = joblib.load(model_path)
    return Stage1Bundle(
        model=bundle["stage1_model"],
        threshold=bundle["stage1_threshold"],
        feature_columns=bundle["feature_columns"],
        species_baselines=bundle["species_baselines"],
        z_feature_cols=bundle.get("z_feature_cols", _DEFAULT_Z_FEATURE_COLS),
        dead_features=set(bundle.get("dead_features", [])),
    )

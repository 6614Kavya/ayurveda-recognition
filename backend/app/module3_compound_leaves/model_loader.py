from pathlib import Path
from functools import lru_cache
import joblib

from app.core.config import settings

# These two imports are NOT unused. joblib/pickle resolves the classifier
# class by the module path baked in at training time
# (models.species_id.classifier.* and, if the pair-specialist safety-guard
# passed on the training run that produced this .pkl,
# models.species_id.pair_specialist.SpeciesClassifierWithPairSpecialist).
# Importing them here guarantees both are registered before joblib.load()
# runs, and turns a silent/cryptic unpickle failure into a clear
# ImportError at startup if either file is ever missing or renamed.
from models.species_id import classifier as _classifier  # noqa: F401
from models.species_id import pair_specialist as _pair_specialist  # noqa: F401


class ModelBundle:
    def __init__(self, model, feature_columns: list[str], train_image_paths: set | None = None):
        self.model = model
        self.feature_columns = feature_columns
        # Not used at inference time; kept so an evaluation/debug script can
        # warn about train/test leakage if it ever loads this same bundle.
        self.train_image_paths = train_image_paths

        # bool flag purely for logging/diagnostics — the pair-specialist
        # wrapper and the plain VotingClassifier both expose .predict()
        # and .predict_proba(), so predictor.py doesn't need to branch on
        # this; it's only here so /health or logs can report which
        # architecture is actually deployed.
        self.is_pair_specialist = isinstance(model, _pair_specialist.SpeciesClassifierWithPairSpecialist)


@lru_cache(maxsize=1)
def get_species_model() -> ModelBundle:
    model_path = Path(settings.module3_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Module 3 species model not found at {model_path.resolve()}. "
            "Set MODULE3_MODEL_PATH in .env to the .pkl produced by "
            "models/species_id/model_training.py "
            "(run as `python -m models.species_id.model_training --train ... --test ...` "
            "from the module_3/ root)."
        )
    bundle = joblib.load(model_path)
    return ModelBundle(
        model=bundle["model"],
        feature_columns=bundle["feature_columns"],
        train_image_paths=bundle.get("train_image_paths"),
    )


from pathlib import Path
from functools import lru_cache
import joblib

from app.core.config import settings


class ModelBundle:
    def __init__(self, model, feature_columns: list[str]):
        self.model = model
        self.feature_columns = feature_columns


@lru_cache(maxsize=1)
def get_species_model() -> ModelBundle:
    model_path = Path(settings.module3_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Module 3 species model not found at {model_path.resolve()}. "
            "Set MODULE3_MODEL_PATH in .env to the .pkl produced by "
            "model_training.py (models/species_id/model_training.py --train ...)."
        )
    bundle = joblib.load(model_path)
    return ModelBundle(model=bundle["model"], feature_columns=bundle["feature_columns"])

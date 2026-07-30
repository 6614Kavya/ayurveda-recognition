import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "leaf_type_router.keras"

_router_model = None


def get_router_model():
    global _router_model

    if _router_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Ensure the file exists and is accessible."
            )
        _router_model = tf.keras.models.load_model(MODEL_PATH)

    return _router_model
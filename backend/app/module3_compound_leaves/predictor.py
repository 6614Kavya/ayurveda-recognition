import numpy as np

from app.module3_compound_leaves.feature_pipeline import decode_image, extract_features
from app.module3_compound_leaves.model_loader import get_species_model


class InvalidImageError(Exception):
    """Raised when the uploaded bytes can't be decoded as an image."""


class LeafNotDetectedError(Exception):
    """Raised when the uploaded image passes decoding but fails QC / leaf detection."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class FeatureMismatchError(Exception):
    """Raised when the extracted features don't match the model's expected feature columns."""
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Missing {len(missing)} expected feature column(s): {missing[:5]}...")


def predict_species(image_bytes: bytes) -> dict:
    """
    Full inference: raw uploaded bytes -> predicted species + confidence.

    Returns
    -------
    dict with keys: species, confidence (0-1 float), mask_choice, coverage_pct
    """
    img_bgr = decode_image(image_bytes)
    if img_bgr is None:
        raise InvalidImageError("Could not decode image — check file format")

    feats, info = extract_features(img_bgr)
    if feats is None:
        raise LeafNotDetectedError(info.get("qc_reason", "leaf not detected"))

    bundle = get_species_model()
    missing = [c for c in bundle.feature_columns if c not in feats]
    if missing:
        raise FeatureMismatchError(missing)

    x_row = np.array([[feats[c] for c in bundle.feature_columns]], dtype=np.float64)

    labels, confidences = bundle.model.predict_with_confidence(x_row)

    return {
        "species": str(labels[0]),
        "confidence": float(confidences[0]),
        "mask_choice": info.get("mask_choice"),
        "coverage_pct": info.get("coverage_pct"),
    }

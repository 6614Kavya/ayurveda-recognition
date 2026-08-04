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

    Note on the model interface (updated for the current flat soft-voting
    ensemble — RF + SVM-RBF + HistGradientBoosting, see classifier.py):
    the model saved by model_training.py is EITHER a plain sklearn
    VotingClassifier, or that same ensemble wrapped in
    SpeciesClassifierWithPairSpecialist (kattakumanjal/kalawal Stage-2,
    only included in the saved model if it did not reduce CV F1-macro on
    the training run — see pair_specialist.py). Both expose .predict()
    and .predict_proba() with the same shapes, so inference code here
    does NOT need to know which one is loaded. There is no
    predict_with_confidence() method on either — that belonged to an
    earlier, now-superseded hierarchical-classifier prototype.
    Confidence is the top-1 class probability from predict_proba(); for
    rows the pair-specialist re-decides, this is the base ensemble's
    probability for its own (pre-override) top class, which
    pair_specialist.py documents as a reasonable approximation rather
    than a mis-calibration.
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

    # Column order MUST match feature_columns exactly — this is the same
    # order the model was trained on (model_training.py's feat_cols).
    x_row = np.array([[feats[c] for c in bundle.feature_columns]], dtype=np.float64)

    model = bundle.model
    pred_label = model.predict(x_row)[0]
    proba = model.predict_proba(x_row)[0]
    confidence = float(np.max(proba))

    return {
        "species": str(pred_label),
        "confidence": confidence,
        "mask_choice": info.get("mask_choice"),
        "coverage_pct": info.get("coverage_pct"),
    }

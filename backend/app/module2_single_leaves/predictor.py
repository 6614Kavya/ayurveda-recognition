import cv2
import numpy as np
from app.module2_single_leaves.preprocessing.Identification.pipeline import preprocess_image
from app.module2_single_leaves.feature_pipeline import extract_all_features
from app.module2_single_leaves.model import get_artifacts

class PredictionError(Exception):
    """Custom exception for prediction failure."""
    pass

def predict_single_leaf(image_input):
    # 1. Decode raw bytes into OpenCV BGR matrix if bytes were received from router
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        image_bgr = image_input

    if image_bgr is None:
        raise PredictionError("Could not decode image file.")

    # 2. Preprocess image
    processed = preprocess_image(image_bgr)
    if processed is None:
        raise PredictionError(
            "Could not detect a clear leaf in this image. "
            "Try a photo with better lighting and a plain background."
        )

    # 3. Feature extraction
    features = extract_all_features(processed)
    
    # Supply view_top (1 = top view) required by model's feature matrix
    features["view_top"] = 1

    # 4. Load pipeline & artifacts
    pipeline, label_encoder, feature_columns = get_artifacts()

    missing = [c for c in feature_columns if c not in features]
    if missing:
        raise PredictionError(f"Feature extraction did not produce: {missing}")

    ordered = [features[col] for col in feature_columns]
    feature_vector = np.array([ordered])

    # 5. Predict class index
    pred_idx = int(pipeline.predict(feature_vector)[0])

    # 6. Extract real confidence score
    if hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(feature_vector)[0]

        # Temperature parameter: Lower T (0.15 - 0.25) sharpens SVM margins to match 94% model accuracy
        T = 0.20

        if np.ndim(raw_scores) == 0:  # Binary classification
            score = float(raw_scores)
            prob_positive = 1.0 / (1.0 + np.exp(-score / T))
            confidence = prob_positive if pred_idx == 1 else (1.0 - prob_positive)
        else:  # Multi-class classification
            # Scaled Softmax over SVM margins
            scaled_scores = raw_scores / T
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            pseudo_probs = exp_scores / exp_scores.sum()
            confidence = float(pseudo_probs[pred_idx])

        confidence_type = "svm_temperature_scaled"

    else:
        confidence = 1.0
        confidence_type = "unavailable"

    # 7. Decode plant name
    label = label_encoder.inverse_transform([pred_idx])[0]

    return {
        "plant_name": str(label),
        "confidence": round(confidence, 4),
    }
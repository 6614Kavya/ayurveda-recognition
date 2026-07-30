import cv2
import numpy as np
import pandas as pd
from app.module2_single_leaves.preprocessing.Identification.pipeline import preprocess_image
from app.module2_single_leaves.feature_pipeline import extract_all_features
from app.module2_single_leaves.preprocessing.health.pipeline import preprocess_image as preprocess_health_image
from app.module2_single_leaves.feature_extraction.health.feature_pipeline import extract_all_features as extract_health_features
from app.module2_single_leaves.model import get_artifacts, get_health_artifacts 

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

def _decode(image_input):
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image_input

def predict_health(top_input, bottom_input):
    """Predicts leaf health status (Healthy vs Damaged) and Health Index score."""
    # 1. Decode both top and bottom view images
    top_bgr = _decode(top_input)
    bottom_bgr = _decode(bottom_input)
    if top_bgr is None or bottom_bgr is None:
        raise PredictionError("Could not decode one or both image files.")

    # 2. Preprocess both views
    top_processed = preprocess_health_image(top_bgr)
    bottom_processed = preprocess_health_image(bottom_bgr)
    if top_processed is None or bottom_processed is None:
        raise PredictionError(
            "Could not detect a clear leaf in one or both images. "
            "Try photos with better lighting and a plain background."
        )

    # 3. Extract features separately per view
    top_feats = extract_health_features(top_processed)
    bottom_feats = extract_health_features(bottom_processed)
    if top_feats is None or bottom_feats is None:
        raise PredictionError("Feature extraction failed on one or both images.")

    # 4. Combine top and bottom feature dictionaries with suffixes
    row = {}
    row.update({f"{k}_top": v for k, v in top_feats.items()})
    row.update({f"{k}_bottom": v for k, v in bottom_feats.items()})

    # 5. Load model artifacts directly from disk
    artifacts = get_health_artifacts()

    # --- Stage 1: Healthy vs Damaged Classifier ---
    stage1_columns = artifacts["stage1_columns"]
    missing_1 = [c for c in stage1_columns if c not in row]
    if missing_1:
        raise PredictionError(f"Stage 1 feature extraction missing columns: {missing_1}")

    input_df_1 = pd.DataFrame([row])[stage1_columns]
    stage1_pipeline = artifacts["stage1_pipeline"]
    pred_1_idx = int(stage1_pipeline.predict(input_df_1)[0])

    # Decode class string using int_to_class dictionary or LabelEncoder
    int_to_class = artifacts.get("int_to_class", {0: "damaged", 1: "healthy"})
    stage1_status = str(int_to_class.get(pred_1_idx, int_to_class.get(str(pred_1_idx))))

    # Compute classification confidence
    if hasattr(stage1_pipeline, "predict_proba"):
        proba = stage1_pipeline.predict_proba(input_df_1)[0]
        stage1_confidence = float(proba[pred_1_idx])
    elif hasattr(stage1_pipeline, "decision_function"):
        raw_score = float(stage1_pipeline.decision_function(input_df_1)[0])
        T = 0.20
        prob_pos = 1.0 / (1.0 + np.exp(-raw_score / T))
        stage1_confidence = prob_pos if pred_1_idx == 1 else (1.0 - prob_pos)
    else:
        stage1_confidence = 1.0

    # --- Stage 2: SVR Health Index Regressor ---
    if "stage2_pipeline" in artifacts and "stage2_columns" in artifacts:
        stage2_columns = artifacts["stage2_columns"]
        missing_2 = [c for c in stage2_columns if c not in row]
        if missing_2:
            raise PredictionError(f"Stage 2 feature extraction missing columns: {missing_2}")

        input_df_2 = pd.DataFrame([row])[stage2_columns]
        stage2_pipeline = artifacts["stage2_pipeline"]
        
        # Predict damage score (SVR continuous output)
        predicted_damage = float(stage2_pipeline.predict(input_df_2)[0])
        
        # Health index = 1.0 - predicted_damage (clipped between 0.0 and 1.0)
        health_index = float(np.clip(1.0 - predicted_damage, 0.0, 1.0))
    else:
        # Fallback if Stage 2 regressor is unavailable
        health_index = 1.0 if stage1_status == "healthy" else 0.0

    # Calculate percentages AFTER the if-else block so variables are always defined
    confidence_pct = round(float(stage1_confidence) * 100, 2)
    health_pct = round(float(health_index) * 100, 2)

    return {
        "stage1_status": stage1_status,
        "stage1_confidence": f"{confidence_pct}%",
        "health_percentage": f"{health_pct}%"
    }
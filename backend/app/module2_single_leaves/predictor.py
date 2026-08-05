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

def _categorize_feature(feature_name):
    name = feature_name.lower()
    base = name.replace('_top', '').replace('_bottom', '')

    if base.startswith('edge_defect') or base == 'edge_smoothness_deficit':
        return 'Margin Damage'

    if base == 'edge_density':
        return 'Surface Texture (Edge Density)'

    if base.startswith('dark_spot') or base.startswith('pale_spot'):
        return 'Dark Spots / Blemishes'

    if base.startswith('hole_') or base in ('solidity', 'hull_deficit_ratio', 'perim_area_ratio'):
        return 'Holes / Notches'

    if base.startswith('glcm_') or base.startswith('lbp_'):
        return 'Surface Texture (GLCM/LBP)'

    if base in (
        'hue_mean', 'hue_std', 'sat_mean', 'sat_std', 'val_mean', 'val_std',
        'l_mean', 'l_std', 'a_mean', 'b_mean', 'chroma_mean', 'chroma_std',
        'green_ratio', 'yellow_ratio', 'brown_ratio', 'dark_ratio',
        'specular_highlight_ratio',
    ):
        return 'Color'

    return 'Other'


def _compute_damage_breakdown(row, artifacts):
    if "ridge_pipeline" not in artifacts or "ridge_columns" not in artifacts:
        return None

    ridge_columns = artifacts["ridge_columns"]
    missing = [c for c in ridge_columns if c not in row]
    if missing:
        return None

    input_df = pd.DataFrame([row])[ridge_columns]

    ridge_pipeline = artifacts["ridge_pipeline"]
    imputer = ridge_pipeline.named_steps['imputer']
    scaler = ridge_pipeline.named_steps['scaler']
    ridge = ridge_pipeline.named_steps.get('ridge') or ridge_pipeline.named_steps.get('classifier')

    X_imputed = imputer.transform(input_df)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=ridge_columns)

    contributions = X_scaled * ridge.coef_

    feature_categories = pd.Series(
        [_categorize_feature(f) for f in ridge_columns], index=ridge_columns
    )
    
    category_contributions = contributions.T.groupby(feature_categories).sum().squeeze()

    if 'Metadata (ignore)' in category_contributions.index:
        category_contributions = category_contributions.drop('Metadata (ignore)')

    abs_sum = category_contributions.abs().sum()
    if abs_sum == 0:
        return None

    category_pct = (category_contributions / abs_sum * 100).round(1)
    category_pct = category_pct.reindex(category_pct.abs().sort_values(ascending=False).index)

    return [
        {"category": cat, "contribution_percent": float(val)}
        for cat, val in category_pct.items()
    ]

# ── User-friendly grouping for the simplified summary shown to end users 
_USER_FRIENDLY_GROUPS = {
    'Color':                          'Color Change',
    'Dark Spots / Blemishes':         'Dark Spots',
    'Holes / Notches':                'Holes & Tears',
    'Margin Damage':                  'Edge / Margin Damage',
    'Surface Texture (Edge Density)': 'Surface Texture',
    'Surface Texture (GLCM/LBP)':     'Surface Texture',
    'Other':                          'Other Factors',
}

def _simplify_damage_breakdown(technical_breakdown, max_items=5):
    """
    Merges technical categories into user-friendly groups, keeping real
    percentages (still sum to ~100% across ALL groups, just fewer names).
    """
    if not technical_breakdown:
        return []

    grouped = {}
    for item in technical_breakdown:
        display_name = _USER_FRIENDLY_GROUPS.get(item['category'], 'Other Factors')
        grouped[display_name] = grouped.get(display_name, 0.0) + item['contribution_percent']

    sorted_groups = sorted(grouped.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_groups = sorted_groups[:max_items]

    simple_result = []
    for display_name, value in sorted_groups:
        simple_result.append({
            "factor": display_name,
            "percentage": round(abs(value), 1)
        })

    return simple_result


def predict_single_leaf(image_input):
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        image_bgr = image_input

    if image_bgr is None:
        raise PredictionError("Could not decode image file.")

    processed = preprocess_image(image_bgr)
    if processed is None:
        raise PredictionError(
            "Could not detect a clear leaf in this image. "
            "Try a photo with better lighting and a plain background."
        )

    features = extract_all_features(processed)
    features["view_top"] = 1

    pipeline, label_encoder, feature_columns = get_artifacts()

    missing = [c for c in feature_columns if c not in features]
    if missing:
        raise PredictionError(f"Feature extraction did not produce: {missing}")

    ordered = [features[col] for col in feature_columns]
    feature_vector = np.array([ordered])

    pred_idx = int(pipeline.predict(feature_vector)[0])

    if hasattr(pipeline, "decision_function"):
        raw_scores = pipeline.decision_function(feature_vector)[0]
        T = 0.20

        if np.ndim(raw_scores) == 0:
            score = float(raw_scores)
            prob_positive = 1.0 / (1.0 + np.exp(-score / T))
            confidence = prob_positive if pred_idx == 1 else (1.0 - prob_positive)
        else:
            scaled_scores = raw_scores / T
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            pseudo_probs = exp_scores / exp_scores.sum()
            confidence = float(pseudo_probs[pred_idx])

        confidence_type = "svm_temperature_scaled"
    else:
        confidence = 1.0
        confidence_type = "unavailable"

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
    """Predicts leaf health status, Health Index score, and a simple
    plain-language damage summary (e.g. Edge/Margin Damage, Holes & Tears)."""
    top_bgr = _decode(top_input)
    bottom_bgr = _decode(bottom_input)
    if top_bgr is None or bottom_bgr is None:
        raise PredictionError("Could not decode one or both image files.")

    top_processed = preprocess_health_image(top_bgr)
    bottom_processed = preprocess_health_image(bottom_bgr)
    if top_processed is None or bottom_processed is None:
        raise PredictionError(
            "Could not detect a clear leaf in one or both images. "
            "Try photos with better lighting and a plain background."
        )

    top_feats = extract_health_features(top_processed)
    bottom_feats = extract_health_features(bottom_processed)
    if top_feats is None or bottom_feats is None:
        raise PredictionError("Feature extraction failed on one or both images.")

    row = {}
    row.update({f"{k}_top": v for k, v in top_feats.items()})
    row.update({f"{k}_bottom": v for k, v in bottom_feats.items()})

    artifacts = get_health_artifacts()

    # --- Stage 1: Healthy vs Damaged Classifier ---
    stage1_columns = artifacts["stage1_columns"]
    missing_1 = [c for c in stage1_columns if c not in row]
    if missing_1:
        raise PredictionError(f"Stage 1 feature extraction missing columns: {missing_1}")

    input_df_1 = pd.DataFrame([row])[stage1_columns]
    stage1_pipeline = artifacts["stage1_pipeline"]
    pred_1_idx = int(stage1_pipeline.predict(input_df_1)[0])

    int_to_class = artifacts.get("int_to_class", {0: "damaged", 1: "healthy"})
    stage1_status = str(int_to_class.get(pred_1_idx, int_to_class.get(str(pred_1_idx))))

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

        predicted_damage = float(stage2_pipeline.predict(input_df_2)[0])
        health_index = float(np.clip(1.0 - predicted_damage, 0.0, 1.0))
    else:
        health_index = 1.0 if stage1_status == "healthy" else 0.0

    # --- Damage Summary (simple, user-facing -- optional) ---
    technical_breakdown = _compute_damage_breakdown(row, artifacts)
    damage_summary = _simplify_damage_breakdown(technical_breakdown) if technical_breakdown else []

    breakdown_dict = {item['factor']: f"{item['percentage']}%" for item in damage_summary}

    confidence_pct = round(float(stage1_confidence) * 100, 2)
    health_pct = round(float(health_index) * 100, 2)

    result = {
         "decision": stage1_status,
         "decision_confidence": round(float(confidence_pct), 4),
         "health_value": f"{health_pct}%",
         "breakdown": breakdown_dict,
     }

    return result
# module1_flowers/router.py

from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
import joblib
import os
from app.module1_flowers.preprocess import load_and_resize, extract_roi, extract_all_features, load_bytes_as_rgb
from app.shared.schemas import PredictionResponse
from app.core.database import get_db

router = APIRouter(prefix="/predict", tags=["Module 1 — Flowers"])

# Load models once at startup 
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))   # → backend/app
MODEL_DIR = os.path.join(BASE_DIR, 'module1_flowers', 'models')

try:
    model   = joblib.load(os.path.join(MODEL_DIR, 'flower_model.joblib'))
    scaler  = joblib.load(os.path.join(MODEL_DIR, 'flower_scaler.joblib'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'flower_label_encoder.joblib'))
    print(f'Flower model loaded. Classes: {list(encoder.classes_)}')
except FileNotFoundError as e:
    print(f'Model file missing: {e}')
    model = scaler = encoder = None


@router.post("/flower", response_model=PredictionResponse)
async def predict_flower(file: UploadFile = File(...)):
    db = get_db();
    contents = await file.read()

    rgb = load_bytes_as_rgb(contents)
    if rgb is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image. Make sure it's a valid JPEG or PNG file."
        )

    roi = extract_roi(rgb)
    feat_v = extract_all_features(roi)
    feat_scaled = scaler.transform([feat_v])

    pred_encoded = model.predict(feat_scaled)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]
    confidence    = float(model.predict_proba(feat_scaled).max())

    # Fetch additional information from the database
    plant_info = await db.flowers.find_one({"label": pred_label})

    return PredictionResponse(
        plant_name        = pred_label,
        confidence        = round(confidence * 100, 2),
        module            = plant_info.get("module", "") if plant_info else "module1_flowers",
        sinhala_name      = plant_info.get("sinhala_name", "") if plant_info else "",
        uses              = plant_info.get("uses", "") if plant_info else "",
        diseases_treated  = plant_info.get("diseases_treated", []) if plant_info else []
    )
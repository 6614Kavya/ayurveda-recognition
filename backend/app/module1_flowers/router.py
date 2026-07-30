# module1_flowers/router.py

import asyncio
import json
import os

import cv2
import joblib
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, UploadFile, HTTPException
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from app.module1_flowers.preprocess import load_and_resize, extract_roi, extract_all_features, load_bytes_as_rgb
from app.shared.schemas import PredictionResponse
from app.core.database import get_db

router = APIRouter(prefix="/predict", tags=["Module 1 — Flowers"])

# ── Load models once at startup ──────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(__file__))   # → backend/app
MODEL_DIR = os.path.join(BASE_DIR, 'module1_flowers', 'models')
IMG_SIZE  = (224, 224)

try:
    model   = joblib.load(os.path.join(MODEL_DIR, 'flower_model.joblib'))
    scaler  = joblib.load(os.path.join(MODEL_DIR, 'flower_scaler.joblib'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'flower_label_encoder.joblib'))
    print(f'Flower SVM/RF model loaded. Classes: {list(encoder.classes_)}')
except FileNotFoundError as e:
    print(f'Model file missing: {e}')
    model = scaler = encoder = None

try:
    cnn_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model_adapted.keras'))
    with open(os.path.join(MODEL_DIR, 'class_names.json')) as f:
        CNN_CLASS_NAMES = json.load(f)
    print(f'Flower CNN model loaded. Classes: {CNN_CLASS_NAMES}')
except (FileNotFoundError, OSError) as e:
    print(f'CNN model file missing: {e}')
    cnn_model = None
    CNN_CLASS_NAMES = []


@router.post("/flower", response_model=PredictionResponse)
async def predict_flower(file: UploadFile = File(...)):
    db = get_db()
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

    plant_info = await db.flowers.find_one({"label": pred_label})

    return PredictionResponse(
        plant_name        = pred_label,
        confidence        = round(confidence * 100, 2),
        module            = plant_info.get("module", "") if plant_info else "module1_flowers",
        sinhala_name      = plant_info.get("sinhala_name", "") if plant_info else "",
        uses              = plant_info.get("uses", "") if plant_info else "",
        diseases_treated  = plant_info.get("diseases_treated", []) if plant_info else []
    )


def _run_cnn_inference(roi: dict) -> np.ndarray:
    """
    Pulled out as its own plain function (not async) so it can be
    handed to a threadpool via asyncio.to_thread below. model.predict()
    is a blocking, CPU/GPU-bound call — running it directly inside an
    `async def` endpoint would freeze the event loop for every other
    concurrent request until this one finishes.
    """
    cnn_input = cv2.resize(roi['roi_rgb'], IMG_SIZE).astype(np.float32)
    cnn_input = preprocess_input(cnn_input)
    cnn_input = np.expand_dims(cnn_input, axis=0)
    return cnn_model.predict(cnn_input, verbose=0)[0]


@router.post("/flower/cnn", response_model=PredictionResponse)
async def predict_flower_cnn(file: UploadFile = File(...)):
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded on this server.")

    db = get_db()
    contents = await file.read()

    rgb = load_bytes_as_rgb(contents)
    if rgb is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image. Make sure it's a valid JPEG or PNG file."
        )

    roi = extract_roi(rgb)
    if 'warn' in roi['status']:
        raise HTTPException(
            status_code=422,
            detail=f"Could not isolate a flower in this image: {roi['status']}"
        )

    probs = await asyncio.to_thread(_run_cnn_inference, roi)
    idx = int(np.argmax(probs))
    pred_label = CNN_CLASS_NAMES[idx]
    confidence = float(probs[idx])

    plant_info = await db.flowers.find_one({"label": pred_label})

    return PredictionResponse(
        plant_name        = pred_label,
        confidence        = round(confidence * 100, 2),
        module            = plant_info.get("module", "") if plant_info else "module1_flowers",
        sinhala_name      = plant_info.get("sinhala_name", "") if plant_info else "",
        uses              = plant_info.get("uses", "") if plant_info else "",
        diseases_treated  = plant_info.get("diseases_treated", []) if plant_info else []
    )
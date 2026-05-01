from fastapi import APIRouter, UploadFile, File, HTTPException
from app.shared.schemas import PredictionResponse
from app.shared.preprocess import load_and_resize

router = APIRouter(prefix="/predict", tags=["Module 2 — Single leaves"])

@router.post("/single-leaf", response_model=PredictionResponse)
async def predict_single_leaf(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images accepted")
    
    image_bytes = await file.read()
    img = load_and_resize(image_bytes)

    return PredictionResponse(
        plant_name="Thora",
        confidence=0.87,
        module="module2_single_leaves",
        sinhala_name="තොර",
        uses="Leaves used for skin conditions",
        diseases_treated=["eczema", "skin rashes"]
    )
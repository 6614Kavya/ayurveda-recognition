from fastapi import APIRouter, UploadFile, File, HTTPException
from app.shared.schemas import PredictionResponse
from app.shared.preprocess import load_and_resize

router = APIRouter(prefix="/predict", tags=["Module 3 — Compound leaves"])

@router.post("/compound-leaf", response_model=PredictionResponse)
async def predict_compound_leaf(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images accepted")
    
    image_bytes = await file.read()
    img = load_and_resize(image_bytes)

    return PredictionResponse(
        plant_name="Araliya",
        confidence=0.95,
        module="module3_compound_leaves",
        sinhala_name="අරලිය",
        uses="Flowers used in religious ceremonies and traditional medicine",
        diseases_treated=["anxiety", "insomnia"]
    )
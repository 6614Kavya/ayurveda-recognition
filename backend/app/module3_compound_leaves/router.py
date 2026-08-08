from fastapi import APIRouter, UploadFile, File, HTTPException

from app.shared.schemas import PredictionResponse
from app.module3_compound_leaves.predictor import (
    predict_species,
    InvalidImageError,
    LeafNotDetectedError,
    FeatureMismatchError,
)
from app.module3_compound_leaves.species_metadata import get_species_display
from app.module3_compound_leaves.model_loader import get_species_model
from app.core.database import get_db

router = APIRouter(prefix="/predict", tags=["Module 3 — Compound leaves"])


@router.post("/compound-leaf", response_model=PredictionResponse)
async def predict_compound_leaf(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images accepted")

    image_bytes = await file.read()

    try:
        result = predict_species(image_bytes)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LeafNotDetectedError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Leaf not detected in image ({e.reason}). "
                   "Please retake the photo against a plain white background "
                   "with the whole compound leaf in frame.",
        )
    except FeatureMismatchError as e:
        raise HTTPException(status_code=500, detail=str(e))

    db = get_db()
    plant_info = await db.compound_leaves.find_one({"label": result["species"]})

    return PredictionResponse(
        plant_name=result["species"],
        confidence=result["confidence"],
        module=plant_info.get("module", "") if plant_info else "module3_compound_leaves",
        sinhala_name=plant_info.get("sinhala_name", "") if plant_info else "",
        uses=plant_info.get("uses", "") if plant_info else "",
        diseases_treated=plant_info.get("diseases_treated", []) if plant_info else []
    )
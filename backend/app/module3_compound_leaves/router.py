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

router = APIRouter(prefix="/predict", tags=["Module 3 — Compound leaves"])


@router.on_event("startup")
def _warm_model():
    
    get_species_model()


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
        # Version skew between the deployed feature_extraction/ code and the
        # one used to train the loaded model — an ops problem, not a bad
        # user photo, so this stays a 500.
        raise HTTPException(status_code=500, detail=str(e))

    display = get_species_display(result["species"])

    return PredictionResponse(
        plant_name=display["plant_name"],
        confidence=result["confidence"],
        module="module3_compound_leaves",
        sinhala_name=display["sinhala_name"],
        uses=display["uses"],
        diseases_treated=display["diseases_treated"],
    )

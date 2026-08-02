import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.shared.schemas import HealthAssessmentResponse
from app.module3_compound_leaves.predictor_health import (
    assess_leaf_health,
    InvalidImageError,
    LeafNotDetectedError,
    HealthFeatureMismatchError,
)
from app.module3_compound_leaves.model_loader_health import get_health_index_model, get_stage1_model

router = APIRouter(prefix="/predict", tags=["Module 3 — Compound leaves (health)"])

logger = logging.getLogger(__name__)


@router.post("/compound-leaf-health", response_model=HealthAssessmentResponse)
async def predict_compound_leaf_health(
    top_file: UploadFile = File(..., description="Top-view leaf image"),
    bottom_file: UploadFile = File(..., description="Bottom-view leaf image"),
):
    for f in (top_file, bottom_file):
        if f.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images accepted")

    try:
        get_health_index_model()
        get_stage1_model()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health-branch models not available yet: {e}",
        )

    top_bytes = await top_file.read()
    bottom_bytes = await bottom_file.read()

    try:
        result = assess_leaf_health(top_bytes, bottom_bytes)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LeafNotDetectedError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Leaf not detected in the {e.which} image ({e.reason}). "
                   "Please retake that photo against a plain white background "
                   "with the whole compound leaf in frame.",
        )
    except HealthFeatureMismatchError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return HealthAssessmentResponse(
        species=result["species"],
        decision=result["decision"],
        decision_confidence=result["decision_confidence"],
        health_value=result["health_value"],
        severity_score_raw=result["severity_score_raw"],
        breakdown=result["breakdown"],
    )
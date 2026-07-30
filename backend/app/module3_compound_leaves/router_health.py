from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.shared.schemas import HealthAssessmentResponse
from app.module3_compound_leaves.predictor_health import (
    assess_leaf_health,
    InvalidImageError,
    LeafNotDetectedError,
    HealthFeatureMismatchError,
)
from app.module3_compound_leaves.model_loader_health import get_health_index_model, get_stage1_model

router = APIRouter(prefix="/predict", tags=["Module 3 — Compound leaves (health)"])


@router.on_event("startup")
def _warm_health_models():
    get_health_index_model()
    get_stage1_model()


@router.post("/compound-leaf-health", response_model=HealthAssessmentResponse)
async def predict_compound_leaf_health(
    top_file: UploadFile = File(..., description="Top-view leaf image"),
    bottom_file: UploadFile = File(..., description="Bottom-view leaf image"),
    species: Optional[str] = Form(
        default=None,
        description="Species code (e.g. 'ranawara'). If omitted, auto-predicted "
                    "from the top-view image via the species-ID model — health "
                    "features are species-relative, so a species label is always "
                    "required internally, whether supplied or inferred.",
    ),
):
    for f in (top_file, bottom_file):
        if f.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images accepted")

    top_bytes = await top_file.read()
    bottom_bytes = await bottom_file.read()

    try:
        result = assess_leaf_health(top_bytes, bottom_bytes, species=species)
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
        # Version skew between deployed feature_extraction/health/ and the
        # code used to train the loaded models — an ops problem, so 500.
        raise HTTPException(status_code=500, detail=str(e))

    return HealthAssessmentResponse(
        species=result["species"],
        decision=result["decision"],
        decision_confidence=result["decision_confidence"],
        health_value=result["health_value"],
        severity_score_raw=result["severity_score_raw"],
        breakdown=result["breakdown"],
    )

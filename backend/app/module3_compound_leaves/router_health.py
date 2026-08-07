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
from app.module3_compound_leaves.model_loader import get_species_model

router = APIRouter(prefix="/predict", tags=["Module 3 — Compound leaves (health)"])

logger = logging.getLogger(__name__)


@router.on_event("startup")
def _warm_health_models():
    # Species-ID is warmed here too so the health endpoint's first real
    # request doesn't pay the load cost — but deliberately left OUTSIDE
    # the try/except below and uncaught: get_species_model() is cached
    # via @lru_cache, so if module3's own router.py already warmed it at
    # startup this is a no-op; if this hook happens to run first, a
    # missing/broken species pickle should still crash startup the same
    # way it does there (species-ID is required for the app to be
    # useful at all), not soften into this endpoint's 503-and-continue
    # behavior below.
    get_species_model()

    # Lazy/non-fatal on purpose: unlike the species-ID model (which is
    # required for the app to be useful at all), the health branch's two
    # .pkl files may not exist yet while it's still being trained. A
    # missing file here should NOT take the whole API down the way a
    # genuinely broken species pickle should — it should just mean the
    # health endpoint 503s until the files are in place, while every
    # other module keeps working. Loaded bundles are still cached via
    # @lru_cache in model_loader_health.py, so this costs nothing once
    # they do exist — first successful call (here, or the first real
    # request if this warm-up is skipped) loads and caches them for
    # every request after.
    try:
        get_health_index_model()
        get_stage1_model()
        logger.info("Health-branch models loaded and warmed.")
    except FileNotFoundError as e:
        logger.warning(
            "Health-branch models not loaded at startup (%s). "
            "/predict/compound-leaf-health will return 503 until "
            "vedavision_health_index_model.pkl and "
            "vedavision_stage1_svm_model.pkl are present.", e,
        )


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
        # Version skew between deployed feature_extraction/health/ and the
        # code used to train the loaded models — an ops problem, so 500.
        raise HTTPException(status_code=500, detail=str(e))

    return HealthAssessmentResponse(
        species=result["species"],
        decision=result["decision"],
        decision_confidence=result["decision_confidence"],
        health_value=result["health_value"],
        severity_score_raw=result["severity_score_raw"],
        symptoms=result["symptoms"],
    )
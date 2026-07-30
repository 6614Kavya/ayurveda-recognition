from fastapi import APIRouter, UploadFile, File, HTTPException
from app.shared.schemas import PredictionResponse
from app.module2_single_leaves.predictor import predict_single_leaf, predict_health, PredictionError
from app.core.database import get_species_metadata  
 

router = APIRouter(prefix="/predict", tags=["Module 2 — Single Leaves"])


@router.post("/single-leaf", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, detail="Only JPEG and PNG images are accepted."
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Run Machine Learning Inference
    try:
        result = predict_single_leaf(image_bytes)
    except PredictionError as pe:
        raise HTTPException(status_code=422, detail=str(pe))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal prediction error: {str(e)}"
        )

    # 2. Fetch species details from MongoDB asynchronously
    db_metadata = await get_species_metadata(result["plant_name"])

    # 3. Build Pydantic response object
    return PredictionResponse(
        plant_name=result["plant_name"],
        confidence=result["confidence"],
        module="module2_single_leaves",
        sinhala_name=db_metadata["sinhala_name"],
        uses=db_metadata["uses"],
        diseases_treated=db_metadata["diseases_treated"],
    )

# Helath

@router.post("/single-leaf/health")
async def predict_leaf_health(
    top_file: UploadFile = File(...),
    bottom_file: UploadFile = File(...)
):
    for f in (top_file, bottom_file):
        if f.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted.")

    top_bytes = await top_file.read()
    bottom_bytes = await bottom_file.read()
    if not top_bytes or not bottom_bytes:
        raise HTTPException(status_code=400, detail="One or both uploaded files are empty.")

    try:
        result = predict_health(top_bytes, bottom_bytes)
    except PredictionError as pe:
        raise HTTPException(status_code=422, detail=str(pe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

    return result
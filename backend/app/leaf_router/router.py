from fastapi import APIRouter, UploadFile, File, HTTPException

from app.shared.schemas import PredictionResponse

from app.leaf_router.predictor import predict_leaf_type

from app.module2_single_leaves.predictor import (
    predict_single_leaf,
    PredictionError,
)

from app.module3_compound_leaves.predictor import (
    predict_species,
    InvalidImageError,
    LeafNotDetectedError,
    FeatureMismatchError,
)

from app.core.database import get_species_metadata
from app.module3_compound_leaves.species_metadata import get_species_display

router = APIRouter(prefix="/predict", tags=["Leaf Router"])


@router.post("/leaf", response_model=PredictionResponse)
async def predict_leaf(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images accepted.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    route = predict_leaf_type(image_bytes)

    try:

        # SINGLE LEAF
        if route["label"] == "simple":

            result = predict_single_leaf(image_bytes)

            metadata = await get_species_metadata(result["plant_name"])

            return PredictionResponse(
                plant_name=result["plant_name"],
                confidence=result["confidence"],
                module="module2_single_leaves",
                sinhala_name=metadata["sinhala_name"],
                uses=metadata["uses"],
                diseases_treated=metadata["diseases_treated"],
            )

        # COMPOUND LEAF
        result = predict_species(image_bytes)

        display = get_species_display(result["species"])

        return PredictionResponse(
            plant_name=display["plant_name"],
            confidence=result["confidence"],
            module="module3_compound_leaves",
            sinhala_name=display["sinhala_name"],
            uses=display["uses"],
            diseases_treated=display["diseases_treated"],
        )

    except PredictionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except LeafNotDetectedError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Leaf not detected ({e.reason})",
        )

    except FeatureMismatchError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
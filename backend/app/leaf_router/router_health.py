from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)


from app.leaf_router.predictor import (
    predict_leaf_type
)


from app.module2_single_leaves.predictor import (
    predict_health,
    PredictionError
)


from app.module3_compound_leaves.predictor_health import (
    assess_leaf_health,
    InvalidImageError,
    LeafNotDetectedError,
    HealthFeatureMismatchError
)


from app.shared.schemas import (
    HealthAssessmentResponse
)


router = APIRouter(
    prefix="/predict",
    tags=["Leaf Health Router"]
)



@router.post("/leaf-health")
async def predict_leaf_health_router(
    top_file: UploadFile = File(...),
    bottom_file: UploadFile = File(...)
):


    # -------------------------
    # Validate images
    # -------------------------

    for f in [top_file, bottom_file]:

        if f.content_type not in [
            "image/jpeg",
            "image/png",
            "image/jpg"
        ]:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG and PNG images accepted"
            )


    top_bytes = await top_file.read()
    bottom_bytes = await bottom_file.read()


    if not top_bytes or not bottom_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty"
        )



    # -------------------------
    # Route leaf type
    # -------------------------

    route = predict_leaf_type(
        top_bytes
    )


    print(
        "Leaf router result:",
        route
    )



    # =================================================
    # SIMPLE LEAF HEALTH
    # =================================================

    if route["label"] == "simple":

        try:

            result = predict_health(
                top_bytes,
                bottom_bytes
            )


        except PredictionError as e:

            raise HTTPException(
                status_code=422,
                detail=str(e)
            )


        return {
            "leaf_type": "simple",
            "confidence": route["confidence"],
            "health_result": result
        }



    # =================================================
    # COMPOUND LEAF HEALTH
    # =================================================

    else:

        try:

            result = assess_leaf_health(
                top_bytes,
                bottom_bytes
            )


        except InvalidImageError as e:

            raise HTTPException(
                status_code=400,
                detail=str(e)
            )


        except LeafNotDetectedError as e:

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Leaf not detected "
                    f"in {e.which} image ({e.reason})"
                )
            )


        except HealthFeatureMismatchError as e:

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )



        return {
            "leaf_type": "compound",
            "confidence": route["confidence"],
            "health_result": HealthAssessmentResponse(
                species=result["species"],
                decision=result["decision"],
                decision_confidence=result["decision_confidence"],
                health_value=result["health_value"],
                severity_score_raw=result["severity_score_raw"],
                symptoms=result["symptoms"]
            )
        }
import io
import numpy as np
from PIL import Image

from app.leaf_router.model import get_router_model


CLASS_NAMES = ["simple", "compound"]   # adjust to your model


def predict_leaf_type(image_bytes: bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image).astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    model = get_router_model()

    prediction = model.predict(image, verbose=0)[0]
    print("Prediction vector:", prediction)
    print("Argmax:", np.argmax(prediction))
    print("Predicted:", CLASS_NAMES[np.argmax(prediction)])

    # index = np.argmax(prediction)
    prediction = model.predict(image, verbose=0)[0][0]

    if prediction >= 0.5:
        label = "compound"
    else:
        label = "simple"

    return {
        "label": label,
        "confidence": float(prediction),
    }
import io
import cv2
import numpy as np
from PIL import Image

from app.leaf_router.model import get_router_model


CLASS_NAMES = ["simple", "compound"]
IMG_SIZE = (224, 224)


def crop_leaf_roi(image_rgb: np.ndarray, pad_frac: float = 0.04) -> np.ndarray:
    """
    Crop to the leaf's bounding box, matching the exact preprocessing used
    during training (VedaVision_LeafType_Router.ipynb, Block 3).

    Steps:
      1. Grayscale + light blur.
      2. INVERTED Otsu threshold — the leaf is darker than its plain
         background, so inverting makes the dark leaf pixels the
         foreground region Otsu separates out. Plain THRESH_BINARY would
         mark the background as foreground and crop to the whole frame.
      3. Morphological open/close to remove small noise specks (dust,
         shadow) so they can't be mistaken for the leaf.
      4. Crop to the largest contour's bounding box, with a small padding
         margin so a thin leaf tip isn't sliced off.
      5. If no contour is found (corrupted/blank image), fall back to the
         original image untouched rather than raising.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image_rgb  # fallback — never crash on one bad image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    H, W = image_rgb.shape[:2]

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + w + pad_x)
    y1 = min(H, y + h + pad_y)

    return image_rgb[y0:y1, x0:x1]


def predict_leaf_type(image_bytes: bytes) -> dict:
    # Decode straight to an OpenCV-compatible array so crop_leaf_roi can
    # run identically to how it ran on cv2.imread() output during training.
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(pil_image)

    cropped = crop_leaf_roi(image_rgb)
    resized = cv2.resize(cropped, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # NOTE: no /255.0 here. The model's own preprocess_input layer expects
    # raw 0-255 pixels and does its own scaling internally — dividing here
    # too was the double-normalization bug that made every prediction
    # collapse toward the same value regardless of input.
    batch = np.expand_dims(resized.astype("float32"), axis=0)

    model = get_router_model()
    prediction = float(model.predict(batch, verbose=0)[0][0])

    label = "compound" if prediction >= 0.5 else "simple"
    confidence = prediction if label == "compound" else 1 - prediction

    return {
        "label": label,
        "confidence": confidence,
    }
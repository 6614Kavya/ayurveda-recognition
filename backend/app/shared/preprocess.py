import cv2
import numpy as np

def load_and_resize(image_bytes: bytes, size: tuple = (256, 256)) -> np.ndarray:
    """Decode raw image bytes into a resized numpy array."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — check file format")
    img = cv2.resize(img, size)
    return img

def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0
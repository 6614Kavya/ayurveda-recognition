import cv2
import numpy as np

def extract_hsv_color(image_bgr):
    hsv  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 250).astype(np.uint8)  # exclude white background
    h, s, v = cv2.split(hsv)
    features = {}
    for name, ch in [('h', h), ('s', s), ('v', v)]:
        pixels = ch[mask > 0]
        if len(pixels) > 0:
            features[f'hsv_{name}_mean'] = float(pixels.mean())
            features[f'hsv_{name}_std']  = float(pixels.std())
        else:
            features[f'hsv_{name}_mean'] = 0.0
            features[f'hsv_{name}_std']  = 0.0
    return features 
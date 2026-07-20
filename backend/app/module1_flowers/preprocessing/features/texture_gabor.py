"""
BLOCK 5D — Gabor Texture Features (24 total)

For each (frequency, orientation) pair in CFG, computes mean |response|
and std of response over the flower mask -> 3 freqs * 4 orientations * 2 = 24.
"""

import cv2
import numpy as np

from ..config import CFG


def extract_gabor_features(roi_gray: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    features = []
    masked_gray = roi_gray.copy().astype(np.float32)
    masked_gray[roi_mask == 0] = 0

    for freq in CFG['gabor_frequencies']:
        for theta in CFG['gabor_orientations']:
            kernel = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta,
                                         lambd=1.0 / freq, gamma=0.5, psi=0)
            filtered = cv2.filter2D(masked_gray, cv2.CV_32F, kernel)
            flower_response = filtered[roi_mask > 0]
            if len(flower_response) > 0:
                features.extend([float(np.abs(flower_response).mean()), float(flower_response.std())])
            else:
                features.extend([0.0, 0.0])

    return np.array(features)

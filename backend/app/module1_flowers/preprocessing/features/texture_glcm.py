"""
BLOCK 5B — GLCM (Gray-Level Co-occurrence Matrix) Texture Features (10 total)

For each of 5 properties (contrast, correlation, energy, homogeneity,
dissimilarity), takes mean + std across all distance/angle combinations
defined in CFG -> 5 * 2 = 10 features.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from ..config import CFG


def extract_glcm_features(roi_gray: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    masked_gray = roi_gray.copy()
    masked_gray[roi_mask == 0] = 0
    quantized = (masked_gray // 4).astype(np.uint8)

    glcm = graycomatrix(quantized, distances=CFG['glcm_distances'], angles=CFG['glcm_angles'],
                         levels=64, symmetric=True, normed=True)

    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']:
        values = graycoprops(glcm, prop)
        features.extend([float(values.mean()), float(values.std())])

    return np.array(features)

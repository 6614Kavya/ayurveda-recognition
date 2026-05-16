"""
VedaVision — Shape Features
============================
Whole-leaf geometry from the binary mask.
All features are dimensionless — a leaf at any zoom gives the same values.

Removed (scale-dependent):    area_px, perimeter_px, eq_diameter_px, bbox_w, bbox_h
Removed (pose-dependent):     ellipse_angle (placement on table ≠ species trait)
"""

import cv2
import numpy as np


def extract_shape_features(leaf_mask: np.ndarray) -> dict:
    """
    Parameters
    ----------
    leaf_mask : uint8 binary mask (255 = foreground)

    Returns
    -------
    dict with keys:
        aspect_ratio, circularity, convexity, solidity, compactness,
        elongation, hu_1 … hu_7
    """
    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return _empty_shape_features()

    cnt      = max(cnts, key=cv2.contourArea)
    area     = float(cv2.contourArea(cnt))
    perim    = float(cv2.arcLength(cnt, closed=True))
    x, y, w, h = cv2.boundingRect(cnt)

    aspect_ratio = float(w) / h       if h > 0          else 0.0
    compactness  = area / (w * h)     if (w * h) > 0    else 0.0
    circularity  = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0

    hull      = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    convexity = area / hull_area if hull_area > 0 else 0.0
    solidity  = convexity   # alias — standard botany term

    if len(cnt) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
        elongation = mi / ma if ma > 0 else 0.0
    else:
        elongation = 0.0

    M      = cv2.moments(cnt)
    hu     = cv2.HuMoments(M).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    feats = {
        "aspect_ratio": aspect_ratio,
        "circularity" : circularity,
        "convexity"   : convexity,
        "solidity"    : solidity,
        "compactness" : compactness,
        "elongation"  : elongation,
    }
    for i, val in enumerate(hu_log):
        feats[f"hu_{i+1}"] = float(val)

    return feats


def _empty_shape_features() -> dict:
    feats = {k: 0.0 for k in [
        "aspect_ratio", "circularity", "convexity", "solidity",
        "compactness", "elongation"
    ]}
    for i in range(1, 8):
        feats[f"hu_{i}"] = 0.0
    return feats

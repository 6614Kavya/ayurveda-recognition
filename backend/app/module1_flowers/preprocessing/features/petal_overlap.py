"""
BLOCK 5K — Petal Overlap & Appendage Features (5 total)

Looks at convexity-defect *depths* (how far the contour dips inward
between petals) rather than positions. Deep, uneven defects suggest
overlapping/irregular petals; shallow, even defects suggest a clean
non-overlapping radial arrangement. Solidity (area / hull area) is a
cheap global summary of the same idea.

Features: defect_depth.mean, defect_depth.max, defect_depth.cv,
          length_outlier_ratio, solidity
"""

import cv2
import numpy as np

from .utils import normalize_defects, coefficient_of_variation as cv


def extract_petal_overlap_features(roi_mask: np.ndarray, min_defect_depth: float = 0.03) -> np.ndarray:
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(5)
    c = max(contours, key=cv2.contourArea)
    if len(c) < 10:
        return np.zeros(5)

    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    M = cv2.moments(c)
    if M['m00'] == 0:
        return np.array([0, 0, 0, 0, solidity])
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    max_r = np.sqrt(((c[:, 0, 0] - cx) ** 2 + (c[:, 0, 1] - cy) ** 2).max())

    hull_idx = np.sort(cv2.convexHull(c, returnPoints=False).flatten())
    if len(hull_idx) < 3:
        return np.array([0, 0, 0, 0, solidity])
    try:
        defects = cv2.convexityDefects(c, hull_idx)
    except cv2.error:
        return np.array([0, 0, 0, 0, solidity])
    if defects is None:
        return np.array([0, 0, 0, 0, solidity])
    defects = normalize_defects(defects)

    depth_thresh = min_defect_depth * max_r * 256
    keep = [i for i in range(defects.shape[0]) if defects[i, 3] > depth_thresh]
    if not keep:
        return np.array([0, 0, 0, 0, solidity])

    depths_norm = np.array([defects[i, 3] / 256.0 for i in keep]) / (max_r + 1e-6)
    valley_idxs = sorted(int(defects[i, 2]) for i in keep)
    n = len(valley_idxs)

    lengths = []
    if n >= 3:
        for i in range(n):
            i1, i2 = valley_idxs[i], valley_idxs[(i + 1) % n]
            arc = c[i1:i2 + 1, 0, :] if i2 > i1 else np.vstack([c[i1:, 0, :], c[:i2 + 1, 0, :]])
            if len(arc) == 0:
                continue
            dists = np.sqrt((arc[:, 0] - cx) ** 2 + (arc[:, 1] - cy) ** 2)
            lengths.append(dists.max())
    lengths = np.array(lengths)

    if len(lengths) >= 2:
        med = np.median(lengths)
        length_outlier_ratio = float(lengths.max() / (med + 1e-6))
    else:
        length_outlier_ratio = 1.0

    return np.array([
        float(depths_norm.mean()), float(depths_norm.max()), cv(depths_norm),
        length_outlier_ratio, float(solidity),
    ])

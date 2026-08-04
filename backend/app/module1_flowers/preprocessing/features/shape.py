"""
BLOCK 5E — Shape Features (12 total)

  Hu moments (log-scaled, rotation/scale invariant) -> 7
  circularity, solidity, aspect_ratio, extent,
  eccentricity                                       -> 5
                                                          --
                                                          12
"""

import cv2
import numpy as np


def extract_shape_features(roi_mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(roi_mask)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.concatenate([hu_log, np.zeros(5)])

    largest_c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_c)
    perimeter = cv2.arcLength(largest_c, closed=True)
    circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6))

    hull = cv2.convexHull(largest_c)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    x, y, bw, bh = cv2.boundingRect(largest_c)
    aspect_ratio = bw / (bh + 1e-6)
    extent = area / (bw * bh + 1e-6)

    if len(largest_c) >= 5:
        (_, _), (ma, MA), _ = cv2.fitEllipse(largest_c)
        eccentricity = float(ma) / (float(MA) + 1e-6)
    else:
        eccentricity = 1.0

    shape_extra = np.array([circularity, solidity, aspect_ratio, extent, eccentricity])
    return np.concatenate([hu_log, shape_extra])

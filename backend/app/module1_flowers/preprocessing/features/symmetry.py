"""
BLOCK 5M — Floral Symmetry Features (3 total)

  Tests 18 mirror axes (every 10°) through the flower centroid and
  measures reflection overlap (IoU) at each angle.

  mean_axis_iou, max_axis_iou, axis_iou_cv                -> 3
                                                               --
                                                               3

  Radially symmetric flowers (karawila, hendirikka) score high IoU
  at MOST angles. Zygomorphic flowers (ranawara) score high only
  near one true axis and drop off sharply elsewhere.
"""

import cv2
import numpy as np


def _mirror_iou_at_angle(mask: np.ndarray, cx: float, cy: float, angle_deg: float) -> float:
    """
    Reflects the mask across the line passing through (cx, cy) at
    angle_deg, then measures how much the reflected shape overlaps the
    original (Intersection over Union). 1.0 = perfect mirror symmetry
    at this angle, 0.0 = no overlap at all.
    """
    theta = np.radians(angle_deg)
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    R = np.array([[c2, s2], [s2, -c2]])
    p = np.array([cx, cy])
    t = p - R @ p
    M = np.hstack([R, t.reshape(2, 1)])
    h, w = mask.shape
    mirrored = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    inter = np.logical_and(mask > 0, mirrored > 0).sum()
    union = np.logical_or(mask > 0, mirrored > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def extract_symmetry_features(roi_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(roi_mask > 0)
    if len(ys) < 10:
        return np.zeros(3)
    cy, cx = float(ys.mean()), float(xs.mean())

    ious = np.array([_mirror_iou_at_angle(roi_mask, cx, cy, a) for a in range(0, 180, 10)])
    mean_iou = float(ious.mean())
    max_iou  = float(ious.max())
    iou_cv   = float(ious.std() / (mean_iou + 1e-6))
    return np.array([mean_iou, max_iou, iou_cv])
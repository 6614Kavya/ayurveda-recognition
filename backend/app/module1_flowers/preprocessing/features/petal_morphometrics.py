"""
BLOCK 5H — Petal Morphometric Features (10 total)

Uses convexity defects (the "valleys" between petal tips on the contour)
to actually segment individual petals, then measures their length, base
width, tip angle, and how symmetric their angular spacing is.

Features: n_petals, length.mean, length.cv, base_width.mean, base_width.cv,
          length/width ratio.mean, ratio.cv, tip_angle.mean, tip_angle.cv,
          symmetry_score
"""

import cv2
import numpy as np

from .utils import normalize_defects, coefficient_of_variation as cv


def extract_petal_morphometrics(roi_mask: np.ndarray, min_defect_depth: float = 0.03) -> np.ndarray:
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(10)

    c = max(contours, key=cv2.contourArea)
    if len(c) < 10:
        return np.zeros(10)

    M = cv2.moments(c)
    if M['m00'] == 0:
        return np.zeros(10)
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']

    hull_idx = np.sort(cv2.convexHull(c, returnPoints=False).flatten())
    if len(hull_idx) < 3:
        return np.zeros(10)

    try:
        defects = cv2.convexityDefects(c, hull_idx)
    except cv2.error:
        return np.zeros(10)
    if defects is None:
        return np.zeros(10)
    defects = normalize_defects(defects)

    max_r = np.sqrt(((c[:, 0, 0] - cx) ** 2 + (c[:, 0, 1] - cy) ** 2).max())
    depth_thresh = min_defect_depth * max_r * 256

    valley_idxs = sorted(int(defects[i, 2]) for i in range(defects.shape[0])
                          if defects[i, 3] > depth_thresh)
    if len(valley_idxs) < 2:
        return np.zeros(10)

    n_petals = len(valley_idxs)
    valley_pts = np.array([c[idx, 0, :] for idx in valley_idxs], dtype=np.float32)
    lengths, base_widths, tip_angles, tip_theta_list = [], [], [], []

    for i in range(n_petals):
        i1, i2 = valley_idxs[i], valley_idxs[(i + 1) % n_petals]
        arc = c[i1:i2 + 1, 0, :] if i2 > i1 else np.vstack([c[i1:, 0, :], c[:i2 + 1, 0, :]])
        if len(arc) == 0:
            continue

        dists = np.sqrt((arc[:, 0] - cx) ** 2 + (arc[:, 1] - cy) ** 2)
        tip = arc[np.argmax(dists)]
        tip_len = dists.max()

        v1, v2 = valley_pts[i], valley_pts[(i + 1) % n_petals]
        base_w = np.linalg.norm(v1 - v2)

        a, b = v1 - tip, v2 - tip
        cos_ang = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6), -1, 1)
        tip_angle = np.degrees(np.arccos(cos_ang))
        tip_theta = np.arctan2(tip[1] - cy, tip[0] - cx)

        lengths.append(tip_len); base_widths.append(base_w)
        tip_angles.append(tip_angle); tip_theta_list.append(tip_theta)

    if not lengths:
        return np.array([float(n_petals), 0, 0, 0, 0, 0, 0, 0, 0, 0])

    lengths, base_widths, tip_angles = map(np.array, (lengths, base_widths, tip_angles))
    ratio = lengths / (base_widths + 1e-6)

    sorted_theta = np.sort(np.array(tip_theta_list))
    gaps = np.diff(np.concatenate([sorted_theta, [sorted_theta[0] + 2 * np.pi]]))
    symmetry_score = 1.0 / (1.0 + gaps.std())

    return np.array([
        float(n_petals), float(lengths.mean()), cv(lengths),
        float(base_widths.mean()), cv(base_widths),
        float(ratio.mean()), cv(ratio),
        float(tip_angles.mean()), cv(tip_angles),
        float(symmetry_score),
    ])

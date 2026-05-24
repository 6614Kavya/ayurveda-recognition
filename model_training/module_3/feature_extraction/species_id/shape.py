"""
VedaVision — Shape Features  (shadow-robust revision  v2 — bug-fix)
====================================================================
Whole-leaf geometry from the binary mask.
All features are dimensionless — a leaf at any zoom gives the same values.

BUG FIXED in v2
---------------
convexity was computed as  area / hull_area  — which is IDENTICAL to solidity.
The correct definition of convexity is the PERIMETER RATIO:

    convexity = hull_perimeter / contour_perimeter

A convex shape has hull_perim ≈ contour_perim  → convexity ≈ 1.0.
A deeply lobed or pinnate outline has many notches  → contour_perim >> hull_perim
→ convexity < 1.0.

solidity  captures AREA fill (how much of the convex hull is occupied).
convexity captures BOUNDARY smoothness (how close the outline is to a convex curve).
They are complementary descriptors and must NOT be the same formula.

Shadow-robustness notes (unchanged)
------------------------------------
Shape features operate on the BINARY MASK CONTOUR, not on pixel colour values,
so they are naturally more robust to shadow contamination than colour or texture
features.  Shadow pixels at the leaf boundary slightly inflate the contour area
and perimeter, but because both numerator and denominator grow together in all
ratios, the ratio values stay nearly constant.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_shape_features(leaf_mask: np.ndarray) -> dict:
    """
    Parameters
    ----------
    leaf_mask : uint8 binary mask (255 = foreground)

    Returns
    -------
    dict with keys:
        aspect_ratio  — bounding-box width / height
        circularity   — 4π·area / perimeter²   (1.0 = perfect circle)
        solidity      — contour area / convex-hull area   (fill ratio)
        convexity     — hull perimeter / contour perimeter (boundary smoothness)
        compactness   — contour area / bounding-box area
        elongation    — minor axis / major axis of fitted ellipse
        hu_1 … hu_7   — log-normalised Hu moments (from binary mask)

    Notes
    -----
    solidity  and convexity are now distinct features:
        solidity  = area / hull_area        (AREA ratio — as before)
        convexity = hull_perim / perim      (PERIMETER ratio — corrected)
    """
    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return _empty_shape_features()

    cnt   = max(cnts, key=cv2.contourArea)
    area  = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, closed=True))
    x, y, w, h = cv2.boundingRect(cnt)

    # ── Basic ratios ──────────────────────────────────────────────────────
    aspect_ratio = float(w) / h    if h > 0       else 0.0
    compactness  = area / (w * h)  if (w * h) > 0 else 0.0
    circularity  = (4.0 * np.pi * area / perim ** 2) if perim > 0 else 0.0

    # ── Convex hull ───────────────────────────────────────────────────────
    hull       = cv2.convexHull(cnt)
    hull_area  = float(cv2.contourArea(hull))
    hull_perim = float(cv2.arcLength(hull, closed=True))

    # solidity  — how well the contour FILLS its convex hull (area ratio)
    solidity   = area / hull_area        if hull_area  > 0 else 0.0

    # convexity — how SMOOTH the boundary is relative to its convex hull
    #             (perimeter ratio).  FIXED: was identical to solidity before.
    #             hull_perim <= perim always, so convexity is in (0, 1].
    convexity  = hull_perim / perim      if perim     > 0 else 0.0

    # ── Ellipse fit ───────────────────────────────────────────────────────
    if len(cnt) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
        elongation = float(mi / ma) if ma > 0 else 0.0
    else:
        elongation = 0.0

    # ── Hu moments from binary MASK (not contour) ─────────────────────────
    # cv2.moments on the full mask image integrates over all foreground
    # pixels — shadow pixels at the edge produce negligible influence.
    M      = cv2.moments(leaf_mask.astype(np.uint8))
    hu     = cv2.HuMoments(M).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    feats = {
        "aspect_ratio": aspect_ratio,
        "circularity" : circularity,
        "solidity"    : solidity,
        "convexity"   : convexity,      # now hull_perim / perim, not area / hull_area
        "compactness" : compactness,
        "elongation"  : elongation,
    }
    for i, val in enumerate(hu_log):
        feats[f"hu_{i+1}"] = float(val)

    return feats


def _empty_shape_features() -> dict:
    feats = {k: 0.0 for k in [
        "aspect_ratio", "circularity", "solidity", "convexity",
        "compactness", "elongation",
    ]}
    for i in range(1, 8):
        feats[f"hu_{i}"] = 0.0
    return feats
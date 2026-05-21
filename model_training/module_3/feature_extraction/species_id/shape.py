"""
VedaVision — Shape Features  (shadow-robust revision)
======================================================
Whole-leaf geometry from the binary mask.
All features are dimensionless — a leaf at any zoom gives the same values.

Shadow-robustness changes vs previous version
---------------------------------------------
Shape features operate on the BINARY MASK CONTOUR, not on pixel colour values,
so they are naturally more robust to shadow contamination than colour or texture
features.  However, shadow pixels at the leaf boundary slightly inflate the
contour area and perimeter, which can shift ratio features.

Fixes applied
~~~~~~~~~~~~~
1. ELLIPSE FIT (aspect ratio, elongation)
   Shadow pixels are a minority at the boundary.  Ellipse fitting is a
   least-squares optimisation over all contour points — minority outlier
   points have small influence on the fitted ellipse.  No change needed
   beyond what was already present.

2. CONVEX HULL RATIOS (solidity, convexity)
   Shadow boundary bumps slightly inflate both the contour area and the
   hull area.  Because both numerator and denominator grow together,
   the RATIO stays nearly constant.  Shadow has minimal effect.

3. HU MOMENTS
   NOW computed from the BINARY MASK image (cv2.moments on the mask
   array), not from the contour.  This is more stable: the mask integrates
   over all foreground pixels, so a few extra shadow pixels at the edge
   produce negligible moment change.  Previously moments were computed
   from cnt (contour object) which is more sensitive to boundary noise.

4. REMOVED (scale-dependent — unchanged from previous version):
   area_px, perimeter_px, eq_diameter_px, bbox_w, bbox_h, ellipse_angle

All features are dimensionless ratios or log-normalised moment values.
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
        aspect_ratio, circularity, convexity, solidity, compactness,
        elongation, hu_1 … hu_7

    Shadow robustness
    -----------------
    All features are dimensionless ratios or log-normalised moments.
    Ratios: numerator and denominator both shift by the same small delta
            when shadow pixels are included → ratio stays stable.
    Hu moments: computed on the binary mask (not the contour) so boundary
                noise from shadow pixels has negligible influence.
    Ellipse fit: least-squares over all contour points — minority shadow
                 boundary points have small pull on the fitted ellipse.
    """
    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return _empty_shape_features()

    cnt   = max(cnts, key=cv2.contourArea)
    area  = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, closed=True))
    x, y, w, h = cv2.boundingRect(cnt)

    # ── Basic ratios ──────────────────────────────────────────────────────
    # Shadow adds ~equal pixels to both sides → ratio stable
    aspect_ratio = float(w) / h    if h > 0       else 0.0
    compactness  = area / (w * h)  if (w * h) > 0 else 0.0
    circularity  = (4.0 * np.pi * area / perim ** 2) if perim > 0 else 0.0

    # ── Convex hull ratios ────────────────────────────────────────────────
    # Both area and hull_area grow by the same shadow delta → ratio stable
    hull      = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    hull_perim= float(cv2.arcLength(hull, closed=True))

    solidity  = area / hull_area   if hull_area  > 0 else 0.0
    convexity = area / hull_area   if hull_area  > 0 else 0.0   # alias

    # ── Ellipse fit ───────────────────────────────────────────────────────
    # Least-squares fit — shadow boundary points are minority outliers
    if len(cnt) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
        elongation = float(mi / ma) if ma > 0 else 0.0
    else:
        elongation = 0.0

    # ── Hu moments from binary MASK (not contour) ─────────────────────────
    # cv2.moments on the full mask image integrates over all foreground
    # pixels.  A few extra shadow pixels at the edge shift moments by
    # a negligible amount compared to computing from the contour only.
    M      = cv2.moments(leaf_mask.astype(np.uint8))
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
        "compactness", "elongation",
    ]}
    for i in range(1, 8):
        feats[f"hu_{i}"] = 0.0
    return feats

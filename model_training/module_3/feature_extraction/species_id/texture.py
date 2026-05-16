"""
VedaVision — Texture Features
==============================
GLCM + LBP texture descriptors extracted from the ENHANCED (sharpened) image.
Enhancement improves vein contrast, making texture boundaries crisper for GLCM.

GLCM: 2 distances × 4 angles → mean+std per property (rotation-invariant)
LBP : P=24, R=3 → 26 uniform pattern bins, normalised histogram
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from preprocessing.config import LBP_RADIUS, LBP_POINTS, GLCM_DIST, GLCM_ANGLES


def extract_texture_features(img_sharp_bgr: np.ndarray,
                              leaf_mask: np.ndarray) -> dict:
    """
    Parameters
    ----------
    img_sharp_bgr : enhanced BGR uint8 image (output of enhance.py)
    leaf_mask     : uint8 binary mask (255 = foreground)

    Returns
    -------
    dict — 8 GLCM stats (4 props × mean+std) + (LBP_POINTS+2) histogram bins
           + lbp_mean, lbp_std = 36 dims total
    """
    gray    = cv2.cvtColor(img_sharp_bgr, cv2.COLOR_BGR2GRAY)
    px_mask = leaf_mask > 0
    if px_mask.sum() < 50:
        return {}

    feats = {}

    # ── GLCM ──────────────────────────────────────────────────────────────────
    # Crop to leaf bounding box to reduce background zeros in co-occurrence matrix
    ys, xs  = np.where(px_mask)
    y1, y2  = ys.min(), ys.max() + 1
    x1, x2  = xs.min(), xs.max() + 1
    # Quantise to 64 levels (256→64); each bin spans 4 intensity units.
    # Sufficient to distinguish dark veins from bright lamina while keeping
    # the GLCM matrix computationally tractable.
    gray_q = (gray[y1:y2, x1:x2] // 4).astype(np.uint8)

    glcm = graycomatrix(
        gray_q, distances=GLCM_DIST, angles=GLCM_ANGLES,
        levels=64, symmetric=True, normed=True
    )
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop)   # shape: (n_dist, n_angles)
        feats[f"glcm_{prop}_mean"] = float(vals.mean())
        feats[f"glcm_{prop}_std"]  = float(vals.std())

    # ── LBP ───────────────────────────────────────────────────────────────────
    lbp      = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins   = LBP_POINTS + 2
    lbp_vals = lbp[px_mask]
    lbp_hist, _ = np.histogram(lbp_vals, bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    for bi, bv in enumerate(lbp_hist):
        feats[f"lbp_{bi:02d}"] = float(bv)
    feats["lbp_mean"] = float(lbp_vals.mean()) if len(lbp_vals) else 0.0
    feats["lbp_std"]  = float(lbp_vals.std())  if len(lbp_vals) else 0.0

    return feats

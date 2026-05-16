"""
VedaVision — Vein Features
===========================
Skeleton-based vein descriptors, all normalised by leaf area or perimeter.

Removed (scale-dependent raw counts):
    vein_pixel_count, vein_branch_points, vein_end_points

Kept (dimensionless):
    vein_density          skeleton_px / leaf_area_px
    vein_length_ratio     skeleton_px / contour_perimeter
    vein_branch_density   branch_points / leaf_area_px
    vein_end_point_density end_points   / leaf_area_px

Extraction pipeline:
    Black Top-Hat (highlights dark veins on bright lamina)
    → Adaptive threshold
    → Skeletonize (Zhang-Suen, 1-px centrelines)
    → Neighbour-count analysis for branch/end points
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def extract_vein_features(img_sharp_bgr: np.ndarray,
                          leaf_mask: np.ndarray
                          ) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    img_sharp_bgr : enhanced BGR uint8 image (output of enhance.py)
    leaf_mask     : uint8 binary mask (255 = foreground)

    Returns
    -------
    feats       : dict with 4 dimensionless vein features
    vein_skel   : uint8 skeleton image (for visualisation)
    vein_binary : uint8 thresholded vein map (for visualisation)
    """
    gray    = cv2.cvtColor(img_sharp_bgr, cv2.COLOR_BGR2GRAY)
    px_mask = leaf_mask > 0
    if px_mask.sum() < 100:
        return {}, np.zeros_like(gray), np.zeros_like(gray)

    feats = {}

    # ── Black Top-Hat: reveals dark veins on bright lamina ────────────────────
    # Kernel = expected vein width at 512px resolution.
    # Secondary veins ≈ 10–15px → kernel=15 captures them without merging.
    k_bthat      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    black_tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_bthat)
    black_tophat = cv2.bitwise_and(black_tophat, black_tophat, mask=leaf_mask)

    # ── Adaptive threshold ────────────────────────────────────────────────────
    # Block size ~image_height/20 (forced odd) — scale-adaptive automatically.
    # C=−2: include only pixels ABOVE local mean (suppress flat lamina).
    blk         = max(11, (gray.shape[0] // 20) | 1)
    vein_binary = cv2.adaptiveThreshold(
        black_tophat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blk, -2
    )
    vein_binary = cv2.bitwise_and(vein_binary, vein_binary, mask=leaf_mask)

    # ── Skeletonize → 1-px vein centrelines ──────────────────────────────────
    vein_skel    = skeletonize(vein_binary > 0).astype(np.uint8) * 255
    leaf_area_px = float(px_mask.sum())
    skel_px      = float((vein_skel > 0).sum())

    feats["vein_density"] = skel_px / leaf_area_px if leaf_area_px > 0 else 0.0

    # ── Length ratio (normalised by perimeter) ────────────────────────────────
    cnts, _   = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(cv2.arcLength(max(cnts, key=cv2.contourArea), True)) if cnts else 1.0
    feats["vein_length_ratio"] = skel_px / perimeter if perimeter > 0 else 0.0

    # ── Branch & end point densities ─────────────────────────────────────────
    # Neighbour count via filter2D (3×3 cross, excluding centre pixel).
    k_n    = np.ones((3, 3), np.uint8);  k_n[1, 1] = 0
    skel_b = (vein_skel > 0).astype(np.uint8)
    nbr    = cv2.filter2D(skel_b.astype(np.float32), -1, k_n.astype(np.float32))
    nbr    = (nbr * skel_b).astype(np.uint8)

    branch_pts = int((nbr >= 3).sum())
    end_pts    = int((nbr == 1).sum())

    feats["vein_branch_density"]    = branch_pts / leaf_area_px if leaf_area_px > 0 else 0.0
    feats["vein_end_point_density"] = end_pts    / leaf_area_px if leaf_area_px > 0 else 0.0

    return feats, vein_skel, vein_binary

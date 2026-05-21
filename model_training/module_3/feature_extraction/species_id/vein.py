"""
VedaVision — Vein Features  (shadow-robust revision)
=====================================================
Skeleton-based vein descriptors, all normalised by leaf area or perimeter.

Shadow-robustness changes vs previous version
---------------------------------------------
PROBLEM 1 — Shadow suppresses vein visibility
  Shadow regions are darker than lamina, so the black top-hat
  (which highlights dark-on-bright structures) produces WEAK response
  in shadow areas even where veins exist.  Veins in shadow are missed.

  FIX: Apply CLAHE to the grayscale image BEFORE top-hat.
  CLAHE is locally adaptive — it brightens shadow regions relative to
  their local neighbourhood, normalising the contrast so veins become
  visible even in dark areas.  After CLAHE, top-hat sees consistent
  contrast across both lit and shadowed leaf regions.

PROBLEM 2 — Density ratios use wrong denominator
  If shadow pixels are included in the mask (slightly over-segmented),
  leaf_area_px is inflated, diluting density ratios.
  
  FIX (was already correct in previous version — preserved):
  All denominators use (mask > 0).sum() — foreground area.
  NOT total image area (512 × 512).

PROBLEM 3 — Top-hat on masked image has a zero-border
  cv2.bitwise_and zeros the background before top-hat, creating a
  bright→zero edge at the mask boundary that can produce false vein
  detections near the leaf edge.

  FIX: Apply top-hat to the FULL grayscale image (no masking yet),
  then clamp the result to the foreground mask afterward.
  Top-hat then sees real image content at the boundary, not an
  artificial zero-edge.

Removed (scale-dependent raw counts — unchanged from previous version):
    vein_pixel_count, vein_branch_points, vein_end_points

Kept (dimensionless):
    vein_density            skeleton_px / leaf_area_px
    vein_length_ratio       skeleton_px / contour_perimeter
    vein_branch_density     branch_points / leaf_area_px
    vein_end_point_density  end_points    / leaf_area_px
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from preprocessing.config import GLCM_DIST, GLCM_ANGLES   # unused here but kept for config parity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    Shadow robustness
    -----------------
    CLAHE normalises local contrast before top-hat, so shadow regions are
    brightened relative to their neighbourhood and veins become visible.
    Top-hat is applied to the full image first, then clamped to the mask,
    avoiding the artificial zero-border edge that caused false detections.
    All density ratios use foreground pixel count as denominator.
    """
    gray    = cv2.cvtColor(img_sharp_bgr, cv2.COLOR_BGR2GRAY)
    px_mask = leaf_mask > 0
    if px_mask.sum() < 100:
        return {}, np.zeros_like(gray), np.zeros_like(gray)

    feats: dict = {}

    # ── Step 1: CLAHE on full grayscale (shadow normalisation) ────────────
    # Apply BEFORE masking so CLAHE can use full local neighbourhood context.
    # Shadow regions get locally brightened → veins become detectable.
    # tileGridSize (8×8) at 512px → 64×64 px tiles — fine enough for leaflet level.
    clahe      = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq    = clahe.apply(gray)

    # ── Step 2: Black top-hat on FULL equalised image (no mask yet) ───────
    # Apply to the unmasked image so the boundary is real image content,
    # not an artificial zero-edge from bitwise_and.
    # Kernel = expected secondary vein width at 512px (≈ 10–15px).
    k_bthat      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    black_tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, k_bthat)

    # ── Step 3: Clamp to foreground mask AFTER top-hat ────────────────────
    black_tophat = cv2.bitwise_and(black_tophat, black_tophat, mask=leaf_mask)

    # ── Step 4: Adaptive threshold ────────────────────────────────────────
    blk = max(11, (gray.shape[0] // 20) | 1)
    vein_binary = cv2.adaptiveThreshold(
        black_tophat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blk, -2,
    )
    vein_binary = cv2.bitwise_and(vein_binary, vein_binary, mask=leaf_mask)

    # ── Step 5: Skeletonise → 1-px vein centrelines ───────────────────────
    vein_skel    = skeletonize(vein_binary > 0).astype(np.uint8) * 255

    # ── Step 6: Density features (foreground area as denominator) ─────────
    leaf_area_px = float(px_mask.sum())          # foreground pixels, NOT 512²
    skel_px      = float((vein_skel > 0).sum())

    feats["vein_density"] = skel_px / leaf_area_px if leaf_area_px > 0 else 0.0

    # ── Step 7: Length ratio (normalised by perimeter) ────────────────────
    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(cv2.arcLength(max(cnts, key=cv2.contourArea), True)) if cnts else 1.0
    feats["vein_length_ratio"] = skel_px / perimeter if perimeter > 0 else 0.0

    # ── Step 8: Branch & end point densities ─────────────────────────────
    k_n    = np.ones((3, 3), np.uint8);  k_n[1, 1] = 0
    skel_b = (vein_skel > 0).astype(np.uint8)
    nbr    = cv2.filter2D(skel_b.astype(np.float32), -1, k_n.astype(np.float32))
    nbr    = (nbr * skel_b).astype(np.uint8)

    branch_pts = int((nbr >= 3).sum())
    end_pts    = int((nbr == 1).sum())

    feats["vein_branch_density"]    = branch_pts / leaf_area_px if leaf_area_px > 0 else 0.0
    feats["vein_end_point_density"] = end_pts    / leaf_area_px if leaf_area_px > 0 else 0.0

    return feats, vein_skel, vein_binary

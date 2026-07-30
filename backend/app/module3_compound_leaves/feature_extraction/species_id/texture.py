"""
VedaVision — Texture Features  (shadow-robust revision)
========================================================
GLCM + LBP texture descriptors extracted from the ENHANCED (sharpened) image.

Shadow-robustness changes vs previous version
---------------------------------------------
GLCM:
  OLD — bounding-box crop only; background/shadow pixels set to 0
        → zero creates a massive artificial contrast edge in the GLCM
  NEW — excluded pixels (outside mask OR suspiciously dark) filled with
        the MEDIAN foreground value instead of zero.
        Median fill is co-occurrence-neutral: a pixel surrounded by
        median-valued neighbours contributes nothing unusual to any GLCM
        property.  Shadow pixels that sneak past the mask are treated the
        same way: they are clamped toward the median before GLCM runs.

  ADDED: confident_mask — pixels with V < 40 inside the foreground are
         flagged as probable shadow contamination and median-filled rather
         than included in the GLCM.  This threshold is conservative
         (V < 40 means very dark, not just shaded).

LBP:
  UNCHANGED — LBP compares each pixel to its neighbours RELATIVELY.
  A shadow pixel surrounded by other shadow pixels produces the same
  LBP code as a bright pixel in a bright region.  LBP is inherently
  illumination-invariant and needs no shadow-specific modification.
  Histogram is still computed on foreground pixels only (px_mask gate).
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from app.module3_compound_leaves.preprocessing.config import LBP_RADIUS, LBP_POINTS, GLCM_DIST, GLCM_ANGLES

# ---------------------------------------------------------------------------
# Shadow-confidence threshold
# ---------------------------------------------------------------------------
_SHADOW_V_THRESH = 40   # HSV Value below this inside the mask → probable shadow
                        # Conservative: real deep-green leaves have V ≈ 60-100

_BOTANICAL_SENTINEL = -1.0


def _extract_botanical_texture_features(img_sharp_bgr: np.ndarray,
                                         confident_mask: np.ndarray) -> dict:
    """
    BOTANICAL / HANDCRAFTED addition, prefixed `botanical_`.
    A single backup cross-check feature, deliberately minimal — the
    primary signals for the traits this could address (glossiness,
    reticulation density) already live in colour.py and vein.py; this is
    a secondary confirmation, not a new primary signal, so scope is kept
    small here rather than duplicating machinery.

    botanical_local_contrast_variance — spatial variance of local
    micro-contrast (std of L in small windows) across the leaf. Glossy
    surfaces (Kattakumanjal) show patchy local contrast from specular
    highlights; matte surfaces (Kalawal) and leaves with a fine, uniform
    reticulate network (Siyabala) show more spatially uniform local
    contrast. Backs up botanical_gloss_* in colour.py.

    STATUS: not yet visually validated on real photos.
    """
    if confident_mask.sum() < 200:
        return {"botanical_local_contrast_variance": _BOTANICAL_SENTINEL}
    try:
        lab_l = cv2.cvtColor(img_sharp_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        # local std via a small sliding window (mean of squares - square of mean)
        k = 7
        mean = cv2.blur(lab_l, (k, k))
        mean_sq = cv2.blur(lab_l * lab_l, (k, k))
        local_std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
        vals = local_std[confident_mask]
        # variance OF the local-contrast map -- patchiness, not overall texture level
        variance_of_local_contrast = float(np.var(vals)) if len(vals) else 0.0
        return {"botanical_local_contrast_variance": variance_of_local_contrast}
    except Exception:
        return {"botanical_local_contrast_variance": _BOTANICAL_SENTINEL}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_texture_features(img_sharp_bgr: np.ndarray,
                              leaf_mask: np.ndarray) -> dict:
    """
    Parameters
    ----------
    img_sharp_bgr : enhanced BGR uint8 image (output of enhance.py)
    leaf_mask     : uint8 binary mask (255 = foreground)

    Returns
    -------
    dict — 8 GLCM stats (4 props × mean+std, rotation-invariant)
           + (LBP_POINTS+2) histogram bins + lbp_mean + lbp_std
           = same dimensionality as previous version

    Shadow robustness
    -----------------
    Excluded pixels (background OR dark shadow) are filled with the median
    foreground grey value BEFORE GLCM.  This neutralises artificial contrast
    edges that zero-fill would create at the shadow/leaf boundary.
    LBP is illumination-invariant by design — no change needed.
    """
    gray    = cv2.cvtColor(img_sharp_bgr, cv2.COLOR_BGR2GRAY)
    px_mask = leaf_mask > 0
    if px_mask.sum() < 50:
        return {}

    feats: dict = {}

    # ── Confident foreground mask ─────────────────────────────────────────
    # Mark very dark foreground pixels as shadow-contaminated.
    hsv_v    = cv2.cvtColor(img_sharp_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    shadow   = (hsv_v < _SHADOW_V_THRESH) & px_mask   # dark pixels inside mask
    confident_mask = px_mask & ~shadow                  # boolean: true foreground

    # Fall back to full foreground mask if confident region is too small
    if confident_mask.sum() < 50:
        confident_mask = px_mask

    # ── GLCM — bounding box crop + median fill ────────────────────────────
    ys, xs = np.where(confident_mask)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    gray_crop   = gray[y1:y2, x1:x2].copy()
    conf_crop   = confident_mask[y1:y2, x1:x2]

    # Median of confident foreground pixels — our neutral fill value
    median_val  = int(np.median(gray_crop[conf_crop]))

    # Fill pixels that are NOT confident foreground with median value.
    # This covers: (a) white/shadow background outside leaf,
    #              (b) dark shadow pixels inside leaf boundary.
    # Using median instead of zero prevents artificial contrast spikes
    # in the GLCM co-occurrence matrix.
    gray_crop[~conf_crop] = median_val

    # Quantise to 64 levels — sufficient to distinguish dark veins from lamina
    gray_q = (gray_crop // 4).astype(np.uint8)

    glcm = graycomatrix(
        gray_q,
        distances=GLCM_DIST,
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True,
    )
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop)   # (n_dist, n_angles)
        feats[f"glcm_{prop}_mean"] = float(vals.mean())
        feats[f"glcm_{prop}_std"]  = float(vals.std())

    # ── LBP — inherently shadow-robust, no modification needed ───────────
    # LBP encodes relative brightness patterns; shadow is a global darkening
    # so local patterns (vein edges, cell boundaries) are preserved.
    lbp      = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins   = LBP_POINTS + 2
    lbp_vals = lbp[px_mask]                  # foreground pixels only
    lbp_hist, _ = np.histogram(lbp_vals, bins=n_bins, range=(0, n_bins))
    lbp_hist     = lbp_hist / (lbp_hist.sum() + 1e-6)
    for bi, bv in enumerate(lbp_hist):
        feats[f"lbp_{bi:02d}"] = float(bv)
    feats["lbp_mean"] = float(lbp_vals.mean()) if len(lbp_vals) else 0.0
    feats["lbp_std"]  = float(lbp_vals.std())  if len(lbp_vals) else 0.0

    # ── NEW: botanical local-contrast-variance (glossiness backup) ────────
    try:
        feats.update(_extract_botanical_texture_features(img_sharp_bgr, confident_mask))
    except Exception:
        feats["botanical_local_contrast_variance"] = _BOTANICAL_SENTINEL

    return feats
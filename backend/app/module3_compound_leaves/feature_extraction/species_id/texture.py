
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from app.module3_compound_leaves.preprocessing.config import LBP_RADIUS, LBP_POINTS, GLCM_DIST, GLCM_ANGLES


# Shadow-confidence threshold

_SHADOW_V_THRESH = 40   # HSV Value below this inside the mask → probable shadow
                        # Conservative: real deep-green leaves have V ≈ 60-100

_BOTANICAL_SENTINEL = -1.0


def _extract_botanical_texture_features(img_sharp_bgr: np.ndarray,
                                         confident_mask: np.ndarray) -> dict:
    
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



# Public API

def extract_texture_features(img_sharp_bgr: np.ndarray,
                              leaf_mask: np.ndarray) -> dict:
    
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
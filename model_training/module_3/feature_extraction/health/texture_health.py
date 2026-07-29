"""
Texture-degradation features for the health branch.

HARD RULE (unchanged from every other module in this branch): computed on
masked_raw (unenhanced) -- never img_sharp. Enhancement (bilateral/CLAHE/
unsharp) actively smooths and sharpens texture, which would corrupt exactly
the surface-roughening signal this module is trying to measure.

Rationale (why texture at all): necrosis, chlorosis, fungal lesions, and
insect-scarring all locally disrupt the smooth, regular cell-surface
pattern of healthy lamina tissue -- wrinkling, pitting, and patchy
discoloration all show up as texture irregularity even before/alongside
colour changes. This was the highest-priority gap identified after the
colour/boundary/hole/scar/miner feature groups plateaued around
Spearman rho ~0.19-0.23 against severity order (see project notes) --
none of the existing groups measure surface texture directly.

Shadow-robust GLCM (mirrors feature_extraction/species_id/texture.py):
  Zero/black-fill of excluded pixels creates an artificial contrast edge
  at the mask/shadow boundary that pollutes every GLCM property. Instead,
  excluded pixels (background OR suspiciously dark inside the mask) are
  filled with the MEDIAN foreground grey value -- co-occurrence-neutral,
  since a pixel surrounded by median-valued neighbours contributes
  nothing unusual to any GLCM statistic.

LBP is left unmodified -- it encodes relative (not absolute) brightness
patterns between a pixel and its neighbours, so a uniform darkening
(shadow) does not change the LBP code. Histogram is computed on the full
foreground mask, not the shadow-excluded "confident" mask, since LBP does
not need the exclusion.

Parameters (GLCM distances/angles, LBP radius/points) are pulled from the
SAME shared preprocessing/config.py the species-ID branch's texture.py
already uses -- there is no health-only config module, and reusing the
one source of truth keeps both branches' texture descriptors comparable
and avoids a second place for these numbers to drift out of sync.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from preprocessing.config import LBP_RADIUS, LBP_POINTS, GLCM_DIST, GLCM_ANGLES

SENTINEL = -1.0

_SHADOW_V_THRESH = 40      # HSV Value below this inside the mask -> probable
                           # shadow contamination, same threshold and
                           # justification as species_id/texture.py
_MIN_FOREGROUND_PX = 200   # below this, GLCM/LBP are unstable -- return
                           # sentinels rather than a noisy number
_GLCM_LEVELS = 64          # matches species_id/texture.py's own quantisation
                           # (gray // 4 -> 64 levels); not present in the
                           # shared config, kept local + explicit here


def extract_texture_health_features(masked_raw_bgr: np.ndarray,
                                     mask_final: np.ndarray) -> dict:
    """
    Parameters
    ----------
    masked_raw_bgr : unenhanced BGR image, background already zeroed
                      (output of run_health_pipeline's masked_raw step)
    mask_final     : uint8 or bool array, foreground mask (same one used
                      by every other health feature module)

    Returns
    -------
    dict of texture_h_* features:
      texture_h_glcm_contrast_mean / _std
      texture_h_glcm_homogeneity_mean / _std
      texture_h_glcm_energy_mean / _std
      texture_h_glcm_correlation_mean / _std
      texture_h_lbp_uniformity   -- fraction of pixels in the dominant
                                     (most common) LBP bin; healthy smooth
                                     tissue -> a few bins dominate, ragged/
                                     necrotic tissue -> more spread out, so
                                     LOWER uniformity ~ MORE degradation
      texture_h_lbp_entropy      -- Shannon entropy of the LBP histogram;
                                     complementary to uniformity, HIGHER
                                     entropy ~ MORE degradation
    """
    mask_bool = mask_final.astype(bool)
    if mask_bool.sum() < _MIN_FOREGROUND_PX:
        return {
            "texture_h_glcm_contrast_mean": SENTINEL,
            "texture_h_glcm_contrast_std": SENTINEL,
            "texture_h_glcm_homogeneity_mean": SENTINEL,
            "texture_h_glcm_homogeneity_std": SENTINEL,
            "texture_h_glcm_energy_mean": SENTINEL,
            "texture_h_glcm_energy_std": SENTINEL,
            "texture_h_glcm_correlation_mean": SENTINEL,
            "texture_h_glcm_correlation_std": SENTINEL,
            "texture_h_lbp_uniformity": SENTINEL,
            "texture_h_lbp_entropy": SENTINEL,
        }

    gray = cv2.cvtColor(masked_raw_bgr, cv2.COLOR_BGR2GRAY)
    feats: dict = {}

    # ---- shadow-confident subset (GLCM only) ------------------------------
    hsv_v = cv2.cvtColor(masked_raw_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    shadow = (hsv_v < _SHADOW_V_THRESH) & mask_bool
    confident_mask = mask_bool & ~shadow
    if confident_mask.sum() < _MIN_FOREGROUND_PX:
        confident_mask = mask_bool  # fall back if shadow-exclusion left too little

    ys, xs = np.where(confident_mask)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    gray_crop = gray[y1:y2, x1:x2].copy()
    conf_crop = confident_mask[y1:y2, x1:x2]

    median_val = int(np.median(gray_crop[conf_crop]))
    gray_crop[~conf_crop] = median_val  # median-fill, NOT zero -- see module docstring

    # Quantise to 64 levels, same convention as species_id/texture.py
    gray_q = (gray_crop // 4).astype(np.uint8)
    glcm = graycomatrix(
        gray_q,
        distances=GLCM_DIST,
        angles=GLCM_ANGLES,
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop)
        feats[f"texture_h_glcm_{prop}_mean"] = float(vals.mean())
        feats[f"texture_h_glcm_{prop}_std"] = float(vals.std())

    # ---- LBP (illumination-invariant, no shadow handling needed) ----------
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    lbp_vals = lbp[mask_bool]
    hist, _ = np.histogram(lbp_vals, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-9)

    feats["texture_h_lbp_uniformity"] = float(hist.max())
    nz = hist[hist > 0]
    feats["texture_h_lbp_entropy"] = float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0

    return feats

"""
VedaVision — Whole-Leaf Structural Features
============================================
Captures the arrangement of leaflets in the compound-leaf structure.
All features are dimensionless ratios — independent of camera distance and zoom.

Removed (Fix 3 — scale-dependent or biologically plastic):
    n_leaflets            plastic: 7 vs 9 on same Moringa branch
    leaf_type_inferred    derived from n_leaflets + it's a string
    whole_bbox_w/h        raw px — scale-dependent
    whole_area_*          raw px areas — scale-dependent
    whole_spacing_mean    raw px spacing — scale-dependent
    whole_rachis_len_est  bbox height px — scale-dependent

Kept / Added (all dimensionless):
    whole_aspect          bbox W/H ratio  (trifoliate > 1, pinnate < 0.5)
    area_cv               std/mean of component areas (size uniformity)
    area_max_min_ratio    largest/smallest component area ratio
    symmetry_lr_ratio     left-area / total-area (0.5 = symmetric)
    symmetry_score        1 − 2|ratio − 0.5|  (1.0 = perfect bilateral)
    spacing_cv            std/mean of inter-component y-spacing (regularity)
"""

import cv2
import numpy as np
from preprocessing.config import TARGET_LONG, MIN_COMP_FRAC


def extract_whole_leaf_features(mask_full: np.ndarray) -> dict:
    """
    Parameters
    ----------
    mask_full : uint8 binary mask (255 = foreground); may have multiple components

    Returns
    -------
    dict with keys: whole_aspect, area_cv, area_max_min_ratio,
                    symmetry_lr_ratio, symmetry_score, spacing_cv
    """
    feats     = {}
    min_area  = TARGET_LONG * TARGET_LONG * MIN_COMP_FRAC

    # ── Whole-leaf bounding box aspect ratio ──────────────────────────────────
    cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        wx, wy, ww, wh = cv2.boundingRect(np.vstack(cnts))
        feats["whole_aspect"] = float(ww) / wh if wh > 0 else 0.0
    else:
        feats["whole_aspect"] = 0.0

    # ── Component-level structural features ───────────────────────────────────
    n_cc, lbl_cc, stats_cc, _ = cv2.connectedComponentsWithStats(mask_full)
    sig = [i for i in range(1, n_cc)
           if stats_cc[i, cv2.CC_STAT_AREA] > min_area]

    if len(sig) >= 2:
        areas   = np.array([float(stats_cc[i, cv2.CC_STAT_AREA]) for i in sig])
        cx_vals = [stats_cc[i, cv2.CC_STAT_LEFT] + stats_cc[i, cv2.CC_STAT_WIDTH]  / 2
                   for i in sig]
        cy_vals = [stats_cc[i, cv2.CC_STAT_TOP]  + stats_cc[i, cv2.CC_STAT_HEIGHT] / 2
                   for i in sig]

        feats["area_cv"]            = float(areas.std() / (areas.mean() + 1e-6))
        feats["area_max_min_ratio"] = float(areas.max() / (areas.min() + 1e-6))

        # Bilateral symmetry
        cx_med  = np.median(cx_vals)
        a_left  = sum(a for cx, a in zip(cx_vals, areas) if cx <  cx_med)
        a_right = sum(a for cx, a in zip(cx_vals, areas) if cx >= cx_med)
        total_lr = a_left + a_right
        feats["symmetry_lr_ratio"] = float(a_left / total_lr) if total_lr > 0 else 0.5
        feats["symmetry_score"]    = float(1 - 2 * abs(feats["symmetry_lr_ratio"] - 0.5))

        # Spacing regularity
        if len(cy_vals) >= 3:
            spacings = np.diff(np.sort(cy_vals))
            feats["spacing_cv"] = float(spacings.std() / (spacings.mean() + 1e-6))
        else:
            feats["spacing_cv"] = float("nan")
    else:
        # Single-component mask (trifoliate fused or low-contrast) — use safe defaults
        feats.update({
            "area_cv"           : float("nan"),
            "area_max_min_ratio": float("nan"),
            "symmetry_lr_ratio" : 0.5,
            "symmetry_score"    : 1.0,
            "spacing_cv"        : float("nan"),
        })

    return feats

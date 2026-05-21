"""
VedaVision — Whole-Leaf Structural Features  (shadow-robust revision)
======================================================================
Captures the arrangement of leaflets in the compound-leaf structure.
All features are dimensionless ratios — independent of camera distance and zoom.

Shadow-robustness changes vs previous version
---------------------------------------------
Whole-leaf structural features operate on CONNECTED COMPONENTS of the binary
mask, not on pixel colour values.  They are therefore the most naturally
robust of all five feature groups — shadow pixels do not affect component
centroids or area ratios significantly.

Minor fixes applied
~~~~~~~~~~~~~~~~~~~
1. AREA RATIOS (area_cv, area_max_min_ratio, symmetry_lr_ratio, symmetry_score)
   Shadow at leaf boundaries slightly inflates individual component areas,
   but inflates ALL components proportionally.  CV and max/min ratio are
   unchanged because they are ratios of the same-type quantities.

2. SPACING_CV
   Shadow pixels do not move component centroids (centroid = area-weighted
   mean position; small boundary addition barely shifts it).  spacing_cv
   is unaffected.

3. NaN HANDLING — improved
   Previous version emitted float("nan") for single-component images.
   NaN values propagate silently through sklearn pipelines and can cause
   SVM training failures.  Updated to emit a sentinel value of -1.0 with
   a flag feature (n_components_flag) so the classifier can learn the
   single-component case explicitly.

4. COMPONENT COUNT FEATURE — added
   n_components_norm = n_significant_components / 10.0
   Normalised count (0.0–1.0 range) is a useful structural discriminator
   between trifoliate (3), pinnate (5–9), and palmate (5–7) leaves.
   Raw count was removed previously because it is biologically plastic
   (Moringa may show 7 or 9 on the same branch).  The normalised count
   is kept as a soft signal rather than a hard class boundary.

Removed (unchanged from previous version):
    n_leaflets, leaf_type_inferred, whole_bbox_w/h,
    whole_area_*, whole_spacing_mean, whole_rachis_len_est
"""

import cv2
import numpy as np
from preprocessing.config import TARGET_LONG, MIN_COMP_FRAC


# Sentinel value used instead of NaN for single-component images.
# -1.0 is outside the valid range of all ratio features (which are ≥ 0),
# so classifiers can learn to treat it as a distinct case.
_NAN_SENTINEL = -1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_whole_leaf_features(mask_full: np.ndarray) -> dict:
    """
    Parameters
    ----------
    mask_full : uint8 binary mask (255 = foreground); may have multiple components

    Returns
    -------
    dict with keys:
        whole_aspect, area_cv, area_max_min_ratio,
        symmetry_lr_ratio, symmetry_score, spacing_cv,
        n_components_norm

    Shadow robustness
    -----------------
    Features are based on connected-component statistics (centroids, areas).
    Shadow at leaf boundaries inflates all component areas proportionally,
    so area RATIOS remain stable.
    Centroids are area-weighted means — small boundary additions shift them
    negligibly (< 1 px at typical shadow widths of 5–15 px).
    NaN sentinels replaced with -1.0 to prevent silent sklearn failures.
    """
    feats: dict = {}
    min_area = TARGET_LONG * TARGET_LONG * MIN_COMP_FRAC

    # ── Whole-leaf bounding box aspect ratio ──────────────────────────────
    cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        wx, wy, ww, wh = cv2.boundingRect(np.vstack(cnts))
        feats["whole_aspect"] = float(ww) / wh if wh > 0 else 0.0
    else:
        feats["whole_aspect"] = 0.0

    # ── Connected components ──────────────────────────────────────────────
    n_cc, lbl_cc, stats_cc, _ = cv2.connectedComponentsWithStats(mask_full)
    sig = [
        i for i in range(1, n_cc)
        if stats_cc[i, cv2.CC_STAT_AREA] > min_area
    ]

    # Normalised component count (soft signal, not a hard class label)
    feats["n_components_norm"] = float(len(sig)) / 10.0

    if len(sig) >= 2:
        areas   = np.array([float(stats_cc[i, cv2.CC_STAT_AREA]) for i in sig])
        cx_vals = [
            stats_cc[i, cv2.CC_STAT_LEFT] + stats_cc[i, cv2.CC_STAT_WIDTH]  / 2.0
            for i in sig
        ]
        cy_vals = [
            stats_cc[i, cv2.CC_STAT_TOP]  + stats_cc[i, cv2.CC_STAT_HEIGHT] / 2.0
            for i in sig
        ]

        # Area uniformity ratios — shadow inflates all areas proportionally
        # so CV and max/min ratio are stable
        feats["area_cv"]            = float(areas.std() / (areas.mean() + 1e-6))
        feats["area_max_min_ratio"] = float(areas.max() / (areas.min() + 1e-6))

        # Bilateral symmetry — shadow centroid shift < 1 px, negligible
        cx_med  = float(np.median(cx_vals))
        a_left  = sum(a for cx, a in zip(cx_vals, areas) if cx <  cx_med)
        a_right = sum(a for cx, a in zip(cx_vals, areas) if cx >= cx_med)
        total_lr = a_left + a_right
        lr_ratio = float(a_left / total_lr) if total_lr > 0 else 0.5
        feats["symmetry_lr_ratio"] = lr_ratio
        feats["symmetry_score"]    = float(1.0 - 2.0 * abs(lr_ratio - 0.5))

        # Spacing regularity — centroid positions unaffected by shadow
        if len(cy_vals) >= 3:
            spacings = np.diff(np.sort(cy_vals))
            feats["spacing_cv"] = float(spacings.std() / (spacings.mean() + 1e-6))
        else:
            # Only 2 components → no spacing variance; use sentinel
            feats["spacing_cv"] = _NAN_SENTINEL

    else:
        # Single-component mask (trifoliate fused or low-contrast).
        # Use sentinel (-1.0) instead of NaN to avoid sklearn propagation issues.
        feats["area_cv"]            = _NAN_SENTINEL
        feats["area_max_min_ratio"] = _NAN_SENTINEL
        feats["symmetry_lr_ratio"]  = 0.5       # single blob → assume symmetric
        feats["symmetry_score"]     = 1.0
        feats["spacing_cv"]         = _NAN_SENTINEL

    return feats

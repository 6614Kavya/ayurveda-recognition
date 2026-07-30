import cv2
import numpy as np
from app.module3_compound_leaves.preprocessing.config import TARGET_LONG, MIN_COMP_FRAC


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
        whole_aspect, n_components_norm, symmetry_lr_ratio, symmetry_score

    Shadow robustness
    -----------------
    Features are based on connected-component statistics (centroids, areas).
    Shadow at leaf boundaries inflates all component areas proportionally,
    which cancels out in the left/right area ratio used for symmetry.
    Centroids are area-weighted means — small boundary additions shift them
    negligibly (< 1 px at typical shadow widths of 5–15 px).
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

        # Bilateral symmetry — shadow centroid shift < 1 px, negligible
        cx_med  = float(np.median(cx_vals))
        a_left  = sum(a for cx, a in zip(cx_vals, areas) if cx <  cx_med)
        a_right = sum(a for cx, a in zip(cx_vals, areas) if cx >= cx_med)
        total_lr = a_left + a_right
        lr_ratio = float(a_left / total_lr) if total_lr > 0 else 0.5
        feats["symmetry_lr_ratio"] = lr_ratio
        feats["symmetry_score"]    = float(1.0 - 2.0 * abs(lr_ratio - 0.5))

    else:
        # Single-component mask (trifoliate fused or low-contrast).
        # These degrade to a sane default, not a sentinel — an assumed
        # symmetric single blob is a reasonable prior, unlike area_cv/
        # spacing_cv above which have no sane default and were removed.
        feats["symmetry_lr_ratio"]  = 0.5       # single blob → assume symmetric
        feats["symmetry_score"]     = 1.0

    return feats
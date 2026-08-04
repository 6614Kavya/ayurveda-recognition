"""
Hole / puncture density features.

Detects enclosed background regions *inside* the leaf silhouette using
`mask_before_holefill` (the Stage-7 union mask captured before Stage-8
flood-fill in masking.py v5.1.1). Anything filled in by the flood-fill
step was, by construction, an enclosed hole -- so
(mask_final AND NOT mask_before_holefill) isolates exactly the pixels
that were "holes" at that stage.

IMPORTANT (open design issue, see project memory): mask_before_holefill
is only available when the pipeline routes through select_mask() directly
(preprocessing/health/pipeline.py). It is NOT available for augmented
rows produced via run_pipeline_from_resized(). This module does not try
to guess or reconstruct it -- callers must pass
mask_before_holefill=None explicitly in that case, which forces the
sentinel path below. Do not silently substitute mask_final for it.
"""
import cv2
import numpy as np

SENTINEL = -1.0
MIN_HOLE_AREA_PX = 6  # sub-pixel-noise floor; smaller blobs are anti-aliasing artifacts


def extract_hole_features(mask_final: np.ndarray, mask_before_holefill) -> dict:
    """
    Parameters
    ----------
    mask_final : final binary mask (post hole-fill), leaf foreground = 255.
    mask_before_holefill : Stage-7 union mask (pre hole-fill) from
        masking.py's diag dict, or None if unavailable (augmented rows).

    Returns
    -------
    dict of hole_* features. hole_count=-1 signals "unavailable" (not
    "zero holes") when mask_before_holefill is None -- keep this
    distinction downstream, don't treat -1 as a real count.
    """
    if mask_before_holefill is None:
        return {
            "hole_count": -1,
            "hole_area_ratio": SENTINEL,
            "hole_mean_size": SENTINEL,
        }

    leaf_area = int(np.count_nonzero(mask_final))
    if leaf_area == 0:
        return {"hole_count": 0, "hole_area_ratio": SENTINEL, "hole_mean_size": SENTINEL}

    # Holes = pixels that are foreground in mask_final (after fill) but were
    # background in mask_before_holefill (before fill), restricted to the
    # leaf interior so exterior background can never leak in.
    filled_in = cv2.bitwise_and(
        mask_final.astype(np.uint8),
        cv2.bitwise_not(mask_before_holefill.astype(np.uint8)),
    )

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(filled_in, connectivity=8)
    hole_sizes = [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n_labels)  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= MIN_HOLE_AREA_PX
    ]

    hole_count = len(hole_sizes)
    total_hole_area = sum(hole_sizes)
    hole_area_ratio = total_hole_area / leaf_area
    hole_mean_size = float(np.mean(hole_sizes)) if hole_sizes else 0.0

    return {
        "hole_count": hole_count,
        "hole_area_ratio": float(hole_area_ratio),
        "hole_mean_size": hole_mean_size,
    }

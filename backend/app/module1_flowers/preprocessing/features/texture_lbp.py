"""
BLOCK 5C — Local Binary Pattern Texture Features (26 total)

Uniform LBP histogram with `lbp_n_points + 2` bins (24 + 2 = 26 when
lbp_n_points=24), computed only over pixels inside the flower mask.
"""

import numpy as np
from skimage.feature import local_binary_pattern

from ..config import CFG


def extract_lbp_features(roi_gray: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    masked_gray = roi_gray.copy()
    masked_gray[roi_mask == 0] = 0

    lbp = local_binary_pattern(masked_gray, P=CFG['lbp_n_points'], R=CFG['lbp_radius'], method='uniform')

    n_bins = CFG['lbp_n_points'] + 2
    lbp_flower = lbp[roi_mask > 0]
    hist, _ = np.histogram(lbp_flower, bins=n_bins, range=(0, n_bins), density=True)

    return hist.astype(np.float32)

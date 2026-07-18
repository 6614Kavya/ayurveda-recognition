"""
BLOCK 5A — Color Features (105 total)

  HSV histograms (H, S, V; `hist_bins` bins each)  -> 3 * hist_bins = 96
  LAB per-channel mean / std / skew (L, A, B)        -> 3 * 3        = 9
                                                          -----
                                                          105 (when hist_bins=32)
"""

import cv2
import numpy as np
from scipy.stats import skew

from ..config import CFG


def extract_color_features(roi_rgb: np.ndarray, roi_mask: np.ndarray,
                            bins: int = CFG['hist_bins']) -> np.ndarray:
    flower_pixels_rgb = roi_rgb[roi_mask > 0]
    if len(flower_pixels_rgb) == 0:
        return np.zeros(bins * 3 + 9)

    flower_img = flower_pixels_rgb.reshape(-1, 1, 3).astype(np.uint8)
    hsv_pixels = cv2.cvtColor(flower_img, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    h_hist, _ = np.histogram(hsv_pixels[:, 0], bins=bins, range=(0, 180), density=True)
    s_hist, _ = np.histogram(hsv_pixels[:, 1], bins=bins, range=(0, 256), density=True)
    v_hist, _ = np.histogram(hsv_pixels[:, 2], bins=bins, range=(0, 256), density=True)
    color_hist = np.concatenate([h_hist, s_hist, v_hist])

    lab_pixels = cv2.cvtColor(flower_img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab_features = []
    for ch in range(3):
        channel = lab_pixels[:, ch]
        lab_features.extend([float(channel.mean()), float(channel.std()), float(skew(channel))])

    return np.concatenate([color_hist, lab_features])

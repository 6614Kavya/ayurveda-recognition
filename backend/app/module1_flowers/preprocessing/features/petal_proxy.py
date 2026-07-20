"""
BLOCK 5F — Petal Count Proxy (4 total)

Casts rays outward from the mask centroid at 360 angles, smooths the
resulting radius profile, and counts peaks as a rough proxy for petal
count. Cheaper and more robust to noise than full petal segmentation
(see petal_morphometrics.py), used as a complementary signal.

Features: petal_count, radii.mean, radii.std, radii.max
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


def extract_petal_proxy(roi_mask: np.ndarray) -> np.ndarray:
    h, w = roi_mask.shape
    cy, cx = h // 2, w // 2

    n_angles = 360
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    max_r = min(cy, cx)
    radii = np.zeros(n_angles)

    for i, angle in enumerate(angles):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for r in range(1, max_r):
            px = int(cx + r * cos_a)
            py = int(cy + r * sin_a)
            if not (0 <= px < w and 0 <= py < h):
                radii[i] = r - 1
                break
            if roi_mask[py, px] == 0:
                radii[i] = r - 1
                break
        else:
            radii[i] = max_r

    kernel_size = 15
    radii_smooth = uniform_filter1d(radii, size=kernel_size, mode='wrap')

    peaks, _ = find_peaks(radii_smooth, height=radii_smooth.mean() * 0.7, distance=n_angles // 16)
    petal_count = len(peaks)

    return np.array([float(petal_count), float(radii.mean()), float(radii.std()), float(radii.max())])

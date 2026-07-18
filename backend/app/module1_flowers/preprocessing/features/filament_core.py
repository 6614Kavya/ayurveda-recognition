"""
BLOCK 5J — Filament & Core-Contrast Features (9 total)

Compares the flower's inner zone (near center, r_norm < 0.4 — where
stamens/filaments live) against its outer zone (r_norm >= 0.6 — petal
tips) in hue/value/saturation, plus contour jaggedness and radial-profile
peak statistics that pick up on filament-induced texture.

Features: hue_diff, val_diff, sat_diff, jaggedness, fine_peaks,
          coarse_peaks, peak_ratio, edge_density, radial_cv
"""

import cv2
import numpy as np
from scipy.signal import find_peaks


def extract_filament_core_features(roi_rgb: np.ndarray, roi_gray: np.ndarray,
                                    roi_mask: np.ndarray) -> np.ndarray:
    h, w = roi_mask.shape
    ys, xs = np.where(roi_mask > 0)
    if len(ys) == 0:
        return np.zeros(9)

    cy, cx = ys.mean(), xs.mean()
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = dist.max() if dist.size else 1.0
    r_norm = dist / (max_r + 1e-6)

    inner = r_norm < 0.4
    outer = r_norm >= 0.6

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    def zone_mean(channel, zone_mask):
        vals = channel[ys[zone_mask], xs[zone_mask]]
        return float(vals.mean()) if vals.size else 0.0

    inner_hue, outer_hue = zone_mean(h_ch, inner), zone_mean(h_ch, outer)
    inner_val, outer_val = zone_mean(v_ch, inner), zone_mean(v_ch, outer)
    inner_sat, outer_sat = zone_mean(s_ch, inner), zone_mean(s_ch, outer)

    hue_diff = abs(inner_hue - outer_hue)
    hue_diff = min(hue_diff, 180 - hue_diff)
    val_diff = inner_val - outer_val
    sat_diff = inner_sat - outer_sat

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([hue_diff, val_diff, sat_diff, 0, 0, 0, 0, 0, 0])
    c = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(c, True)
    hull_perimeter = cv2.arcLength(cv2.convexHull(c), True)
    jaggedness = perimeter / (hull_perimeter + 1e-6)

    n_angles = 360
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cxi, cyi = int(round(cx)), int(round(cy))
    max_search = min(h, w) // 2
    radii = np.zeros(n_angles)
    for i, a in enumerate(angles):
        ca, sa = np.cos(a), np.sin(a)
        r_found = 0
        for r in range(1, max_search):
            px, py = int(cxi + r * ca), int(cyi + r * sa)
            if not (0 <= px < w and 0 <= py < h) or roi_mask[py, px] == 0:
                r_found = r - 1
                break
            r_found = r
        radii[i] = r_found

    def count_peaks(kernel_size):
        pad = kernel_size // 2
        padded = np.concatenate([radii[-pad:], radii, radii[:pad]]) if pad > 0 else radii
        kernel = np.ones(kernel_size) / kernel_size
        smooth = np.convolve(padded, kernel, mode='valid')
        peaks, _ = find_peaks(smooth, distance=max(2, n_angles // 90))
        return len(peaks)

    fine_peaks = count_peaks(3)
    coarse_peaks = count_peaks(21)
    peak_ratio = fine_peaks / (coarse_peaks + 1e-6)

    masked_gray = roi_gray.copy()
    masked_gray[roi_mask == 0] = 0
    edges = cv2.Canny(masked_gray, 50, 150)
    edge_density = float(np.sum((edges > 0) & (roi_mask > 0))) / (np.sum(roi_mask > 0) + 1e-6)

    radial_cv = float(radii.std() / (radii.mean() + 1e-6))

    return np.array([hue_diff, val_diff, sat_diff, jaggedness,
                      float(fine_peaks), float(coarse_peaks), peak_ratio,
                      edge_density, radial_cv])

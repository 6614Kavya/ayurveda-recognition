"""
BLOCK 5L — Vein Ridge & Center Aperture Features (5 total)

Samples pixel intensities around concentric rings (at 35%, 50%, 65% of
max radius) and counts oscillations in each ring — radial veins create a
regular ripple pattern as the ring crosses light ridge / dark groove /
light ridge repeatedly. Also compares brightness at the very center
(possible aperture/throat of the flower) against the mid-zone.

Features: vein_oscillation_count, vein_ring_cv, center_brightness,
          center_edge_density, center_vs_midzone_contrast
"""

import cv2
import numpy as np
from scipy.signal import find_peaks

from .utils import circular_smooth


def _sample_ring(gray: np.ndarray, mask: np.ndarray, cx: float, cy: float,
                  radius: float, n_angles: int = 180) -> np.ndarray:
    """Sample `gray` at `n_angles` points around a circle of given radius,
    centered at (cx, cy). Points off-mask or off-image are filled with the
    ring's own mean so they don't distort the oscillation signal."""
    h, w = gray.shape
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    xs = (cx + radius * np.cos(angles)).astype(int)
    ys = (cy + radius * np.sin(angles)).astype(int)

    vals = np.full(n_angles, np.nan)
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    idx = np.where(in_bounds)[0]
    on_flower = mask[ys[idx], xs[idx]] > 0
    idx = idx[on_flower]
    vals[idx] = gray[ys[idx], xs[idx]]

    valid_frac = np.isnan(vals).mean()
    if valid_frac > 0.5:
        return None
    mean_val = np.nanmean(vals)
    vals = np.where(np.isnan(vals), mean_val, vals)
    return vals


def extract_vein_center_features(roi_rgb: np.ndarray, roi_gray: np.ndarray,
                                  roi_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(roi_mask > 0)
    if len(ys) == 0:
        return np.zeros(5)

    cy, cx = ys.mean(), xs.mean()
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = dist.max() if dist.size else 1.0

    osc_counts, ring_cvs = [], []
    for frac in (0.35, 0.5, 0.65):
        ring = _sample_ring(roi_gray, roi_mask, cx, cy, frac * max_r)
        if ring is None:
            continue
        smooth = circular_smooth(ring, 5)
        peaks, _ = find_peaks(smooth, distance=6, prominence=max(1.0, smooth.std() * 0.3))
        osc_counts.append(len(peaks))
        m = ring.mean()
        ring_cvs.append(ring.std() / m if m > 1e-6 else 0.0)

    vein_oscillation_count = float(np.mean(osc_counts)) if osc_counts else 0.0
    vein_ring_cv = float(np.mean(ring_cvs)) if ring_cvs else 0.0

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    v_ch = hsv[..., 2]
    r_norm = dist / (max_r + 1e-6)

    inner_zone = r_norm < 0.15
    mid_zone = (r_norm >= 0.4) & (r_norm < 0.6)

    center_brightness = float(v_ch[ys[inner_zone], xs[inner_zone]].mean()) if inner_zone.any() else 0.0
    mid_brightness = float(v_ch[ys[mid_zone], xs[mid_zone]].mean()) if mid_zone.any() else 0.0
    center_vs_midzone_contrast = center_brightness - mid_brightness

    masked_gray = roi_gray.copy()
    masked_gray[roi_mask == 0] = 0
    edges = cv2.Canny(masked_gray, 50, 150)
    inner_mask_img = np.zeros_like(roi_mask)
    inner_mask_img[ys[inner_zone], xs[inner_zone]] = 255
    inner_px = np.sum(inner_mask_img > 0)
    center_edge_density = float(np.sum((edges > 0) & (inner_mask_img > 0))) / (inner_px + 1e-6)

    return np.array([
        vein_oscillation_count, vein_ring_cv,
        center_brightness, center_edge_density, center_vs_midzone_contrast,
    ])

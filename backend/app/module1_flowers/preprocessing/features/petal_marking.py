"""
BLOCK 5N — Petal Marking / Nectar-Guide Color Features (4 total)

  marking_fraction, marking_mean_rnorm,
  radial_hue_std, dominant_hue                             -> 4
                                                                --
                                                                4

  Targets karawila vs hendirikka — presence of contrasting streaks
  radiating from the petal base, relative to the flower's own
  dominant hue (adapts to base colour rather than hardcoding one).
"""

import cv2
import numpy as np


def extract_petal_marking_features(roi_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(roi_mask > 0)
    if len(ys) < 10:
        return np.zeros(4)

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32)

    flower_hue, flower_sat = hue[ys, xs], sat[ys, xs]
    sat_ok = flower_sat > 40
    if sat_ok.sum() < 10:
        return np.zeros(4)

    hist, edges = np.histogram(flower_hue[sat_ok], bins=36, range=(0, 180))
    dominant_hue = float(edges[np.argmax(hist)] + 2.5)

    hue_dev = np.abs(flower_hue - dominant_hue)
    hue_dev = np.minimum(hue_dev, 180 - hue_dev)
    is_marking = (hue_dev > 20) & sat_ok
    marking_fraction = float(is_marking.mean())

    cx, cy = xs.mean(), ys.mean()
    max_r = np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max())
    if is_marking.sum() > 0:
        r_marking = np.sqrt((xs[is_marking]-cx)**2 + (ys[is_marking]-cy)**2) / (max_r + 1e-6)
        marking_mean_rnorm = float(r_marking.mean())
    else:
        marking_mean_rnorm = -1.0   # sentinel: no markings detected at all

    ray_stds = []
    for a in np.linspace(0, 2*np.pi, 24, endpoint=False):
        rr = np.linspace(0.1, 0.95, 15) * max_r
        px = (cx + rr*np.cos(a)).astype(int)
        py = (cy + rr*np.sin(a)).astype(int)
        valid = (px>=0)&(px<roi_mask.shape[1])&(py>=0)&(py<roi_mask.shape[0])
        px, py = px[valid], py[valid]
        on_flower = roi_mask[py, px] > 0
        px, py = px[on_flower], py[on_flower]
        if len(px) >= 4:
            ray_stds.append(float(hue[py, px].std()))
    radial_hue_std = float(np.mean(ray_stds)) if ray_stds else 0.0

    return np.array([marking_fraction, marking_mean_rnorm, radial_hue_std, dominant_hue])
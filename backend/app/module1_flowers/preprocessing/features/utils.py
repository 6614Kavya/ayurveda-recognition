"""
Small helpers shared by 2+ feature-extraction modules.

Keeping these in one place avoids copy-pasted (and possibly drifting)
duplicates across petal_morphometrics.py, petal_overlap.py, filament_core.py,
and vein_center.py.
"""

import numpy as np


def normalize_defects(defects):
    """
    cv2.convexityDefects() is documented to return shape (N, 1, 4), but
    depending on the installed OpenCV build/version it can come back as
    (N, 4) instead — a known cross-version inconsistency, not a bug in
    the contour itself. This squeezes away that inconsistent middle axis
    so callers can always safely index defects[i, 2] / defects[i, 3],
    regardless of which machine or OpenCV version produced it.
    """
    if defects is None:
        return None
    return defects.reshape(-1, 4)


def coefficient_of_variation(x: np.ndarray) -> float:
    """std / mean, guarded against a near-zero mean. Used across several
    petal/vein feature functions as a scale-independent measure of spread."""
    m = x.mean()
    return float(x.std() / m) if m > 1e-6 else 0.0


def circular_smooth(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    """Moving-average smoothing that wraps around at the ends — appropriate
    for signals sampled around a full circle (e.g. radius-vs-angle profiles)."""
    pad = kernel_size // 2
    if pad == 0:
        return signal
    padded = np.concatenate([signal[-pad:], signal, signal[:pad]])
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(padded, kernel, mode='valid')

"""
VedaVision — Colour Features
==============================
Per-pixel intensity statistics from the ORIGINAL masked image (pre-enhancement).
Extracted from BGR, HSV, LAB colour spaces + ExG index + normalised hue histogram.

NOTE: Always pass img_bgr = the RAW letterboxed image (img_resized), NOT img_sharp.
      Enhancement steps change colour distributions and would corrupt these features.
"""

import cv2
import numpy as np


def extract_colour_features(img_bgr: np.ndarray,
                             leaf_mask: np.ndarray) -> dict:
    """
    Parameters
    ----------
    img_bgr   : original (pre-enhancement) letterboxed BGR uint8 image
    leaf_mask : uint8 binary mask (255 = foreground)

    Returns
    -------
    dict — 26 channel stats (mean+std for B,G,R,H,S,V,L,a,b,ExG)
           + 18-bin normalised hue histogram
           = 44 dims total, all scale-independent
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_f   = img_bgr.astype(np.float32)
    exg     = 2 * img_f[:, :, 1] - img_f[:, :, 2] - img_f[:, :, 0]
    px      = leaf_mask > 0
    feats   = {}

    def _stats(arr, prefix):
        vals = arr[px]
        feats[f"{prefix}_mean"] = float(np.mean(vals)) if len(vals) else 0.0
        feats[f"{prefix}_std"]  = float(np.std(vals))  if len(vals) else 0.0

    for i, n in enumerate(["b", "g", "r"]):
        _stats(img_bgr[:, :, i], f"bgr_{n}")
    for i, n in enumerate(["h", "s", "v"]):
        _stats(img_hsv[:, :, i], f"hsv_{n}")
    for i, n in enumerate(["l", "a", "b"]):
        _stats(img_lab[:, :, i], f"lab_{n}")
    _stats(exg, "exg")

    # 18-bin normalised hue histogram (0°–180° OpenCV scale, 10°/bin)
    h_vals = img_hsv[:, :, 0][px].astype(np.float32)
    if len(h_vals) > 0:
        hist, _ = np.histogram(h_vals, bins=18, range=(0, 180))
        hist    = hist / (hist.sum() + 1e-6)
        for bi, bv in enumerate(hist):
            feats[f"hue_hist_{bi:02d}"] = float(bv)

    return feats

"""
VedaVision — Letterbox Resize
==============================
Scales the longest image side to TARGET_LONG and pads the short side with white
to produce a square image, preserving the true aspect ratio.

Why this matters:
  - Aspect ratio is a discriminative shape feature between compound leaf species
    (trifoliate leaves are wider than pinnate leaves).
  - INTER_AREA for downscale: averages source pixels → no aliasing on vein patterns.
  - White padding matches the dataset background → mask seeding is unaffected.
"""

import cv2
import numpy as np
from app.module3_compound_leaves.preprocessing.config import TARGET_LONG


def letterbox_resize(img: np.ndarray,
                     target_long: int = TARGET_LONG,
                     pad_colour: tuple = (255, 255, 255)) -> tuple[np.ndarray, dict]:
    """
    Scale the longest side of `img` to `target_long` and pad to a square.

    Parameters
    ----------
    img        : BGR uint8 image (H × W × 3)
    target_long: target square side length (default: config.TARGET_LONG)
    pad_colour : RGB tuple for padding (default: white)

    Returns
    -------
    padded : square BGR uint8 image (target_long × target_long × 3)
    meta   : dict with scale, pad positions, and original dimensions
    """
    h, w = img.shape[:2]
    scale = target_long / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_h      = target_long - new_h
    pad_w      = target_long - new_w
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    padded = cv2.copyMakeBorder(
        scaled, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_colour
    )

    meta = dict(
        scale=scale,
        pad_top=pad_top, pad_bottom=pad_bottom,
        pad_left=pad_left, pad_right=pad_right,
        orig_h=h, orig_w=w,
        new_h=new_h, new_w=new_w,
    )
    return padded, meta

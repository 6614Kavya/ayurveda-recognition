""

import cv2
import numpy as np
from app.module3_compound_leaves.preprocessing.config import TARGET_LONG


def letterbox_resize(img: np.ndarray,
                     target_long: int = TARGET_LONG,
                     pad_colour: tuple = (255, 255, 255)) -> tuple[np.ndarray, dict]:
  
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

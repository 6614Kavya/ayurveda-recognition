"""
VedaVision — Species-ID Branch: Image Enhancement
===================================================
Three-stage enhancement applied ONLY to the species-identification branch.

WHY NOT applied to the health-assessment branch
------------------------------------------------
• Bilateral filter  — changes local colour distributions → corrupts LAB health signals
• CLAHE             — normalises brightness variation → erases yellowing/browning gradients
• Unsharp mask      — alters edge contrast → distorts lesion boundary measurements

All three operations are deliberately absent from masking.py and health/*.py.
The health branch receives img_masked (raw foreground, no enhancement) directly.
"""

import cv2
import numpy as np
from app.module3_compound_leaves.preprocessing.config import (
    BILATERAL_D, BILATERAL_SIGMA,
    CLAHE_CLIP, CLAHE_TILE,
    UNSHARP_SIGMA, UNSHARP_STRENGTH,
)


def enhance_for_species_id(img_masked: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    """
    Apply three-stage vein-enhancement pipeline to the masked leaf image.

    Parameters
    ----------
    img_masked : BGR uint8 image with background zeroed (output of masking.py)
    mask       : uint8 binary mask (255 = foreground) — reapplied after sharpening

    Returns
    -------
    img_sharp : enhanced BGR uint8 image (background remains zero / black)

    Enhancement stages
    ------------------
    1. Bilateral filter  — smooths within-leaflet noise while preserving vein edges.
       d=9:   OpenCV-recommended diameter for real-time use.
       σ=75:  ~30% of 255; bridges camera noise (~10 units) but keeps
              vein-lamina boundary (~40 units) intact.

    2. CLAHE on L channel only — local contrast boost without hue shift.
       Operating in LAB: L=lightness, a/b=colour channels.
       Enhancing L boosts vein contrast without shifting the green hue,
       which is critical because colour features are extracted from the
       ORIGINAL image (img_masked), not from the enhanced image.
       clipLimit=2.5: mid-range; 1.0=no enhancement, 4.0=amplifies noise.
       tileGridSize=(8,8): 512/8=64px tiles → spatially adaptive, not global.

    3. Unsharp mask — sharpens vein boundaries for skeleton extraction.
       GaussianBlur σ=3 targets structures up to ~9px wide (3σ rule).
       sharp = 1.5×original − 0.5×blurred = original + 0.5×(orig−blur).
       Mask reapplied after sharpening to suppress edge halos in padding area.
    """
    # 1. Bilateral filter
    img_bilateral = cv2.bilateralFilter(
        img_masked, d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA, sigmaSpace=BILATERAL_SIGMA
    )

    # 2. CLAHE on L channel
    lab_e         = cv2.cvtColor(img_bilateral, cv2.COLOR_BGR2LAB)
    l, a_e, b_e   = cv2.split(lab_e)
    clahe         = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l_clahe       = clahe.apply(l)
    img_clahe     = cv2.cvtColor(cv2.merge([l_clahe, a_e, b_e]), cv2.COLOR_LAB2BGR)

    # 3. Unsharp mask
    blur      = cv2.GaussianBlur(img_clahe, (0, 0), sigmaX=UNSHARP_SIGMA)
    img_sharp = cv2.addWeighted(img_clahe, UNSHARP_STRENGTH,
                                blur, -(UNSHARP_STRENGTH - 1.0), 0)
    img_sharp = cv2.bitwise_and(img_sharp, img_sharp, mask=mask)  # re-apply mask

    return img_sharp

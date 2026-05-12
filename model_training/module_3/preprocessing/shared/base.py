"""
preprocessing/shared/base.py
Steps 1–4 — runs for EVERY image, both species ID and health branches.

Responsibility: clean and standardise the raw image.
Output: clean 512×512 BGR image + binary leaf mask.

Steps:
    1. Resize + format
    2. Background removal (saliency-guided GrabCut)
    3. Histogram matching (reserved / skipped)
    4. Bilateral filter
"""

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from preprocessing.config import (
    IMG_SIZE, BLUR_THRESHOLD,
    GRABCUT_ITERATIONS, SALIENCY_THRESH,
    BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
)


# =============================================================================
# QUALITY CHECK (runs before step 1)
# =============================================================================

def check_quality(img: np.ndarray) -> tuple[bool, float]:
    """
    Measures sharpness using Laplacian variance.
    Returns (is_acceptable, score). Reject if score < BLUR_THRESHOLD.
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= BLUR_THRESHOLD, round(score, 2)


# =============================================================================
# STEP 1 — RESIZE + FORMAT
# =============================================================================

def step1_resize(img: np.ndarray) -> np.ndarray:
    """
    Resize to 512×512 using Lanczos4.
    512×512 chosen because classical CV features (GLCM, skeleton)
    need spatial resolution — not just raw pixels for a CNN.
    """
    return cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)


# =============================================================================
# STEP 2 — BACKGROUND REMOVAL (saliency-guided GrabCut)
# =============================================================================

def _compute_saliency(img: np.ndarray) -> np.ndarray:
    """
    Spectral Residual saliency — finds the most visually distinct region.
    Used to locate the leaf even when it is off-centre.
    Returns float32 map normalised to [0, 1].
    """
    saliency_obj = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_map  = saliency_obj.computeSaliency(img)

    if not ok:
        # Fallback: simple frequency-based approximation
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        sal_map = np.abs(gray - blurred)

    sal_map = sal_map.astype(np.float32)
    sal_map = cv2.GaussianBlur(sal_map, (11, 11), 2.5)

    s_min, s_max = sal_map.min(), sal_map.max()
    if s_max > s_min:
        sal_map = (sal_map - s_min) / (s_max - s_min)
    return sal_map


def _saliency_to_rect(sal_map: np.ndarray, img_shape: tuple) -> tuple:
    """
    Converts saliency map → GrabCut initialisation rectangle.
    Threshold → find bounding box of salient region → add padding.
    """
    h_img, w_img = img_shape[:2]
    binary = (sal_map > SALIENCY_THRESH).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: centre 70%
        m = 0.15
        return (int(w_img*m), int(h_img*m), int(w_img*0.7), int(h_img*0.7))

    cnt  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    pad_x = int(w_img * 0.05)
    pad_y = int(h_img * 0.05)
    x = max(0, x - pad_x);        y = max(0, y - pad_y)
    w = min(w_img - x, w + 2*pad_x); h = min(h_img - y, h + 2*pad_y)

    return (x, y, w, h)


def step2_remove_background(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Saliency-guided GrabCut background removal.

    Why saliency first:
        Standard GrabCut uses a fixed centre rectangle — assumes the leaf
        is always centred. Field images often have the leaf off to one side.
        Saliency finds where the leaf actually IS, then GrabCut refines the
        boundary precisely. Handles off-centre, tilted, partially cropped leaves.

    Returns:
        masked_img  : BGR image, background = black (0,0,0)
        binary_mask : uint8, 255=leaf region, 0=background
    """
    sal_map = _compute_saliency(img)
    rect    = _saliency_to_rect(sal_map, img.shape)

    x, y, w, h = rect
    if w < 10 or h < 10:
        x, y = int(img.shape[1]*0.1), int(img.shape[0]*0.1)
        w, h = int(img.shape[1]*0.8), int(img.shape[0]*0.8)
        rect = (x, y, w, h)

    gc_mask   = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model,
                    GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        binary_mask = np.ones(img.shape[:2], np.uint8) * 255
        return img.copy(), binary_mask

    binary_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Fill holes + keep only largest region (the compound leaf)
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        clean = np.zeros_like(binary_mask)
        cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        binary_mask = clean

    return cv2.bitwise_and(img, img, mask=binary_mask), binary_mask


# =============================================================================
# STEP 3 — HISTOGRAM MATCHING (reserved)
# =============================================================================

def step3_histogram_matching(img: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
    """
    SKIPPED — reserved for Phase 2.
    Will use skimage.exposure.match_histograms when reference image is ready.
    Currently returns image unchanged.
    """
    return img


# =============================================================================
# STEP 4 — BILATERAL FILTER
# =============================================================================

def step4_bilateral_filter(img: np.ndarray) -> np.ndarray:
    """
    Edge-preserving noise reduction. d=9, sigmaColor=75, sigmaSpace=75.

    Why bilateral (not Gaussian):
        Gaussian blur removes noise but also blurs vein edges.
        Bilateral filter smooths flat leaf surface but KEEPS sharp vein boundaries.
        Critical — feature extraction (GLCM, skeleton) depends on sharp edges.
    """
    return cv2.bilateralFilter(img, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
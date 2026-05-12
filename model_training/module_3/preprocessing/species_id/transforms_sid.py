"""
preprocessing/species_id/transforms_sid.py
Steps 5a–6a — species ID branch only.

Responsibility: enhance image for structural feature extraction.
Input:  clean bilateral-filtered BGR image from shared base
Output: contrast-enhanced, edge-sharpened BGR image

Steps:
    5a. CLAHE on L channel (LAB space) — enhance vein contrast
    6a. Laplacian edge sharpening — crisp vein + leaflet boundaries
"""

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from preprocessing.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE, SHARPEN_KERNEL


def step5a_clahe(img: np.ndarray) -> np.ndarray:
    """
    CLAHE on L channel only (LAB space). clipLimit=2.0, tile 8×8.

    Enhances local contrast → makes vein patterns more visible.
    L channel = luminance (where texture/structure lives).
    A and B channels (colour) are NOT touched — colour preserved.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def step6a_sharpen(img: np.ndarray) -> np.ndarray:
    """
    Laplacian sharpening kernel. Makes vein edges and leaflet boundaries crisper.
    Kernel: [[0,-1,0],[-1,5,-1],[0,-1,0]]
    Applied before feature extraction so segmentation sees sharp edges.
    """
    kernel    = np.array(SHARPEN_KERNEL, dtype=np.float32)
    sharpened = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
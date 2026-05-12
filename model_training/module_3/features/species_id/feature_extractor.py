"""
features/species_id/feature_extractor.py
Step 8a — Shape + Texture + Venation features.

Responsibility: compute a numerical feature vector from one preprocessed image.
Input:  preprocessed BGR image + binary mask + leaflet list (from step 7a)
Output: dict of feature_name → float value

Features extracted:
    Shape   : AR, solidity, Hu moments, leaflet count, area ratio
    Texture : GLCM (contrast, dissimilarity, homogeneity, energy, correlation)
    Vein    : skeleton density, branch points, branch density
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import convolve
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from preprocessing.config import GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS


def extract_features(img: np.ndarray,
                     binary_mask: np.ndarray,
                     leaflets: list[dict]) -> dict:
    """
    Extracts all features from one preprocessed compound leaf image.

    Returns flat dict: {'leaf_area': 12345.0, 'hu_moment_0': ..., ...}
    This dict becomes one row in the training feature matrix (X).
    """
    features = {}
    features.update(_shape_features(img, binary_mask, leaflets))
    features.update(_texture_features(img, binary_mask))
    features.update(_vein_features(img, binary_mask))
    return features


# =============================================================================
# SHAPE FEATURES
# =============================================================================

def _shape_features(img, mask, leaflets) -> dict:
    f = {}

    # Whole-leaf shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt       = max(contours, key=cv2.contourArea)
        area      = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter  = cv2.arcLength(cnt, True)

        f['leaf_area']         = float(area)
        f['leaf_aspect_ratio'] = float(w) / max(h, 1)
        f['leaf_solidity']     = float(area) / max(hull_area, 1)
        f['leaf_compactness']  = (4 * np.pi * area) / max(perimeter**2, 1)

        # Hu moments — 7 rotation/scale invariant descriptors
        hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
        for i, v in enumerate(hu):
            f[f'hu_{i}'] = float(-np.sign(v) * np.log10(abs(v) + 1e-10))
    else:
        f.update({k: 0.0 for k in [
            'leaf_area','leaf_aspect_ratio','leaf_solidity','leaf_compactness',
            *[f'hu_{i}' for i in range(7)]
        ]})

    # Leaflet-level structural features
    f['leaflet_count'] = len(leaflets)
    if leaflets:
        areas = [lf['area'] for lf in leaflets]
        f['leaflet_area_mean']  = float(np.mean(areas))
        f['leaflet_area_std']   = float(np.std(areas))
        f['leaflet_area_ratio'] = float(max(areas)) / max(min(areas), 1)

        ars = []
        for lf in leaflets:
            x1,y1,x2,y2 = lf['bbox']
            ars.append(float(x2-x1) / max(y2-y1, 1))
        f['leaflet_ar_mean'] = float(np.mean(ars))
        f['leaflet_ar_std']  = float(np.std(ars))
    else:
        f.update({k: 0.0 for k in [
            'leaflet_area_mean','leaflet_area_std','leaflet_area_ratio',
            'leaflet_ar_mean','leaflet_ar_std'
        ]})

    return f


# =============================================================================
# TEXTURE FEATURES (GLCM)
# =============================================================================

def _texture_features(img, mask) -> dict:
    f    = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    gray_q = (gray // (256 // GLCM_LEVELS)).astype(np.uint8)

    angles_rad = [np.deg2rad(a) for a in GLCM_ANGLES]
    glcm = graycomatrix(
        gray_q,
        distances  = GLCM_DISTANCES,
        angles     = angles_rad,
        levels     = GLCM_LEVELS,
        symmetric  = True,
        normed     = True
    )
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation']:
        vals = graycoprops(glcm, prop).flatten()
        f[f'glcm_{prop}_mean'] = float(np.mean(vals))
        f[f'glcm_{prop}_std']  = float(np.std(vals))

    return f


# =============================================================================
# VEIN / SKELETON FEATURES
# =============================================================================

def _vein_features(img, mask) -> dict:
    f    = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Top-hat: isolates thin structures (veins) vs thick structures (leaf body)
    k       = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat  = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    _, vein = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)

    # Skeletonize → 1-pixel-wide vein map
    skel        = skeletonize(vein.astype(bool))
    leaf_area   = max(cv2.countNonZero(mask), 1)
    vein_pixels = int(np.sum(skel))

    f['vein_density']     = vein_pixels / leaf_area
    f['vein_pixel_count'] = float(vein_pixels)

    # Branch points = skeleton pixels with ≥3 neighbours
    kernel_3x3  = np.array([[1,1,1],[1,0,1],[1,1,1]], np.uint8)
    nbr         = convolve(skel.astype(np.uint8), kernel_3x3, mode='constant')
    branch_pts  = int(np.sum((skel > 0) & (nbr >= 3)))
    f['vein_branch_points']  = float(branch_pts)
    f['vein_branch_density'] = branch_pts / max(vein_pixels, 1)

    return f
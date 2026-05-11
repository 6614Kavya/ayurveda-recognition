"""
features/species_id/leaflet_segmentation.py
Step 7a — Watershed + contour detection.

Responsibility: locate individual leaflets in the compound leaf image.
Input:  preprocessed BGR image + binary mask (from preprocessing/)
Output: list of leaflet dicts (id, bbox, mask, image, area, centroid)

This is in features/ not preprocessing/ because:
    - It computes structural information (leaflet locations)
    - That information feeds directly into feature extraction (step 8a)
    - It does NOT clean or standardise the image
"""

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from preprocessing.config import MIN_LEAFLET_AREA


def segment_leaflets(img: np.ndarray,
                     binary_mask: np.ndarray
                     ) -> tuple[list[dict], np.ndarray]:
    """
    Segment individual leaflets using Distance Transform + Watershed.

    Why segment leaflets for species ID:
        Leaflet COUNT, SIZE RATIO, and ARRANGEMENT are primary species identifiers.
        e.g. neem: 9–17 leaflets | ashwagandha: fewer, larger leaflets
        These structural features require knowing where each leaflet is.

    Args:
        img         : preprocessed uint8 BGR 512×512
        binary_mask : uint8, 255=leaf, 0=background

    Returns:
        leaflets  : list of dicts, each with keys:
                    id, bbox(x1,y1,x2,y2), mask, image, area, centroid, contour
        label_vis : colour-coded uint8 BGR visualisation
    """
    mask = binary_mask.copy()

    # ── Distance transform → leaflet centres ─────────────────────────────────
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)

    # ── Definite background ────────────────────────────────────────────────────
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # ── Markers for watershed ─────────────────────────────────────────────────
    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(img.copy(), markers)

    # ── Extract leaflets ──────────────────────────────────────────────────────
    n_labels  = markers.max()
    leaflets  = []
    label_vis = np.zeros_like(img)

    for label_id in range(2, n_labels + 1):
        lf_mask = np.zeros(img.shape[:2], np.uint8)
        lf_mask[markers == label_id] = 255

        area = cv2.countNonZero(lf_mask)
        if area < MIN_LEAFLET_AREA:
            continue

        contours, _ = cv2.findContours(lf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(cnt)
        M  = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00']) if M['m00'] else x + w // 2
        cy = int(M['m01'] / M['m00']) if M['m00'] else y + h // 2

        pad = 5
        x1 = max(0, x-pad);          y1 = max(0, y-pad)
        x2 = min(img.shape[1], x+w+pad); y2 = min(img.shape[0], y+h+pad)

        colour = tuple(int(c) for c in np.random.randint(80, 255, 3))
        label_vis[markers == label_id] = colour

        leaflets.append({
            'id'      : len(leaflets),
            'bbox'    : (x1, y1, x2, y2),
            'mask'    : lf_mask[y1:y2, x1:x2],
            'image'   : img[y1:y2, x1:x2].copy(),
            'area'    : area,
            'centroid': (cx, cy),
            'contour' : cnt,
        })

    return leaflets, label_vis
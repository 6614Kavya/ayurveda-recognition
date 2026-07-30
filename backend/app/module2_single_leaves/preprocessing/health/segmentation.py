import cv2
import numpy as np

def get_border_connected_background(candidate_bg, bridge_break_px=7):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_break_px, bridge_break_px))
    eroded_bg = cv2.erode(candidate_bg, kernel)

    h, w = eroded_bg.shape
    padded = cv2.copyMakeBorder(eroded_bg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood_seed = np.zeros((h + 4, w + 4), np.uint8)
    filled = padded.copy()
    cv2.floodFill(filled, flood_seed, (0, 0), 128)
    border_bg = ((filled == 128).astype(np.uint8) * 255)[1:-1, 1:-1]

    border_bg = cv2.dilate(border_bg, kernel)
    border_bg = cv2.bitwise_and(border_bg, candidate_bg)
    return border_bg


def remove_small_blobs(mask, min_area=100):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255
    return clean


def keep_largest_contours(mask, keep_ratio=0.05, min_absolute_area=2000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    areas = [cv2.contourArea(c) for c in contours]
    largest_area = max(areas)
    clean = np.zeros_like(mask)
    for c, a in zip(contours, areas):
        if a >= largest_area * keep_ratio or a >= min_absolute_area:
            cv2.drawContours(clean, [c], -1, 255, thickness=cv2.FILLED)
    return clean


def segment_leaf(image, dark_v_thresh=90, min_shadow_area=1500,
                  max_artifact_area=150, bridge_break_px=7,
                  morph_kernel_size=3, min_blob_area=100,
                  otsu_relax_factor=0.6, keep_ratio=0.05,
                  min_absolute_area=2000):
    """
    Same segmentation logic as the species-ID pipeline.

    KEY DIFFERENCE: min_shadow_area defaults to 1500 here (vs 200 for
    species-ID). This is the single most important tuning knob for
    health data -- it's the size threshold above which a dark enclosed
    blob gets treated as background/shadow and removed. Set too low, a
    real lesion gets wiped out. Set too high, a real background gap
    between leaflets survives as leaf. There's no universal correct
    value -- it depends on how large your actual lesions are relative
    to any real leaflet gaps in your species. CHECK THIS on your "high"
    severity images in Cell 7 before running the full batch.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    otsu_thresh, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    relaxed_thresh = otsu_thresh * otsu_relax_factor
    _, candidate_leaf = cv2.threshold(s, relaxed_thresh, 255, cv2.THRESH_BINARY)
    candidate_bg = cv2.bitwise_not(candidate_leaf)

    border_bg = get_border_connected_background(candidate_bg, bridge_break_px)

    enclosed = cv2.bitwise_and(candidate_bg, cv2.bitwise_not(border_bg))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, connectivity=8)
    dark_enclosed = np.zeros_like(enclosed)
    undecided = []
    for label in range(1, num_labels):
        comp = (labels == label)
        if stats[label, cv2.CC_STAT_AREA] < min_shadow_area:
            undecided.append(label)
            continue
        if np.median(v[comp]) < dark_v_thresh:
            dark_enclosed[comp] = 255
        else:
            undecided.append(label)

    kernel_adj = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_dark = cv2.dilate(dark_enclosed, kernel_adj, iterations=1)
    for label in undecided:
        comp = (labels == label)
        if stats[label, cv2.CC_STAT_AREA] < max_artifact_area and np.any(dilated_dark[comp]):
            dark_enclosed[comp] = 255

    background_mask = cv2.bitwise_or(border_bg, dark_enclosed)
    leaf_mask = cv2.bitwise_not(background_mask)

    leaf_mask = remove_small_blobs(leaf_mask, min_area=min_blob_area)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    leaf_mask = keep_largest_contours(leaf_mask, keep_ratio=keep_ratio,
                                       min_absolute_area=min_absolute_area)

    return leaf_mask


def crop_to_leaf_bbox(image, mask, padding_frac=0.10):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image, mask
    x, y, w, h = cv2.boundingRect(coords)
    pad_x, pad_y = int(w * padding_frac), int(h * padding_frac)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1 = min(image.shape[1], x + w + pad_x)
    y1 = min(image.shape[0], y + h + pad_y)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


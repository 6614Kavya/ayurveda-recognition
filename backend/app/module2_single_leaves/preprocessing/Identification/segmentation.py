import cv2
import numpy as np

def get_border_connected_background(candidate_bg, bridge_break_px=7):
    """
    Flood-fills candidate_bg in from the image border to find TRUE
    background, with a bridge-break step first: eroding candidate_bg
    by bridge_break_px before flood-filling breaks any thin (1-2px)
    hairline connection between real background and a pale patch deep
    inside the leaf, so the fill can't "leak" through it. The result is
    dilated back out and intersected with the original candidate mask,
    so true background extent is preserved -- only the leaked tendrils
    into the leaf are cut off.
    """
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


def keep_largest_contours(mask, keep_ratio=0.05, min_absolute_area=1500):
    """
    Keeps the largest leaf contour, plus:
      - anything at least keep_ratio of its size (relative), OR
      - anything at least min_absolute_area pixels (absolute floor)
    A pure relative ratio (the old 0.3) can delete a real, legitimately
    separated piece of leaf (e.g. one cut off by a strong glare/shadow
    region reaching the edge) just for being smaller than 30% of the
    main lobe -- which is what produced the clean "missing bite" you're
    seeing on some leaves. The absolute floor protects real leaf area
    regardless of how big the main lobe happens to be, while still
    dropping genuinely tiny debris/specks.
    """
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


def segment_leaf(image,
                 dark_v_thresh=90,
                 min_shadow_area=200,
                 max_artifact_area=150,
                 bridge_break_px=7,
                 morph_kernel_size=3,
                 min_blob_area=100,
                 otsu_relax_factor=0.8):
    """
    Full segmentation:
    Saturation threshold -> bridge-break fill-holes
    -> component-level dark/bright classification
    -> small-blob cleanup
    -> morphological close/open
    -> keep largest contour(s).

    Returns a binary mask:
    255 = leaf
    0 = background/shadow.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # --------------------------------------------------
    # 1. Otsu saturation threshold
    # --------------------------------------------------

    otsu_thresh, _ = cv2.threshold(
        s,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Relax the threshold for pale / low-saturation leaves.
    # Example:
    # Otsu threshold = 50
    # Relax factor = 0.6
    # New threshold = 30
    relaxed_thresh = otsu_thresh * otsu_relax_factor

    _, candidate_leaf = cv2.threshold(
        s,
        relaxed_thresh,
        255,
        cv2.THRESH_BINARY
    )

    candidate_bg = cv2.bitwise_not(candidate_leaf)

    # --------------------------------------------------
    # 2. Find border-connected background
    # --------------------------------------------------

    border_bg = get_border_connected_background(
        candidate_bg,
        bridge_break_px
    )

    # --------------------------------------------------
    # 3. Analyze enclosed low-saturation regions
    # --------------------------------------------------

    enclosed = cv2.bitwise_and(
        candidate_bg,
        cv2.bitwise_not(border_bg)
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        enclosed,
        connectivity=8
    )

    dark_enclosed = np.zeros_like(enclosed)
    undecided = []

    for label in range(1, num_labels):

        comp = (labels == label)

        area = stats[label, cv2.CC_STAT_AREA]

        # Very small regions are not immediately classified
        if area < min_shadow_area:
            undecided.append(label)
            continue

        # Classify the whole component using median brightness
        if np.median(v[comp]) < dark_v_thresh:

            # Dark enclosed region = background/shadow
            dark_enclosed[comp] = 255

        else:

            # Bright/pale enclosed region = protect as leaf
            undecided.append(label)

    # --------------------------------------------------
    # 4. Adjacency promotion for small shadow artifacts
    # --------------------------------------------------

    kernel_adj = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11)
    )

    dilated_dark = cv2.dilate(
        dark_enclosed,
        kernel_adj,
        iterations=1
    )

    for label in undecided:

        comp = (labels == label)

        area = stats[label, cv2.CC_STAT_AREA]

        if (
            area < max_artifact_area
            and np.any(dilated_dark[comp])
        ):
            dark_enclosed[comp] = 255

    # --------------------------------------------------
    # 5. Combine background regions
    # --------------------------------------------------

    background_mask = cv2.bitwise_or(
        border_bg,
        dark_enclosed
    )

    leaf_mask = cv2.bitwise_not(background_mask)

    # --------------------------------------------------
    # 6. Remove small leaf blobs
    # --------------------------------------------------

    leaf_mask = remove_small_blobs(
        leaf_mask,
        min_area=min_blob_area
    )

    # --------------------------------------------------
    # 7. Morphological cleanup
    # --------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            morph_kernel_size,
            morph_kernel_size
        )
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # --------------------------------------------------
    # 8. Keep main leaf contour(s)
    # --------------------------------------------------

    leaf_mask = keep_largest_contours(
        leaf_mask
    )

    return leaf_mask

def crop_to_leaf_bbox(image, mask, padding_frac=0.10):
    """
    Crops both image and mask to the leaf's bounding box. Removes any
    camera-distance artifact -- a real user can't choose their photo
    distance based on species (they don't know it yet), so leaf size
    determined purely by camera distance cannot be a trustworthy
    training signal.
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image, mask
    x, y, w, h = cv2.boundingRect(coords)
    pad_x, pad_y = int(w * padding_frac), int(h * padding_frac)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1 = min(image.shape[1], x + w + pad_x)
    y1 = min(image.shape[0], y + h + pad_y)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

def remove_background(image, mask):
    result = image.copy()
    result[mask == 0] = [255, 255, 255]
    return result

def resize_with_padding(image, target_size, pad_value=255):
    """Resize preserving aspect ratio, pad the rest with white."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h))

    # center on a padded canvas
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

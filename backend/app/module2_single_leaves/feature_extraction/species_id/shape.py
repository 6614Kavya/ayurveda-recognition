import cv2
import numpy as np
from skimage.feature import hog

def extract_hog(gray_img):
    """
    Histogram of Oriented Gradients (HOG).
    Captures DIRECTION of edges across the image.
    Why for leaves:
    → Long narrow leaf: edges run vertically → high vertical HOG
    → Round leaf: edges curve in all directions → spread HOG
    → Gotukola fan shape: distinct HOG pattern
    → Most powerful for distinguishing different leaf SHAPES
    Returns 5 summary values: mean, std, max, min, sum
    """
    hog_feats, _ = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True)
    return {
        'hog_mean': hog_feats.mean(),
        'hog_std' : hog_feats.std(),
        'hog_max' : hog_feats.max(),
        'hog_min' : hog_feats.min(),
        'hog_sum' : hog_feats.sum()
    }  # 5 values


# ── 4. Hu-Moments — Shape Descriptor Features ─────────────
def extract_hu_moments(gray_img):
    """
    Hu-Moments — 7 rotation/scale invariant shape descriptors.
    Captures OVERALL SHAPE of the leaf as numbers.
    Why for leaves:
    → Long narrow leaf → high Hu values (elongated)
    → Round leaf → low Hu values (circular)
    → Gotukola fan → medium unique Hu pattern
    Log-transformed for numerical stability.
    Returns 7 values.
    """
    moments    = cv2.moments(gray_img)
    hu         = cv2.HuMoments(moments).flatten()
    hu_log     = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return {f'hu_{i+1}': v for i, v in enumerate(hu_log)}  # 7 values


def extract_contour_shape(gray_img):
    """
    Geometric shape features from leaf contour.
    Why for leaves:
    → Aspect ratio: long narrow (high) vs round (near 1.0)
    → Circularity: how circular is the leaf
    → Solidity: how solid/compact vs lobed/notched
    → Most useful for separating morphologically similar species
       that differ mainly in shape ratios ✅
    Returns 7 values:
    area, perimeter, aspect_ratio, extent,
    solidity, circularity, equiv_diameter
    """
    _, thresh = cv2.threshold(
        gray_img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {k: 0.0 for k in [
            'shape_area', 'shape_perimeter', 'shape_aspect_ratio',
            'shape_extent', 'shape_solidity',
            'shape_circularity', 'shape_equiv_diameter']}

    cnt  = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    hull        = cv2.convexHull(cnt)
    hull_area   = cv2.contourArea(hull)

    aspect_ratio   = float(w) / h if h > 0 else 0
    extent         = area / (w * h) if (w * h) > 0 else 0
    solidity       = area / hull_area if hull_area > 0 else 0
    circularity    = (4 * np.pi * area / (peri ** 2)
                      if peri > 0 else 0)
    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

    return {
        'shape_area'          : area,
        'shape_perimeter'     : peri,
        'shape_aspect_ratio'  : aspect_ratio,
        'shape_extent'        : extent,
        'shape_solidity'      : solidity,
        'shape_circularity'   : circularity,
        'shape_equiv_diameter': equiv_diameter
    }  # 7 values

def get_leaf_mask_and_contour(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 245).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None
    cnt = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [cnt], -1, 255, cv2.FILLED)
    return clean, cnt

def extract_notch_features(cnt, leaf_length, min_defect_depth_px=3):
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if len(hull_idx) < 4:
        return {'notch_depth': 0.0, 'notch_angle': 180.0}
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
    except cv2.error:
        return {'notch_depth': 0.0, 'notch_angle': 180.0}
    if defects is None:
        return {'notch_depth': 0.0, 'notch_angle': 180.0}

    defects = defects.reshape(-1, 4)
    deepest = max(defects, key=lambda d: d[3])
    s, e, f, depth = deepest
    depth_px = depth / 256.0
    if depth_px < min_defect_depth_px:
        return {'notch_depth': 0.0, 'notch_angle': 180.0}

    start, end, far = cnt[s][0], cnt[e][0], cnt[f][0]
    v1, v2 = start - far, end - far
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_ang, -1, 1)))

    return {
        'notch_depth': depth_px / max(leaf_length, 1),
        'notch_angle': angle,
    }


def extract_margin_features(cnt, area, peri, defect_depth_thresh_px=4):
    margin_roughness = (peri ** 2) / (4 * np.pi * area) if area > 0 else 0

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    serration_count = 0
    if len(hull_idx) >= 4:
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None:
                defects = defects.reshape(-1, 4)
                depths_px = defects[:, 3] / 256.0
                serration_count = int((depths_px > defect_depth_thresh_px).sum())
        except cv2.error:
            pass

    return {
        'margin_roughness': margin_roughness,
        'serration_count': serration_count,
    }

def extract_principal_axis_features(cnt):
    """
    Length/width measured along the leaf's own principal axis (via PCA
    on the contour points), not the image's bounding box.

    Why: a bounding-box aspect ratio changes if the leaf is even
    slightly rotated in-frame; the principal axis doesn't. More
    accurate version of shape_aspect_ratio for a real, imperfectly
    aligned photo. Useful across most pairs since elongation was a
    strong discriminator in nearly every one measured (e.g. Diya NA
    4.93 vs Na 3.62; Walikaha 2.27 vs Kora kaha 1.84).
    """
    points = cnt.reshape(-1, 2).astype(np.float32)
    mean, eigvecs, eigvals = cv2.PCACompute2(points, mean=None)
    projected = (points - mean) @ eigvecs.T
    length = projected[:, 0].max() - projected[:, 0].min()
    width = projected[:, 1].max() - projected[:, 1].min()
    ratio = length / width if width > 0 else 0
    return {
        'principal_length': length,
        'principal_width': width,
        'principal_aspect_ratio': ratio,
    }


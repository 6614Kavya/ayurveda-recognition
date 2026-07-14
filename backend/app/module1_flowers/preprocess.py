# app/module1_flowers/preprocess.py
#
# Synced from preprocessing_pipeline_v3(2).ipynb — produces 210 features.
# If you retrain in the notebook again and the feature count changes,
# this file must be regenerated/updated to match, or router.py will
# throw: "ValueError: X has N features, but StandardScaler is expecting M"

import cv2
import numpy as np
import io
from PIL import Image
from scipy.stats import skew
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# pyright: reportMissingImports=false
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print('✅ HEIC/HEIF support registered (pillow-heif found).')
except ImportError:
    print('⚠️  pillow-heif NOT installed — HEIC/HEIF uploads will fail. '
          'Run: pip install pillow-heif')


# ════════════════════════════════════════════════════════════════
# CONFIG — must match the values used when the model was trained
# ════════════════════════════════════════════════════════════════
CFG = {
    'roi_size'            : (224, 224),
    'min_flower_coverage' : 0.01,
    'max_flower_coverage' : 0.92,
    'glcm_distances'      : [1, 3],
    'glcm_angles'         : [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    'gabor_frequencies'   : [0.1, 0.3, 0.5],
    'gabor_orientations'  : [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    'lbp_radius'          : 3,
    'lbp_n_points'        : 24,
    'hist_bins'           : 32,
}


# ════════════════════════════════════════════════════════════════
# Image loading
# ════════════════════════════════════════════════════════════════
def load_as_rgb(path: str):
    """Load any image (including HEIC) as an RGB numpy array."""
    try:
        return np.array(Image.open(path).convert('RGB'))
    except Exception as e:
        print(f'  ⚠️  Failed to load {path}: {e}')
        return None

def load_bytes_as_rgb(image_bytes: bytes):
    """Load an image from raw bytes (e.g. an UploadFile's contents) as RGB numpy array."""
    if not image_bytes:
        print('⚠️  load_bytes_as_rgb received empty bytes (0-byte upload).')
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()   # force full decode now, so a truncated/corrupt file fails
                     # HERE with a clear reason, not later during feature extraction
        return np.array(img.convert('RGB'))
    except Exception as e:
        # First bytes of a file are its "magic number" — a reliable fingerprint
        # of the real format, regardless of what extension the client sent.
        # JPEG starts \xff\xd8\xff, PNG starts \x89PNG, HEIC starts with an
        # ftyp box (bytes 4-12 spell out 'ftyp' + a brand like 'heic'/'mif1').
        print(f'⚠️  Failed to load image ({len(image_bytes)} bytes, '
              f'header={image_bytes[:12]!r}): {e}')
        return None

def _normalize_defects(defects):
    """
    cv2.convexityDefects() is documented to return shape (N, 1, 4), but
    depending on the installed OpenCV build/version it can come back as
    (N, 4) instead — a known cross-version inconsistency, not a bug in
    the contour itself. This squeezes away that inconsistent middle axis
    so the rest of the code can always safely index defects[i, 2] /
    defects[i, 3], regardless of which machine or OpenCV version produced it.
    """
    if defects is None:
        return None
    return defects.reshape(-1, 4)


def load_and_resize(path: str):
    """Kept for backward compatibility with existing router.py imports."""
    return load_as_rgb(path)


# ════════════════════════════════════════════════════════════════
# BLOCK 4 — ROI extractor v4.1 (multi-cue ensemble, downscaled)
# ════════════════════════════════════════════════════════════════
def _detect_background_type(bgr: np.ndarray, border_frac: float = 0.12) -> str:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bh, bw = int(h * border_frac), int(w * border_frac)
    border = np.concatenate([
        gray[:bh, :].flatten(), gray[-bh:, :].flatten(),
        gray[:, :bw].flatten(), gray[:, -bw:].flatten()
    ])
    return 'dark' if np.median(border) < 100 else 'bright'


def _mask_hsv_value(bgr: np.ndarray, bg_type: str) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:, :, 2], (7, 7), 0)
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bg_type == 'bright':
        mask = cv2.bitwise_not(mask)
    return mask


def _mask_lab_chroma(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    chroma = np.sqrt(a ** 2 + b ** 2)
    chroma_u8 = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    chroma_u8 = cv2.GaussianBlur(chroma_u8, (7, 7), 0)
    _, mask = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _mask_grabcut(bgr: np.ndarray, coarse_mask: np.ndarray, margin_frac: float = 0.06) -> np.ndarray:
    h, w = coarse_mask.shape
    ys, xs = np.where(coarse_mask > 0)
    if len(ys) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    pad = int(margin_frac * min(h, w))
    x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x2, y2 = min(w, xs.max() + pad), min(h, ys.max() + pad)
    rect = (x1, y1, x2 - x1, y2 - y1)

    gc_mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return coarse_mask.copy()
    return np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)


def extract_roi(rgb: np.ndarray,
                 roi_size: tuple = CFG['roi_size'],
                 padding_frac: float = 0.10,
                 work_max_dim: int = 640) -> dict:
    """
    Multi-cue ensemble ROI extractor. Mask computation (Otsu + GrabCut)
    runs on a downscaled working copy for speed; final crop is taken
    from the FULL-resolution original.
    """
    h, w = rgb.shape[:2]
    bgr_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    scale = work_max_dim / max(h, w)
    if scale < 1.0:
        work_w, work_h = int(round(w * scale)), int(round(h * scale))
        bgr = cv2.resize(bgr_full, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        bgr, scale = bgr_full, 1.0

    bg_type = _detect_background_type(bgr)
    mask_v = _mask_hsv_value(bgr, bg_type)
    mask_chroma = _mask_lab_chroma(bgr)

    cov_v, cov_c = np.mean(mask_v > 0), np.mean(mask_chroma > 0)
    seed = mask_v if abs(cov_v - 0.25) < abs(cov_c - 0.25) else mask_chroma
    mask_gc = _mask_grabcut(bgr, seed)

    votes = ((mask_v > 0).astype(np.uint8) + (mask_chroma > 0).astype(np.uint8)
             + (mask_gc > 0).astype(np.uint8))
    mask = np.where(votes >= 2, 255, 0).astype(np.uint8)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    coverage = float(np.sum(mask_full > 0)) / (h * w)

    if coverage > CFG['max_flower_coverage']:
        mask_full = cv2.bitwise_not(mask_full)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_full, connectivity=8)
        if n_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_full = np.where(labels == largest, 255, 0).astype(np.uint8)
        coverage = float(np.sum(mask_full > 0)) / (h * w)

    ys, xs = np.where(mask_full > 0)
    if len(ys) == 0:
        x1, y1, x2, y2 = 0, 0, w, h
        status = 'warn: no flower detected, using full frame'
    else:
        pad = int(padding_frac * min(h, w))
        x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x2, y2 = min(w, xs.max() + pad), min(h, ys.max() + pad)
        status = 'ok'

    roi_bgr = cv2.resize(bgr_full[y1:y2, x1:x2], roi_size)
    roi_rgb = cv2.resize(rgb[y1:y2, x1:x2], roi_size)
    roi_mask = cv2.resize(mask_full[y1:y2, x1:x2], roi_size, interpolation=cv2.INTER_NEAREST)
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    if coverage < CFG['min_flower_coverage']:
        status = f'warn: very small flower (coverage={coverage:.3f})'
    elif coverage > CFG['max_flower_coverage']:
        status = f'warn: coverage too high even after invert ({coverage:.3f})'

    return {
        'roi_bgr': roi_bgr, 'roi_rgb': roi_rgb, 'roi_gray': roi_gray,
        'roi_mask': roi_mask, 'coverage': coverage,
        'status': f'{status} [bg={bg_type}]',
    }


# ════════════════════════════════════════════════════════════════
# BLOCK 5A — Color Features (105)
# ════════════════════════════════════════════════════════════════
def extract_color_features(roi_rgb: np.ndarray, roi_mask: np.ndarray,
                            bins: int = CFG['hist_bins']) -> np.ndarray:
    flower_pixels_rgb = roi_rgb[roi_mask > 0]
    if len(flower_pixels_rgb) == 0:
        return np.zeros(bins * 3 + 9)

    flower_img = flower_pixels_rgb.reshape(-1, 1, 3).astype(np.uint8)
    hsv_pixels = cv2.cvtColor(flower_img, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    h_hist, _ = np.histogram(hsv_pixels[:, 0], bins=bins, range=(0, 180), density=True)
    s_hist, _ = np.histogram(hsv_pixels[:, 1], bins=bins, range=(0, 256), density=True)
    v_hist, _ = np.histogram(hsv_pixels[:, 2], bins=bins, range=(0, 256), density=True)
    color_hist = np.concatenate([h_hist, s_hist, v_hist])

    lab_pixels = cv2.cvtColor(flower_img, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab_features = []
    for ch in range(3):
        channel = lab_pixels[:, ch]
        lab_features.extend([float(channel.mean()), float(channel.std()), float(skew(channel))])

    return np.concatenate([color_hist, lab_features])


# ════════════════════════════════════════════════════════════════
# BLOCK 5B — GLCM Texture Features (10)
# ════════════════════════════════════════════════════════════════
def extract_glcm_features(roi_gray: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    masked_gray = roi_gray.copy()
    masked_gray[roi_mask == 0] = 0
    quantized = (masked_gray // 4).astype(np.uint8)

    glcm = graycomatrix(quantized, distances=CFG['glcm_distances'], angles=CFG['glcm_angles'],
                         levels=64, symmetric=True, normed=True)

    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']:
        values = graycoprops(glcm, prop)
        features.extend([float(values.mean()), float(values.std())])

    return np.array(features)


# ════════════════════════════════════════════════════════════════
# BLOCK 5C — LBP Texture Features (26)
# ════════════════════════════════════════════════════════════════
def extract_lbp_features(roi_gray: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    masked_gray = roi_gray.copy()
    masked_gray[roi_mask == 0] = 0

    lbp = local_binary_pattern(masked_gray, P=CFG['lbp_n_points'], R=CFG['lbp_radius'], method='uniform')

    n_bins = CFG['lbp_n_points'] + 2
    lbp_flower = lbp[roi_mask > 0]
    hist, _ = np.histogram(lbp_flower, bins=n_bins, range=(0, n_bins), density=True)

    return hist.astype(np.float32)


# ════════════════════════════════════════════════════════════════
# BLOCK 5D — Gabor Texture Features (24)
# ════════════════════════════════════════════════════════════════
def extract_gabor_features(roi_gray: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    features = []
    masked_gray = roi_gray.copy().astype(np.float32)
    masked_gray[roi_mask == 0] = 0

    for freq in CFG['gabor_frequencies']:
        for theta in CFG['gabor_orientations']:
            kernel = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta,
                                         lambd=1.0 / freq, gamma=0.5, psi=0)
            filtered = cv2.filter2D(masked_gray, cv2.CV_32F, kernel)
            flower_response = filtered[roi_mask > 0]
            if len(flower_response) > 0:
                features.extend([float(np.abs(flower_response).mean()), float(flower_response.std())])
            else:
                features.extend([0.0, 0.0])

    return np.array(features)


# ════════════════════════════════════════════════════════════════
# BLOCK 5E — Shape Features (12)
# ════════════════════════════════════════════════════════════════
def extract_shape_features(roi_mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(roi_mask)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.concatenate([hu_log, np.zeros(5)])

    largest_c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_c)
    perimeter = cv2.arcLength(largest_c, closed=True)
    circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6))

    hull = cv2.convexHull(largest_c)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    x, y, bw, bh = cv2.boundingRect(largest_c)
    aspect_ratio = bw / (bh + 1e-6)
    extent = area / (bw * bh + 1e-6)

    if len(largest_c) >= 5:
        (_, _), (ma, MA), _ = cv2.fitEllipse(largest_c)
        eccentricity = float(ma) / (float(MA) + 1e-6)
    else:
        eccentricity = 1.0

    shape_extra = np.array([circularity, solidity, aspect_ratio, extent, eccentricity])
    return np.concatenate([hu_log, shape_extra])


# ════════════════════════════════════════════════════════════════
# BLOCK 5F — Petal Count Proxy (4)
# ════════════════════════════════════════════════════════════════
def extract_petal_proxy(roi_mask: np.ndarray) -> np.ndarray:
    h, w = roi_mask.shape
    cy, cx = h // 2, w // 2

    n_angles = 360
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    max_r = min(cy, cx)
    radii = np.zeros(n_angles)

    for i, angle in enumerate(angles):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for r in range(1, max_r):
            px = int(cx + r * cos_a)
            py = int(cy + r * sin_a)
            if not (0 <= px < w and 0 <= py < h):
                radii[i] = r - 1
                break
            if roi_mask[py, px] == 0:
                radii[i] = r - 1
                break
        else:
            radii[i] = max_r

    kernel_size = 15
    radii_smooth = uniform_filter1d(radii, size=kernel_size, mode='wrap')

    peaks, _ = find_peaks(radii_smooth, height=radii_smooth.mean() * 0.7, distance=n_angles // 16)
    petal_count = len(peaks)

    return np.array([float(petal_count), float(radii.mean()), float(radii.std()), float(radii.max())])


# ════════════════════════════════════════════════════════════════
# BLOCK 5H — Petal Morphometric Features (10)
# ════════════════════════════════════════════════════════════════
def extract_petal_morphometrics(roi_mask: np.ndarray, min_defect_depth: float = 0.03) -> np.ndarray:
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(10)

    c = max(contours, key=cv2.contourArea)
    if len(c) < 10:
        return np.zeros(10)

    M = cv2.moments(c)
    if M['m00'] == 0:
        return np.zeros(10)
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']

    hull_idx = np.sort(cv2.convexHull(c, returnPoints=False).flatten())
    if len(hull_idx) < 3:
        return np.zeros(10)

    try:
        defects = cv2.convexityDefects(c, hull_idx)
    except cv2.error:
        return np.zeros(10)
    if defects is None:
        return np.zeros(10)
    defects = _normalize_defects(defects)      # ← fix: always (N, 4) from here on

    max_r = np.sqrt(((c[:, 0, 0] - cx) ** 2 + (c[:, 0, 1] - cy) ** 2).max())
    depth_thresh = min_defect_depth * max_r * 256

    valley_idxs = sorted(int(defects[i, 2]) for i in range(defects.shape[0])  # ← was defects[i, 0, 2]
                          if defects[i, 3] > depth_thresh)                     # ← was defects[i, 0, 3]
    if len(valley_idxs) < 2:
        return np.zeros(10)

    n_petals = len(valley_idxs)
    valley_pts = np.array([c[idx, 0, :] for idx in valley_idxs], dtype=np.float32)
    lengths, base_widths, tip_angles, tip_theta_list = [], [], [], []

    for i in range(n_petals):
        i1, i2 = valley_idxs[i], valley_idxs[(i + 1) % n_petals]
        arc = c[i1:i2 + 1, 0, :] if i2 > i1 else np.vstack([c[i1:, 0, :], c[:i2 + 1, 0, :]])
        if len(arc) == 0:
            continue

        dists = np.sqrt((arc[:, 0] - cx) ** 2 + (arc[:, 1] - cy) ** 2)
        tip = arc[np.argmax(dists)]
        tip_len = dists.max()

        v1, v2 = valley_pts[i], valley_pts[(i + 1) % n_petals]
        base_w = np.linalg.norm(v1 - v2)

        a, b = v1 - tip, v2 - tip
        cos_ang = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6), -1, 1)
        tip_angle = np.degrees(np.arccos(cos_ang))
        tip_theta = np.arctan2(tip[1] - cy, tip[0] - cx)

        lengths.append(tip_len); base_widths.append(base_w)
        tip_angles.append(tip_angle); tip_theta_list.append(tip_theta)

    if not lengths:
        return np.array([float(n_petals), 0, 0, 0, 0, 0, 0, 0, 0, 0])

    lengths, base_widths, tip_angles = map(np.array, (lengths, base_widths, tip_angles))
    ratio = lengths / (base_widths + 1e-6)

    def cv(x):
        m = x.mean()
        return float(x.std() / m) if m > 1e-6 else 0.0

    sorted_theta = np.sort(np.array(tip_theta_list))
    gaps = np.diff(np.concatenate([sorted_theta, [sorted_theta[0] + 2 * np.pi]]))
    symmetry_score = 1.0 / (1.0 + gaps.std())

    return np.array([
        float(n_petals), float(lengths.mean()), cv(lengths),
        float(base_widths.mean()), cv(base_widths),
        float(ratio.mean()), cv(ratio),
        float(tip_angles.mean()), cv(tip_angles),
        float(symmetry_score),
    ])


# ════════════════════════════════════════════════════════════════
# BLOCK 5J — Filament & Core-Contrast Features (9)
# ════════════════════════════════════════════════════════════════
def extract_filament_core_features(roi_rgb: np.ndarray, roi_gray: np.ndarray,
                                    roi_mask: np.ndarray) -> np.ndarray:
    h, w = roi_mask.shape
    ys, xs = np.where(roi_mask > 0)
    if len(ys) == 0:
        return np.zeros(9)

    cy, cx = ys.mean(), xs.mean()
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = dist.max() if dist.size else 1.0
    r_norm = dist / (max_r + 1e-6)

    inner = r_norm < 0.4
    outer = r_norm >= 0.6

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    def zone_mean(channel, zone_mask):
        vals = channel[ys[zone_mask], xs[zone_mask]]
        return float(vals.mean()) if vals.size else 0.0

    inner_hue, outer_hue = zone_mean(h_ch, inner), zone_mean(h_ch, outer)
    inner_val, outer_val = zone_mean(v_ch, inner), zone_mean(v_ch, outer)
    inner_sat, outer_sat = zone_mean(s_ch, inner), zone_mean(s_ch, outer)

    hue_diff = abs(inner_hue - outer_hue)
    hue_diff = min(hue_diff, 180 - hue_diff)
    val_diff = inner_val - outer_val
    sat_diff = inner_sat - outer_sat

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([hue_diff, val_diff, sat_diff, 0, 0, 0, 0, 0, 0])
    c = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(c, True)
    hull_perimeter = cv2.arcLength(cv2.convexHull(c), True)
    jaggedness = perimeter / (hull_perimeter + 1e-6)

    n_angles = 360
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cxi, cyi = int(round(cx)), int(round(cy))
    max_search = min(h, w) // 2
    radii = np.zeros(n_angles)
    for i, a in enumerate(angles):
        ca, sa = np.cos(a), np.sin(a)
        r_found = 0
        for r in range(1, max_search):
            px, py = int(cxi + r * ca), int(cyi + r * sa)
            if not (0 <= px < w and 0 <= py < h) or roi_mask[py, px] == 0:
                r_found = r - 1
                break
            r_found = r
        radii[i] = r_found

    def count_peaks(kernel_size):
        pad = kernel_size // 2
        padded = np.concatenate([radii[-pad:], radii, radii[:pad]]) if pad > 0 else radii
        kernel = np.ones(kernel_size) / kernel_size
        smooth = np.convolve(padded, kernel, mode='valid')
        peaks, _ = find_peaks(smooth, distance=max(2, n_angles // 90))
        return len(peaks)

    fine_peaks = count_peaks(3)
    coarse_peaks = count_peaks(21)
    peak_ratio = fine_peaks / (coarse_peaks + 1e-6)

    masked_gray = roi_gray.copy()
    masked_gray[roi_mask == 0] = 0
    edges = cv2.Canny(masked_gray, 50, 150)
    edge_density = float(np.sum((edges > 0) & (roi_mask > 0))) / (np.sum(roi_mask > 0) + 1e-6)

    radial_cv = float(radii.std() / (radii.mean() + 1e-6))

    return np.array([hue_diff, val_diff, sat_diff, jaggedness,
                      float(fine_peaks), float(coarse_peaks), peak_ratio,
                      edge_density, radial_cv])


# ════════════════════════════════════════════════════════════════
# BLOCK 5K — Petal Overlap & Appendage Features (5)
# ════════════════════════════════════════════════════════════════
def extract_petal_overlap_features(roi_mask: np.ndarray, min_defect_depth: float = 0.03) -> np.ndarray:
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(5)
    c = max(contours, key=cv2.contourArea)
    if len(c) < 10:
        return np.zeros(5)

    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    M = cv2.moments(c)
    if M['m00'] == 0:
        return np.array([0, 0, 0, 0, solidity])
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    max_r = np.sqrt(((c[:, 0, 0] - cx) ** 2 + (c[:, 0, 1] - cy) ** 2).max())

    hull_idx = np.sort(cv2.convexHull(c, returnPoints=False).flatten())
    if len(hull_idx) < 3:
        return np.array([0, 0, 0, 0, solidity])
    try:
        defects = cv2.convexityDefects(c, hull_idx)
    except cv2.error:
        return np.array([0, 0, 0, 0, solidity])
    if defects is None:
        return np.array([0, 0, 0, 0, solidity])
    defects = _normalize_defects(defects)      # ← fix

    depth_thresh = min_defect_depth * max_r * 256
    keep = [i for i in range(defects.shape[0]) if defects[i, 3] > depth_thresh]   # ← was [i, 0, 3]
    if not keep:
        return np.array([0, 0, 0, 0, solidity])

    depths_norm = np.array([defects[i, 3] / 256.0 for i in keep]) / (max_r + 1e-6)  # ← was [i, 0, 3]
    valley_idxs = sorted(int(defects[i, 2]) for i in keep)                          # ← was [i, 0, 2]
    n = len(valley_idxs)

    lengths = []
    if n >= 3:
        for i in range(n):
            i1, i2 = valley_idxs[i], valley_idxs[(i + 1) % n]
            arc = c[i1:i2 + 1, 0, :] if i2 > i1 else np.vstack([c[i1:, 0, :], c[:i2 + 1, 0, :]])
            if len(arc) == 0:
                continue
            dists = np.sqrt((arc[:, 0] - cx) ** 2 + (arc[:, 1] - cy) ** 2)
            lengths.append(dists.max())
    lengths = np.array(lengths)

    if len(lengths) >= 2:
        med = np.median(lengths)
        length_outlier_ratio = float(lengths.max() / (med + 1e-6))
    else:
        length_outlier_ratio = 1.0

    def cv(x):
        m = x.mean()
        return float(x.std() / m) if m > 1e-6 else 0.0

    return np.array([
        float(depths_norm.mean()), float(depths_norm.max()), cv(depths_norm),
        length_outlier_ratio, float(solidity),
    ])


# ════════════════════════════════════════════════════════════════
# BLOCK 5L — Vein Ridge & Center Aperture Features (5)
# ════════════════════════════════════════════════════════════════
def _circular_smooth(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    if pad == 0:
        return signal
    padded = np.concatenate([signal[-pad:], signal, signal[:pad]])
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(padded, kernel, mode='valid')


def _sample_ring(gray: np.ndarray, mask: np.ndarray, cx: float, cy: float,
                  radius: float, n_angles: int = 180) -> np.ndarray:
    h, w = gray.shape
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    xs = (cx + radius * np.cos(angles)).astype(int)
    ys = (cy + radius * np.sin(angles)).astype(int)

    vals = np.full(n_angles, np.nan)
    in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    idx = np.where(in_bounds)[0]
    on_flower = mask[ys[idx], xs[idx]] > 0
    idx = idx[on_flower]
    vals[idx] = gray[ys[idx], xs[idx]]

    valid_frac = np.isnan(vals).mean()
    if valid_frac > 0.5:
        return None
    mean_val = np.nanmean(vals)
    vals = np.where(np.isnan(vals), mean_val, vals)
    return vals


def extract_vein_center_features(roi_rgb: np.ndarray, roi_gray: np.ndarray,
                                  roi_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(roi_mask > 0)
    if len(ys) == 0:
        return np.zeros(5)

    cy, cx = ys.mean(), xs.mean()
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = dist.max() if dist.size else 1.0

    osc_counts, ring_cvs = [], []
    for frac in (0.35, 0.5, 0.65):
        ring = _sample_ring(roi_gray, roi_mask, cx, cy, frac * max_r)
        if ring is None:
            continue
        smooth = _circular_smooth(ring, 5)
        peaks, _ = find_peaks(smooth, distance=6, prominence=max(1.0, smooth.std() * 0.3))
        osc_counts.append(len(peaks))
        m = ring.mean()
        ring_cvs.append(ring.std() / m if m > 1e-6 else 0.0)

    vein_oscillation_count = float(np.mean(osc_counts)) if osc_counts else 0.0
    vein_ring_cv = float(np.mean(ring_cvs)) if ring_cvs else 0.0

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    v_ch = hsv[..., 2]
    r_norm = dist / (max_r + 1e-6)

    inner_zone = r_norm < 0.15
    mid_zone = (r_norm >= 0.4) & (r_norm < 0.6)

    center_brightness = float(v_ch[ys[inner_zone], xs[inner_zone]].mean()) if inner_zone.any() else 0.0
    mid_brightness = float(v_ch[ys[mid_zone], xs[mid_zone]].mean()) if mid_zone.any() else 0.0
    center_vs_midzone_contrast = center_brightness - mid_brightness

    masked_gray = roi_gray.copy()
    masked_gray[roi_mask == 0] = 0
    edges = cv2.Canny(masked_gray, 50, 150)
    inner_mask_img = np.zeros_like(roi_mask)
    inner_mask_img[ys[inner_zone], xs[inner_zone]] = 255
    inner_px = np.sum(inner_mask_img > 0)
    center_edge_density = float(np.sum((edges > 0) & (inner_mask_img > 0))) / (inner_px + 1e-6)

    return np.array([
        vein_oscillation_count, vein_ring_cv,
        center_brightness, center_edge_density, center_vs_midzone_contrast,
    ])


# ════════════════════════════════════════════════════════════════
# BLOCK 5G — Master feature extractor (210 features total)
# ════════════════════════════════════════════════════════════════
def extract_all_features(roi: dict) -> np.ndarray:
    """
    Feature vector breakdown:
      Color             : 105
      GLCM              : 10
      LBP               : 26
      Gabor             : 24
      Shape             : 12
      Petal proxy       : 4
      Petal morph       : 10
      Filament/core     : 9
      Petal overlap     : 5
      Vein/center       : 5
      ─────────────────────
      TOTAL             : 210
    """
    color = extract_color_features(roi['roi_rgb'], roi['roi_mask'])
    glcm = extract_glcm_features(roi['roi_gray'], roi['roi_mask'])
    lbp = extract_lbp_features(roi['roi_gray'], roi['roi_mask'])
    gabor = extract_gabor_features(roi['roi_gray'], roi['roi_mask'])
    shape = extract_shape_features(roi['roi_mask'])
    petal = extract_petal_proxy(roi['roi_mask'])
    petal_morph = extract_petal_morphometrics(roi['roi_mask'])
    filament_core = extract_filament_core_features(roi['roi_rgb'], roi['roi_gray'], roi['roi_mask'])
    petal_overlap = extract_petal_overlap_features(roi['roi_mask'])
    vein_center = extract_vein_center_features(roi['roi_rgb'], roi['roi_gray'], roi['roi_mask'])

    feature_vec = np.concatenate([
        color, glcm, lbp, gabor, shape, petal,
        petal_morph, filament_core, petal_overlap, vein_center
    ])

    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vec

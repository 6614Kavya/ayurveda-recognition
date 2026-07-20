import cv2
import numpy as np

from .config import CFG


def _detect_background_type(bgr: np.ndarray, border_frac: float = 0.12) -> str:
    """Sample a thin border strip to guess whether the background is 'dark' or 'bright'."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bh, bw = int(h * border_frac), int(w * border_frac)
    border = np.concatenate([
        gray[:bh, :].flatten(), gray[-bh:, :].flatten(),
        gray[:, :bw].flatten(), gray[:, -bw:].flatten()
    ])
    return 'dark' if np.median(border) < 100 else 'bright'


def _mask_hsv_value(bgr: np.ndarray, bg_type: str) -> np.ndarray:
    """Cue 1: Otsu threshold on the HSV 'V' (brightness) channel."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:, :, 2], (7, 7), 0)
    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bg_type == 'bright':
        mask = cv2.bitwise_not(mask)
    return mask


def _mask_lab_chroma(bgr: np.ndarray) -> np.ndarray:
    """Cue 2: Otsu threshold on LAB chroma magnitude (how colorful each pixel is)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    chroma = np.sqrt(a ** 2 + b ** 2)
    chroma_u8 = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    chroma_u8 = cv2.GaussianBlur(chroma_u8, (7, 7), 0)
    _, mask = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _mask_grabcut(bgr: np.ndarray, coarse_mask: np.ndarray, margin_frac: float = 0.06) -> np.ndarray:
    """Cue 3: GrabCut, seeded with a bounding rect grown from one of the coarse masks."""
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

    # Pick whichever coarse mask has coverage closer to a "reasonable flower"
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
        # Mask likely picked the background instead of the flower; invert and retry.
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


import cv2
import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurt, trim_mean as _trim
from skimage.feature import blob_log

# Hue histogram configuration

_HUE_BINS  = 6          # 6 bins × 30° = full 0–180° OpenCV hue range
_HUE_RANGE = (0.0, 180.0)

_STAT_SUFFIXES = ("median", "iqr", "q25", "q75", "skew", "kurt")

_REDUNDANT_CHANNEL_PREFIXES = ("hsv_v", "bgr_g", "bgr_r")

REDUNDANT_CLF_COLS = [
    f"colour_{ch}_{suf}"
    for ch in _REDUNDANT_CHANNEL_PREFIXES
    for suf in _STAT_SUFFIXES
]

_OIL_DOT_MIN_SIGMA = 1.0
_OIL_DOT_MAX_SIGMA = 3.0
_OIL_DOT_THRESHOLD = 0.02   # blob_log response threshold (lower = more sensitive)

_BOTANICAL_SENTINEL = -1.0

# Internal helpers

def _robust_stats(vals: np.ndarray, prefix: str, out: dict) -> None:
  
    if len(vals) == 0:
        for suf in _STAT_SUFFIXES:
            out[f"{prefix}_{suf}"] = 0.0
        return

    v = vals.astype(np.float64)
    q25, q50, q75 = (float(np.percentile(v, 25)),
                     float(np.median(v)),
                     float(np.percentile(v, 75)))

    out[f"{prefix}_median"] = q50
    out[f"{prefix}_iqr"]    = q75 - q25
    out[f"{prefix}_q25"]    = q25
    out[f"{prefix}_q75"]    = q75
    out[f"{prefix}_skew"]   = float(_skew(v))
    out[f"{prefix}_kurt"]   = float(_kurt(v))


def _extract_botanical_colour_features(img_bgr: np.ndarray, leaf_mask: np.ndarray) -> dict:
    px = leaf_mask > 0
    feats = {}

    if px.sum() < 100:
        feats["botanical_oil_gland_density"] = _BOTANICAL_SENTINEL
        feats["botanical_gloss_highlight_fraction"] = _BOTANICAL_SENTINEL
        feats["botanical_gloss_v_p95_median_ratio"] = _BOTANICAL_SENTINEL
        return feats

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    # ── Oil/gland dot density (Kasthuri_Dehi vs Thunpath_Kurundu) ─────────
    # Zero out background so blob_log doesn't pick up the white paper edge.
    gray_masked = gray.copy()
    gray_masked[~px] = np.median(gray[px])  # neutral fill, same principle as texture.py's GLCM fill
    try:
        blobs = blob_log(gray_masked, min_sigma=_OIL_DOT_MIN_SIGMA,
                          max_sigma=_OIL_DOT_MAX_SIGMA, num_sigma=3,
                          threshold=_OIL_DOT_THRESHOLD)
        # keep only blobs whose centre actually falls inside the leaf mask
        if len(blobs) > 0:
            ys, xs = blobs[:, 0].astype(int), blobs[:, 1].astype(int)
            valid = (ys >= 0) & (ys < px.shape[0]) & (xs >= 0) & (xs < px.shape[1])
            valid &= px[ys.clip(0, px.shape[0] - 1), xs.clip(0, px.shape[1] - 1)]
            n_dots = int(valid.sum())
        else:
            n_dots = 0
        leaf_area = float(px.sum())
        # density per 10,000 leaf px -- keeps the number in a readable range
        feats["botanical_oil_gland_density"] = float(n_dots / leaf_area * 10000)
    except Exception:
        feats["botanical_oil_gland_density"] = _BOTANICAL_SENTINEL

    # ── Surface glossiness (Kalawal matte vs Kattakumanjal glossy) ────────
    v_vals = img_hsv[:, :, 2][px].astype(np.float64)
    s_vals = img_hsv[:, :, 1][px].astype(np.float64)
    v_median = np.median(v_vals)
    v_iqr = np.percentile(v_vals, 75) - np.percentile(v_vals, 25)

    # Specular highlight = unusually bright AND desaturated relative to the
    # leaf's own distribution (glossy leaves show localised highlights;
    # matte leaves stay uniformly saturated green throughout).
    highlight_thresh_v = v_median + 1.5 * (v_iqr + 1e-6)
    is_highlight = (v_vals > highlight_thresh_v) & (s_vals < np.percentile(s_vals, 25))
    feats["botanical_gloss_highlight_fraction"] = float(is_highlight.mean())

    v_p95 = np.percentile(v_vals, 95)
    feats["botanical_gloss_v_p95_median_ratio"] = float(v_p95 / (v_median + 1e-6))

    return feats



# Public API


def extract_colour_features(img_bgr: np.ndarray,
                             leaf_mask: np.ndarray) -> dict:
    
    # ── Colour-space conversions ───────────────────────────────────────────
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_f   = img_bgr.astype(np.float32)
    exg     = 2.0 * img_f[:, :, 1] - img_f[:, :, 2] - img_f[:, :, 0]

    # ── Foreground pixel gate ─────────────────────────────────────────────
    px = leaf_mask > 0
    feats: dict = {}

    if px.sum() == 0:
        return feats

    # ── Per-channel robust stats ──────────────────────────────────────────
    channel_map = [
        (img_bgr[:, :, 0], "bgr_b"),
        (img_bgr[:, :, 1], "bgr_g"),
        (img_bgr[:, :, 2], "bgr_r"),
        (img_hsv[:, :, 0], "hsv_h"),
        (img_hsv[:, :, 1], "hsv_s"),
        (img_hsv[:, :, 2], "hsv_v"),
        (img_lab[:, :, 0], "lab_l"),
        (img_lab[:, :, 1], "lab_a"),
        (img_lab[:, :, 2], "lab_b"),
        (exg,              "exg"),
    ]
    for arr, prefix in channel_map:
        _robust_stats(arr[px], prefix, feats)

    # ── Trimmed mean on Value channel ─────────────────────────────────────
    v_vals = img_hsv[:, :, 2][px].astype(np.float64)
    feats["hsv_v_trimmed_mean"] = float(_trim(v_vals, 0.10))

    # ── Dominant hue (histogram peak) ─────────────────────────────────────
    hue_vals = img_hsv[:, :, 0][px].astype(np.float32)
    hist36, bin_edges = np.histogram(hue_vals, bins=36, range=_HUE_RANGE)
    peak_bin = int(np.argmax(hist36))
    feats["dominant_hue"]      = float(bin_edges[peak_bin])
    feats["hue_peak_fraction"] = float(hist36[peak_bin] / (hist36.sum() + 1e-6))

    # ── 6-bin normalised hue histogram ────────────────────────────────────
    hist6, _ = np.histogram(hue_vals, bins=_HUE_BINS, range=_HUE_RANGE)
    hist6     = hist6 / (hist6.sum() + 1e-6)
    for bi, bv in enumerate(hist6):
        feats[f"hue_hist_{bi:02d}"] = float(bv)

    # ── NEW: botanical oil-dot / glossiness features ──────────────────────
    try:
        feats.update(_extract_botanical_colour_features(img_bgr, leaf_mask))
    except Exception:
        feats["botanical_oil_gland_density"] = _BOTANICAL_SENTINEL
        feats["botanical_gloss_highlight_fraction"] = _BOTANICAL_SENTINEL
        feats["botanical_gloss_v_p95_median_ratio"] = _BOTANICAL_SENTINEL

    return feats


def extract_colour_features_clf(img_bgr: np.ndarray,
                                  leaf_mask: np.ndarray) -> dict:
   
    
    feats = extract_colour_features(img_bgr, leaf_mask)
    return {k: v for k, v in feats.items() if k not in REDUNDANT_CLF_COLS}
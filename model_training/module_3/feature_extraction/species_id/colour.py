"""
VedaVision — Colour Features  (shadow-robust revision)
=======================================================
Per-pixel intensity statistics from the ORIGINAL masked image (pre-enhancement).
Extracted from BGR, HSV, LAB colour spaces + ExG index + normalised hue histogram.

Shadow-robustness changes vs previous version
---------------------------------------------
REPLACED  mean / std        →  median + IQR  (robust to dark outlier pixels)
ADDED     trimmed_mean      →  10 % trim on Value channel (drops darkest shadow fringe)
ADDED     dominant_hue      →  histogram peak (unaffected by shadow minority peak)
ADDED     hue_peak_fraction →  how dominant the peak colour is (purity signal)
KEPT      skewness/kurtosis →  useful: shadow contamination shifts these detectably
KEPT      18-bin hue hist   →  normalised; shadow adds small dark bin, peak unchanged

NOTE: Always pass img_bgr = the RAW letterboxed image (img_resized), NOT img_sharp.
      Enhancement steps change colour distributions and would corrupt these features.
"""

import cv2
import numpy as np
from scipy.stats import skew as _skew, kurtosis as _kurt, trim_mean as _trim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _robust_stats(vals: np.ndarray, prefix: str, out: dict) -> None:
    """
    Compute shadow-robust statistics for a 1-D array of pixel values.

    median  — unaffected by up to ~49 % outlier pixels
    IQR     — spread without influence from dark shadow tail
    Q25/Q75 — bracket the bulk of the distribution
    skew    — shadow shifts this: useful diagnostic / discriminative feature
    kurt    — same reasoning as skew
    """
    if len(vals) == 0:
        for suf in ("median", "iqr", "q25", "q75", "skew", "kurt"):
            out[f"{prefix}_{suf}"] = 0.0
        return

    v = vals.astype(np.float64)
    q25, q50, q75 = float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75))

    out[f"{prefix}_median"] = q50
    out[f"{prefix}_iqr"]    = q75 - q25
    out[f"{prefix}_q25"]    = q25
    out[f"{prefix}_q75"]    = q75
    out[f"{prefix}_skew"]   = float(_skew(v))
    out[f"{prefix}_kurt"]   = float(_kurt(v))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_colour_features(img_bgr: np.ndarray,
                             leaf_mask: np.ndarray) -> dict:
    """
    Parameters
    ----------
    img_bgr   : original (pre-enhancement) letterboxed BGR uint8 image
    leaf_mask : uint8 binary mask (255 = foreground)

    Returns
    -------
    dict — robust channel stats for B, G, R, H, S, V, L, a, b, ExG
           + trimmed mean on V channel
           + dominant hue + hue peak fraction
           + 18-bin normalised hue histogram
           = 72 dims total (6 stats × 10 channels + trimmed_mean_v + dominant_hue
             + hue_peak_fraction + 18 hue bins)

    Shadow robustness
    -----------------
    All per-channel stats use median/IQR instead of mean/std.
    Shadow pixels are a minority: median and IQR ignore them.
    Histogram peak (dominant_hue) finds the most common leaf colour,
    which is unaffected by a small secondary shadow peak.
    Trimmed mean on V drops the darkest 10 % of pixels before averaging.
    """
    # ── Colour-space conversions ───────────────────────────────────────────
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_f   = img_bgr.astype(np.float32)
    exg     = 2.0 * img_f[:, :, 1] - img_f[:, :, 2] - img_f[:, :, 0]

    # ── Foreground pixel gate ─────────────────────────────────────────────
    px = leaf_mask > 0           # boolean mask — ONLY real leaf pixels
    feats: dict = {}

    if px.sum() == 0:
        return feats             # empty image guard

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
    # Drop the darkest 10 % of V pixels before averaging.
    # Shadow pixels concentrate at the dark end; trimming removes them.
    v_vals = img_hsv[:, :, 2][px].astype(np.float64)
    feats["hsv_v_trimmed_mean"] = float(_trim(v_vals, 0.10))

    # ── Dominant hue (histogram peak) ─────────────────────────────────────
    # Shadow pixels form a separate small peak; argmax finds the main leaf peak.
    hue_vals = img_hsv[:, :, 0][px].astype(np.float32)
    hist36, bin_edges = np.histogram(hue_vals, bins=36, range=(0.0, 180.0))
    peak_bin = int(np.argmax(hist36))
    feats["dominant_hue"]       = float(bin_edges[peak_bin])          # degrees
    feats["hue_peak_fraction"]  = float(hist36[peak_bin] / (hist36.sum() + 1e-6))

    # ── 18-bin normalised hue histogram ───────────────────────────────────
    # Each bin = 10 °.  Shadow pixels add a small dark bin — does not move peak.
    hist18, _ = np.histogram(hue_vals, bins=18, range=(0.0, 180.0))
    hist18     = hist18 / (hist18.sum() + 1e-6)
    for bi, bv in enumerate(hist18):
        feats[f"hue_hist_{bi:02d}"] = float(bv)

    return feats

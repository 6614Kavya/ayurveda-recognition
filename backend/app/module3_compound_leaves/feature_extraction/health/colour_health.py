"""
Colour-degradation features.

MUST be computed on the RAW masked_raw image (no enhance.py) -- the health
branch never enhances, since bilateral+CLAHE+unsharp distorts exactly the
colour signals this module measures. This is a hard architectural rule,
not a style preference.

Pixel classification is done per-pixel in LAB/HSV space against four
named damage categories, following plant-pathology "percent leaf area
affected" convention:
  - necrotic   : dead/brown/black tissue (low L, low S)
  - chlorotic  : yellowing -- chlorophyll loss (high b*, a* shifted warm)
  - pale/patch : bleached / water-soaked (high L, low S, not necrotic)
  - healthy_green : everything else inside the mask

Thresholds below are literature-informed starting points, NOT final --
recalibrate against your own annotated pixels (crop a few known-necrotic /
known-chlorotic patches and check the L/a*/b*/S distributions) before
citing exact numbers in the dissertation. Every threshold here should get
the same docstring-justification treatment as masking.py's thresholds.

--- FIX (this session) ---
Added a `specular` exclusion category. Diagnostic runs on known-healthy
glossy leaves (siyabala) showed the previous `pale` gate (L>170 & S<60)
could catch bright specular reflection off the cuticle -- a genuine
bleached/water-soaked patch and a glossy highlight both read as
bright+desaturated, but a specular highlight is much closer to blown-out
white (L>225) than a real pale lesion. This was NOT the dominant source
of false positives in this session's diagnostic (boundary.py was), but is
a legitimate secondary source worth closing off, especially for glossier
species.

--- FIX (this session, round 2) ---
Confirmed healthy leaves with natural dark/light patchy variegation
(normal pigmentation + specimen lighting/shadow, not disease) were still
occasionally read as damaged by the ABSOLUTE thresholds above, because
absolute LAB cutoffs don't account for per-leaf/per-species baseline
colour (e.g. a naturally darker-green leaf sits closer to the necrotic
gate than a naturally pale-green one, with no damage at all). Added a
per-leaf RELATIVE gate: chlorotic/pale now require BOTH the existing
absolute condition AND a meaningful deviation (in b*/L*) from this leaf's
own median "clearly healthy" tissue colour (sampled from pixels well
inside the absolute healthy range). This is an AND, not a replacement --
absolute thresholds alone still catch severe/diffuse damage (where little
"clearly healthy" tissue remains to anchor against, the relative gate
degrades gracefully back toward the absolute-only behaviour); the
relative gate's job is only to suppress borderline/mild flags that fall
within this leaf's own natural variation.
"""
import cv2
import numpy as np

SENTINEL = -1.0

# How far (in LAB b*/L* units) a pixel must sit from THIS leaf's own
# median healthy-anchor colour before the relative gate agrees a pixel
# looks chlorotic/pale, on top of the absolute gate already agreeing.
# First-pass values -- calibrate against a few confirmed natural-
# variegation vs. genuine-chlorosis crops before citing in the
# dissertation, same caveat as the absolute thresholds above.
REL_B_DELTA_CHLOROTIC = 12.0
REL_L_DELTA_PALE = 15.0
MIN_ANCHOR_PIXELS = 200  # below this, too little "clearly healthy" tissue
                          # to trust an anchor -- fall back to absolute-only


def _healthy_anchor(L_fg, A_fg, B_fg, S_fg):
    """Median L*/b* of pixels that are unambiguously healthy green by the
    absolute rule (deep in 'healthy' territory, not just outside the
    damage gates) -- this leaf's own colour baseline to measure deviation
    against. Returns (None, None) if too few such pixels exist (e.g. a
    leaf that's diffusely damaged all over -- absolute gates then do all
    the work, same as before this fix)."""
    anchor_mask = (S_fg > 60) & (A_fg < 118) & (L_fg > 40) & (L_fg < 170)
    if np.count_nonzero(anchor_mask) < MIN_ANCHOR_PIXELS:
        return None, None
    return float(np.median(L_fg[anchor_mask])), float(np.median(B_fg[anchor_mask]))


def _robust_stats(values: np.ndarray, prefix: str) -> dict:
    if values.size == 0:
        return {f"{prefix}_median": SENTINEL, f"{prefix}_iqr": SENTINEL}
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {f"{prefix}_median": float(med), f"{prefix}_iqr": float(q3 - q1)}


def _classify_damage_masks(img_bgr: np.ndarray, mask_final: np.ndarray) -> dict:
    """
    SINGLE SOURCE OF TRUTH for the necrotic/chlorotic/pale/specular pixel
    gates. Returns full H×W boolean masks (foreground-only; background is
    always False) rather than the flattened foreground-only arrays used
    internally below -- so spatial consumers (spots.py's connected-
    component analysis) can use them directly without re-deriving the
    same thresholds a second time (same principle as the project's
    REDUNDANT_CLF_COLS single-source rule: thresholds live in exactly one
    place).

    extract_colour_health_features() below is just this classifier reduced
    to whole-leaf percentages -- do not duplicate the gates themselves
    anywhere else.

    Returns
    -------
    dict with keys: necrotic, chlorotic, pale, specular, healthy -- each
    a full-size np.bool_ array.
    """
    fg = mask_final.astype(bool)
    H, W = fg.shape
    empty = np.zeros((H, W), dtype=bool)
    if not fg.any():
        return {"necrotic": empty, "chlorotic": empty, "pale": empty,
                "specular": empty, "healthy": empty}

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    S = hsv[..., 1]

    L_fg, A_fg, B_fg, S_fg = L[fg], A[fg], B[fg], S[fg]

    # dark + low saturation -> dead/necrotic brown-black tissue
    necrotic_fg = (L_fg < 90) & (S_fg < 90)

    # this leaf's own healthy-tissue colour baseline, for the relative gate
    anchor_L, anchor_B = _healthy_anchor(L_fg, A_fg, B_fg, S_fg)

    # yellowing: high b* (blue-yellow axis toward yellow), a* shifted off
    # deep-green toward neutral/red, and not already counted as necrotic
    chlorotic_abs_fg = (~necrotic_fg) & (B_fg > 150) & (A_fg > 118)
    if anchor_B is not None:
        chlorotic_fg = chlorotic_abs_fg & (B_fg > anchor_B + REL_B_DELTA_CHLOROTIC)
    else:
        chlorotic_fg = chlorotic_abs_fg  # too little healthy tissue to anchor -- absolute only

    # specular highlight (glossy cuticle reflecting light) -- very bright
    # AND very desaturated, distinct from a genuine bleached/water-soaked
    # patch which is bright but not blown-out white. Excluded before pale
    # so glossy species (e.g. siyabala) don't read shine as damage.
    specular_fg = (L_fg > 225) & (S_fg < 40)

    # bright + desaturated, not necrotic/chlorotic/specular -> bleached / water-soaked
    pale_abs_fg = (~necrotic_fg) & (~chlorotic_fg) & (~specular_fg) & (L_fg > 170) & (S_fg < 60)
    if anchor_L is not None:
        pale_fg = pale_abs_fg & (L_fg > anchor_L + REL_L_DELTA_PALE)
    else:
        pale_fg = pale_abs_fg

    healthy_fg = ~(necrotic_fg | chlorotic_fg | pale_fg)

    def to_full(flat_bool):
        full = empty.copy()
        full[fg] = flat_bool
        return full

    return {
        "necrotic": to_full(necrotic_fg),
        "chlorotic": to_full(chlorotic_fg),
        "pale": to_full(pale_fg),
        "specular": to_full(specular_fg),
        "healthy": to_full(healthy_fg),
    }


def extract_colour_health_features(img_bgr: np.ndarray, mask_final: np.ndarray) -> dict:
    """
    Parameters
    ----------
    img_bgr : masked_raw image (background already zeroed via
        cv2.bitwise_and), un-enhanced.
    mask_final : binary leaf mask, foreground = 255.

    Returns
    -------
    dict of colour_* features. Numerically identical to the pre-refactor
    version -- this now just calls _classify_damage_masks() instead of
    inlining the same gates, so spots.py can reuse them.
    """
    fg = mask_final.astype(bool)
    leaf_px = int(np.count_nonzero(fg))
    if leaf_px == 0:
        return {
            "colour_pct_necrotic": SENTINEL,
            "colour_pct_chlorotic": SENTINEL,
            "colour_pct_pale_patch": SENTINEL,
            "colour_pct_healthy_green": SENTINEL,
            "colour_lab_a_median": SENTINEL,
            "colour_lab_a_iqr": SENTINEL,
            "colour_lab_b_median": SENTINEL,
            "colour_lab_b_iqr": SENTINEL,
        }

    masks = _classify_damage_masks(img_bgr, mask_final)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    A_fg = lab[..., 1][fg]
    B_fg = lab[..., 2][fg]

    def pct(m):
        return float(np.count_nonzero(m) / leaf_px * 100.0)

    out = {
        "colour_pct_necrotic": pct(masks["necrotic"]),
        "colour_pct_chlorotic": pct(masks["chlorotic"]),
        "colour_pct_pale_patch": pct(masks["pale"]),
        "colour_pct_healthy_green": pct(masks["healthy"]),
    }
    out.update(_robust_stats(A_fg, "colour_lab_a"))
    out.update(_robust_stats(B_fg, "colour_lab_b"))
    return out
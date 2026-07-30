"""
Boundary / margin irregularity features for compound-leaf health assessment.

Computes structural cues of leaf-margin damage (chewing, tearing, necrotic
edge loss) from the *outer contour* of `mask_final`. These are shape-only
features -- they do not look at colour -- so they complement the
colour-degradation group in colour_health.py.

No per-leaflet segmentation is required (leaflet instance segmentation is
unreliable per project memory), since these operate on the single largest
outer contour of the whole mask.

--- FIX (this session) ---
For compound leaves, mask_final's outer contour is the WHOLE multi-leaflet
silhouette (leaflet_mask OR rachis_mask, per masking.py). Every gap between
adjacent leaflets is a legitimate concavity -- convexHull deficit and
convexityDefects ("notches") were originally computed over the raw contour
with no way to tell a natural leaflet junction apart from an insect bite.
Empirically this pushed boundary_sub to ~40-45/100 for verified-healthy
compound leaves (beli, siyabala), 5x over the healthy_low=8.0 threshold,
from leaf shape alone.

Fix: gate both notch detection and hull-deficit measurement by proximity to
the rachis mask (already computed and available from masking.py's diag
output, proximity-gated at 15px there -- reuse the same buffer here).
Concavities/deficit area near the rachis are treated as expected structural
gaps, not damage; only concavities away from the rachis (i.e. along the
outer leaflet margins, where real chewing/tearing happens) count.

rachis_mask is optional and defaults to None for backward compatibility
(simple, non-compound leaves, or callers that haven't wired it through
yet) -- but should be passed whenever available, which is every call from
the compound-leaf health pipeline.

--- FIX (this session, round 2) ---
notch_count on its own is a RAW count, not normalised by how much margin
the leaf actually has. A 10-leaflet pinnate leaf has ~3-5x the outer
margin length of a 3-leaflet trifoliate leaf, so it has proportionally
more opportunities for small, real (non-damage) leaflet-tip irregularities
to register as "notches" even after rachis-gating -- these tip regions
don't touch the rachis, so the existing gate doesn't catch them. Added
boundary_notch_density = notch_count normalised per 100px of contour
perimeter, so severity_index.py can score notch prevalence rather than
raw count. boundary_notch_count is still returned unchanged (useful
diagnostically / for the dissertation) -- only what feeds LDSI changes.
"""
import cv2
import numpy as np

SENTINEL = -1.0
DEFAULT_RACHIS_PROXIMITY_PX = 15  # matches masking.py's rachis proximity gate


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _dilate_rachis(rachis_mask: np.ndarray, proximity_px: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * proximity_px + 1,) * 2)
    return cv2.dilate(rachis_mask.astype(np.uint8), kernel).astype(bool)


def _convex_hull_deficit(contour, mask_shape, rachis_dilated=None):
    """
    Fraction of the convex hull "missing" from the leaf outline -- margin
    bites/chewing/tearing remove area, pushing the outline inward of its
    hull. For compound leaves, the natural gaps between leaflets are also
    "missing" from the hull.

    --- FIX (this session, round 3) ---
    The previous version discarded an entire CONNECTED deficit blob the
    instant it touched rachis_dilated anywhere. Verified empirically this
    made boundary_margin_deficit_ratio exactly 0.0 for ALL 6948 rows in
    the real dataset (std==0): on a compound leaf, the natural inter-
    leaflet gap and any real margin damage near a leaflet tip are almost
    always ONE connected deficit region (both originate at/near the
    leaflet-rachis junction and fan outward), so "touches rachis anywhere"
    was true for essentially every blob, every time -- silently zeroing a
    feature that was never actually damage-sensitive.

    Fix: gate PER-PIXEL instead of per-blob. Only the pixels within the
    rachis buffer itself are excluded (the true structural-gap origin);
    the rest of the same connected region -- e.g. a bite near a leaflet
    tip that happens to be topologically connected back to a natural gap
    -- is now correctly kept and counted.
    """
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return SENTINEL

    if rachis_dilated is None:
        leaf_area = cv2.contourArea(contour)
        return float(max(0.0, (hull_area - leaf_area) / hull_area))

    hull_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)
    leaf_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(leaf_mask, [contour], -1, 255, -1)

    deficit_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(leaf_mask)).astype(bool)

    kept_mask = deficit_mask & ~rachis_dilated.astype(bool)
    kept_area = int(np.count_nonzero(kept_mask))

    return float(kept_area / hull_area)


def _contour_roughness(contour):
    """
    Ratio of true perimeter to a Douglas-Peucker-smoothed version of the
    same contour. A torn/notched margin has a much longer true perimeter
    than its smoothed envelope; an intact margin sits near 1.0.

    NOTE: still computed on the raw (ungated) perimeter -- for compound
    leaves this will run somewhat higher than for a simple leaf purely from
    leaflet lobing, same as margin_deficit. Left un-gated deliberately: a
    true per-leaflet roughness metric needs leaflet separation, which is
    the unreliable step documented in project memory. Treat this feature
    as weaker/noisier for compound species and lean on the gated
    margin_deficit_ratio and notch_count as the primary boundary signals
    (this is exactly the kind of species-dependent caveat worth flagging
    in the dissertation rather than silently hoping it washes out).
    """
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return SENTINEL
    epsilon = 0.01 * perimeter
    smoothed = cv2.approxPolyDP(contour, epsilon, True)
    smoothed_perimeter = cv2.arcLength(smoothed, True)
    if smoothed_perimeter <= 0:
        return SENTINEL
    return float(perimeter / smoothed_perimeter)


def _notches(contour, rachis_dilated=None, min_depth_px=3.0):
    """
    Convexity-defect based notch detection. Each defect deeper than
    min_depth_px counts as one notch (chew mark / insect bite / tear) --
    UNLESS its deepest point falls within rachis_dilated, in which case
    it's treated as a natural leaflet-junction gap and discarded.
    Returns (notch_count, mean_notch_depth_px).
    """
    if len(contour) < 5:
        return 0, SENTINEL
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return 0, SENTINEL
    hull_indices = np.sort(hull_indices, axis=0)
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None:
            return 0, 0.0
    except cv2.error:
        return 0, SENTINEL
    if defects is None:
        return 0, SENTINEL

    if defects.ndim == 3:
        depths = defects[:, 0, 3] / 256.0
        far_idxs = defects[:, 0, 2]
    else:
        depths = defects[:, 3] / 256.0
        far_idxs = defects[:, 2]

    kept_depths = []
    for depth, far_idx in zip(depths, far_idxs):
        if depth < min_depth_px:
            continue
        if rachis_dilated is not None:
            fx, fy = contour[far_idx][0]
            if 0 <= fy < rachis_dilated.shape[0] and 0 <= fx < rachis_dilated.shape[1]:
                if rachis_dilated[fy, fx]:
                    continue  # near rachis -- natural leaflet junction, not damage
        kept_depths.append(depth)

    if not kept_depths:
        return 0, 0.0
    return len(kept_depths), float(np.mean(kept_depths))


def extract_boundary_features(
    mask_final: np.ndarray,
    rachis_mask: np.ndarray = None,
    rachis_proximity_px: int = DEFAULT_RACHIS_PROXIMITY_PX,
) -> dict:
    """
    Parameters
    ----------
    mask_final : binary mask (uint8/bool), leaf foreground = 255/True.
    rachis_mask : binary rachis mask from masking.py's diag output
        (Stage 6 in masking.py v5.1.1), or None. Strongly recommended
        whenever the leaf is compound -- without it, boundary features
        will be inflated by leaflet-junction geometry (see module
        docstring). Safe to omit for simple/single-lobe leaves.
    rachis_proximity_px : buffer radius around the rachis mask within
        which concavities are treated as structural, not damage.

    Returns
    -------
    dict of boundary_* features. All SENTINEL (-1.0) if the contour
    cannot be extracted (empty/degenerate mask).
    """
    contour = _largest_contour(mask_final)
    if contour is None or cv2.contourArea(contour) <= 0:
        return {
            "boundary_margin_deficit_ratio": SENTINEL,
            "boundary_contour_roughness": SENTINEL,
            "boundary_notch_count": 0,
            "boundary_notch_density": SENTINEL,
            "boundary_notch_depth_mean": SENTINEL,
        }

    rachis_dilated = _dilate_rachis(rachis_mask, rachis_proximity_px) if rachis_mask is not None else None

    margin_deficit_ratio = _convex_hull_deficit(contour, mask_final.shape[:2], rachis_dilated)
    roughness = _contour_roughness(contour)
    notch_count, notch_depth_mean = _notches(contour, rachis_dilated)

    perimeter = cv2.arcLength(contour, True)
    notch_density = float(notch_count / (perimeter / 100.0)) if perimeter > 0 else SENTINEL

    return {
        "boundary_margin_deficit_ratio": margin_deficit_ratio,
        "boundary_contour_roughness": roughness,
        "boundary_notch_count": notch_count,
        "boundary_notch_density": notch_density,
        "boundary_notch_depth_mean": notch_depth_mean,
    }
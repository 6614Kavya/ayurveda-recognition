"""
feature_extraction/health/spots.py

Discrete lesion/spot features -- connected-component analysis on
necrotic/chlorotic/pale damage pixels, complementing (NOT replacing)
colour_health.py's whole-leaf percentage stats.

WHY this exists (see project memory / gap-diagnostic discussion): a leaf
with twenty small pinpoint fungal spots covering 2% of leaf area and a
leaf with one large 2%-area blotch produce an IDENTICAL
colour_pct_necrotic value, even though a person looking at the photo
would immediately tell them apart -- and diagnose_feature_gaps.py showed
several species with |effect size| ~= 0 on colour_pct_* even at "high"
damage, consistent with this dilution. This module recovers the lost
information: how many distinct lesions, how big each one is, and how
densely they're packed. Works on the flat mask_final union -- no
per-leaflet instance segmentation required (same reasoning as
holes.py: instance separation is unreliable on this dataset, so no
feature here depends on it).

Candidate pixels = necrotic | chlorotic | pale (from
colour_health.py's _classify_damage_masks -- SAME thresholds, reused not
duplicated), MINUS two regions already claimed by other feature modules,
so a single physical patch of damage is never double-counted under two
different feature names:
  - the rachis, dilated by rachis_exclusion_px. Rachis colour genuinely
    overlaps the necrotic gate (masking.py's rachis Tier A:
    LAB b>133,S>35,50<L<150 vs necrotic's L<90,S<90) -- without this
    exclusion every compound leaf's stem reads as one giant false "spot".
    Mirrors scar.py's/boundary.py's rachis-proximity gating exactly
    (same default radius).
  - actual hole pixels (holes.py's territory) -- a hole is missing
    tissue, not discoloured-but-present tissue, so it isn't a "spot"
    here even though it will always also satisfy the necrotic gate
    (a hole shows black background through the leaf).

--- FIX (this session) ---
v1 of this module also excluded a broad margin band, reusing scar.py's
BAND_RADIUS_PX=12 (i.e. fg minus erode(fg, 24px-diameter kernel)), on the
theory that margin damage was scar.py's territory. On the real dataset
this produced spot_count=0 on EVERY leaf, healthy and high-damage alike,
across all 12 species -- confirmed via validate_spot_features.py's
output (1457/1457 images, uniformly zero) and reproduced directly:
eroding a simulated 18px-wide leaflet with a 24px kernel empties it
completely, so margin_band == the ENTIRE leaflet, not just its edge.
Compound-leaf leaflets are routinely narrower than 24px at the standard
512px resize (same reason vein.py needs a crop-then-upscale step), so
this wasn't a rare edge case -- it silently zeroed the candidate mask on
effectively every image. Removed the margin-band exclusion entirely:
scar.py samples a band for a DIFFERENT purpose (periderm colour ratio,
not a per-lesion count), so there was never a real double-counting risk
to guard against -- and a margin/edge lesion is a genuine spot (leaf-spot
disease frequently starts at or near the leaf edge), not something that
should be excluded just for being close to a boundary. Only a couple of
anti-aliased boundary pixels are stripped now (edge_artifact_px, default
2px), not a broad sampling band.
--- FIX (this session, round 2) ---
Even with the margin-band fix above, real "high" damage leaves STILL came
back spot_count=0 across the board. Traced with debug_spot_pipeline.py to
a THIRD failure mode: rachis_mask itself was covering 76.6% of leaf area
on a real high-damage image (confirmed via diagnostic), not a thin stem
line. masking.py's rachis Tier A gate (brown/tan: LAB b>133,S>35,
50<L<150) overlaps the necrotic colour signature -- so on genuinely
diseased leaves, large patches of disease discoloration get misread as
"rachis" by that colour gate. Dilating an already-76%-of-the-leaf mask by
rachis_exclusion_px erases essentially the entire candidate region --
same failure shape as the margin-band bug, just surfacing through a
different mask this time, and worse specifically on high-damage leaves
(exactly the leaves this feature needs to work on).

Added a plausibility guard: if rachis_mask's raw (pre-dilation) coverage
exceeds MAX_PLAUSIBLE_RACHIS_FRACTION of the leaf, treat it as
unreliable/contaminated for THIS image and skip rachis exclusion
entirely (same rachis_mask=None fallback already used when it's
unavailable), rather than trust a value that's statistically implausible
for a rachis. Reports which images tripped the guard via
spot_rachis_guard_triggered so contamination rate can be checked
per-species/level rather than silently discarded.
"""
import cv2
import numpy as np

from feature_extraction.health.colour_health import _classify_damage_masks

SENTINEL = -1.0

# Slightly higher floor than holes.py's MIN_HOLE_AREA_PX=6 -- colour-gate
# noise at mask edges (anti-aliased boundary pixels catching a stray
# necrotic/chlorotic reading) is a bit more common than binary
# foreground/background noise. First-pass value; recalibrate against a
# handful of annotated small-spot crops before citing in the
# dissertation, same caveat as every other threshold in this feature
# bank (colour_health.py, masking.py, etc.).
MIN_SPOT_AREA_PX = 8

DEFAULT_RACHIS_EXCLUSION_PX = 15  # matches scar.py / boundary.py / masking.py
# Thin anti-aliasing strip only -- NOT scar.py's margin-sampling band.
# See FIX note above for why a wider value here silently zeros spot
# detection on narrow leaflets. Keep this small; it exists only to avoid
# a few boundary-blur pixels reading as a false micro-spot, not to carve
# out a "not a spot zone" near the edge.
DEFAULT_EDGE_ARTIFACT_PX = 2

# First-pass guess: for these pinnate compound species, the real rachis
# should occupy a small fraction of total leaf area (a thin central line,
# not a broad region). Confirmed empirically that a contaminated
# rachis_mask can reach 76.6% on a real high-damage leaf -- anything
# anywhere near that magnitude is disease tissue miscoloured as rachis,
# not a real stem. Recalibrate against a handful of known-clean vs
# known-contaminated rachis masks (crop and inspect a few healthy vs.
# high-damage leaves directly) before citing in the dissertation, same
# caveat as every other threshold in this feature bank.
MAX_PLAUSIBLE_RACHIS_FRACTION = 0.25


def _rachis_mask_is_plausible(mask_final: np.ndarray, rachis_mask) -> bool:
    """True if rachis_mask's raw (pre-dilation) coverage is small enough
    to trust as a real stem line rather than damage-tissue contamination.
    A missing rachis_mask (None) is treated as trivially 'plausible' --
    nothing to veto, the caller already handles None by skipping rachis
    exclusion."""
    if rachis_mask is None:
        return True
    leaf_area = int(np.count_nonzero(mask_final))
    if leaf_area == 0:
        return True
    frac = int(np.count_nonzero(rachis_mask)) / leaf_area
    return frac <= MAX_PLAUSIBLE_RACHIS_FRACTION


def _build_exclusion_mask(
    mask_final: np.ndarray,
    mask_before_holefill,
    rachis_mask: np.ndarray = None,
    rachis_exclusion_px: int = DEFAULT_RACHIS_EXCLUSION_PX,
    edge_artifact_px: int = DEFAULT_EDGE_ARTIFACT_PX,
) -> np.ndarray:
    """
    Pixels to REMOVE from spot candidates before connected components.
    Returns a boolean mask, True = excluded (rachis-proximity OR
    anti-aliased boundary sliver OR already-a-hole). Deliberately does
    NOT exclude a broad margin band -- see module docstring's FIX note.
    """
    fg = mask_final.astype(np.uint8)
    H, W = fg.shape
    exclude = np.zeros((H, W), dtype=bool)

    if rachis_mask is not None:
        rkernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * rachis_exclusion_px + 1,) * 2
        )
        rachis_dilated = cv2.dilate(rachis_mask.astype(np.uint8), rkernel)
        exclude |= rachis_dilated.astype(bool)

    if edge_artifact_px > 0:
        ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_artifact_px + 1,) * 2)
        eroded = cv2.erode(fg, ekernel)
        edge_band = cv2.subtract(fg, eroded)
        exclude |= edge_band.astype(bool)

    if mask_before_holefill is not None:
        filled_in = cv2.bitwise_and(fg, cv2.bitwise_not(mask_before_holefill.astype(np.uint8)))
        exclude |= filled_in.astype(bool)

    return exclude


def _blob_sizes(candidate_mask_u8: np.ndarray, min_area: int) -> list:
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate_mask_u8, connectivity=8
    )
    return [
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n_labels)  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]


def extract_spot_features(
    img_bgr: np.ndarray,
    mask_final: np.ndarray,
    mask_before_holefill=None,
    rachis_mask: np.ndarray = None,
    rachis_exclusion_px: int = DEFAULT_RACHIS_EXCLUSION_PX,
    edge_artifact_px: int = DEFAULT_EDGE_ARTIFACT_PX,
    min_spot_area_px: int = MIN_SPOT_AREA_PX,
) -> dict:
    """
    Parameters
    ----------
    img_bgr : masked_raw image, un-enhanced (same hard rule as
        colour_health.py -- never call this on the enhanced image).
    mask_final : binary leaf mask, foreground = 255.
    mask_before_holefill : Stage-7 pre-holefill mask from masking.py's
        diag dict, or None (unavailable for augmented rows).
    rachis_mask : binary rachis mask from masking.py's diag output.
        Strongly recommended for compound leaves -- see module docstring
        for why omitting it contaminates spot_count with rachis pixels.

    Returns
    -------
    dict of spot_* features. spot_count=0 / spot_area_ratio=0.0 is a
    genuine "no lesions found" result, NOT a sentinel -- SENTINEL (-1.0)
    is reserved for "couldn't compute" (empty leaf mask).
    """
    fg = mask_final.astype(bool)
    leaf_area = int(np.count_nonzero(fg))
    if leaf_area == 0:
        return {
            "spot_count": SENTINEL,
            "spot_area_ratio": SENTINEL,
            "spot_density_per_1000px": SENTINEL,
            "spot_mean_size": SENTINEL,
            "spot_size_std": SENTINEL,
            "spot_max_size": SENTINEL,
            "necrotic_spot_count": SENTINEL,
            "chlorotic_spot_count": SENTINEL,
            "spot_rachis_guard_triggered": SENTINEL,
        }

    # GUARD (round 2 fix): a rachis_mask covering an implausible fraction
    # of the leaf is disease tissue miscoloured as rachis (confirmed
    # empirically at 76.6% on a real high-damage leaf), not a real stem --
    # trusting it would erase the entire candidate region exactly like the
    # margin-band bug did. Null it out for this image rather than exclude
    # on it; spot_rachis_guard_triggered records when this fires so
    # contamination rate can be checked per-species/level.
    rachis_guard_triggered = not _rachis_mask_is_plausible(mask_final, rachis_mask)
    effective_rachis_mask = None if rachis_guard_triggered else rachis_mask

    damage_masks = _classify_damage_masks(img_bgr, mask_final)
    exclude = _build_exclusion_mask(
        mask_final, mask_before_holefill, effective_rachis_mask,
        rachis_exclusion_px, edge_artifact_px,
    )

    all_damage = (damage_masks["necrotic"] | damage_masks["chlorotic"] | damage_masks["pale"]) & ~exclude
    sizes = _blob_sizes(all_damage.astype(np.uint8), min_spot_area_px)

    spot_count = len(sizes)
    total_spot_area = sum(sizes)

    # Category-specific counts, reusing the SAME damage masks (not
    # re-thresholding) -- necrotic-dominant vs chlorotic-dominant spotting
    # are different disease signatures (fungal/bacterial vs viral/
    # nutrient-deficiency), worth keeping separate rather than only
    # reporting a combined count.
    necrotic_only = (damage_masks["necrotic"] & ~exclude).astype(np.uint8)
    chlorotic_only = (damage_masks["chlorotic"] & ~exclude).astype(np.uint8)
    necrotic_spot_count = len(_blob_sizes(necrotic_only, min_spot_area_px))
    chlorotic_spot_count = len(_blob_sizes(chlorotic_only, min_spot_area_px))

    return {
        "spot_count": spot_count,
        "spot_area_ratio": float(total_spot_area / leaf_area),
        "spot_density_per_1000px": float(spot_count / leaf_area * 1000.0),
        "spot_mean_size": float(np.mean(sizes)) if sizes else 0.0,
        "spot_size_std": float(np.std(sizes)) if len(sizes) > 1 else 0.0,
        "spot_max_size": int(max(sizes)) if sizes else 0,
        "necrotic_spot_count": necrotic_spot_count,
        "chlorotic_spot_count": chlorotic_spot_count,
        "spot_rachis_guard_triggered": bool(rachis_guard_triggered),
    }
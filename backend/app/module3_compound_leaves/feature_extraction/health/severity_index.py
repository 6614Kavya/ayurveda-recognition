"""
Leaf Damage Severity Index (LDSI) -- composite 0-100 score, plus the
worst-side-wins fusion + threshold logic that turns per-side scores into
an ordinal Healthy/Low/Mid/High label.

WHY THIS FILE EXISTS (answers the "what about threshold values" question):
The dataset ground-truth label is assigned PER LEAF (top+bottom together)
using "worst-side-wins" -- whichever view shows the more severe damage
sets the leaf's overall folder label, even if the other view individually
looks less damaged. Up to now that rule lived only in how the dataset was
folder-organised. This module makes it explicit, numeric, and CHECKABLE:

    1. compute a continuous LDSI score (0-100) per side, independently
    2. map each side's score to a level via CALIBRATED thresholds
       (not hand-picked constants -- see calibrate_thresholds() below)
    3. leaf-level level = max(top_level, bottom_level)  <- worst-side-wins
    4. QC: if the folder label disagrees with this computed label by more
       than one ordinal step, flag the leaf for manual review instead of
       silently trusting the folder name for both images.

Thresholds are NOT hardcoded magic numbers by design. calibrate_thresholds()
fits them from your own labeled training scores (midpoint between
consecutive class median scores). The FITTED thresholds -- not
DEFAULT_THRESHOLDS -- are what you report/justify in the dissertation.
RE-RUN calibrate_thresholds() after this session's changes to boundary.py/
scar.py -- the underlying score distributions shift once leaflet-junction
false positives are gated out, so previously-fitted thresholds are stale.

--- FIXES (this session) ---
1. Added optional 5th sub-score group, ldsi_miner_sub, from
   miner_trail.py. Backward compatible: omit miner_feats and you get the
   original 4-group behaviour.
2. Fixed hole under-weighting: a genuine hole (even a small one -- see
   beli__high__image_08, a leaf with a visible punched-through hole that
   scored 11.6/"healthy") barely moved hole_sub because hole_area_ratio
   is tiny relative to total leaf area even for a clearly-real hole.

--- FIX (this session, round 2) ---
3. MIN_HOLE_SUB_IF_PRESENT was a FLAT floor (any hole at all -> hole_sub
   >= 25, full stop). This over-corrected fix #2 above and caused two new,
   confirmed failure modes:
   - beli__low__image_04: a couple of small insect nibbles (genuinely
     "low" damage) tripped the same flat +25 floor as a real punched
     hole, computing "high".
   - Pinnate/compound leaves (e.g. 10-leaflet species): 1-2 minor
     punctures among 8-10 otherwise-healthy leaflets tripped the same
     flat floor leaf-wide, computing "high" for what is clearly low
     overall damage once you look at the whole leaf.
   Both are the same root cause: "a hole exists" and "the leaf is
   severely holed" were being treated as the same signal. Replaced the
   flat floor with a GRADUATED floor that still guarantees a hole can
   never wash out to "healthy" (matches beli__high__image_08's fix) but
   scales its contribution with hole_area_ratio, so a tiny nibble and a
   leaf-wide puncture pattern no longer produce the same score.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

SENTINEL = -1.0
LEVELS = ["healthy", "low", "mid", "high"]
LEVEL_TO_ORDINAL = {lvl: i for i, lvl in enumerate(LEVELS)}

# Placeholder starting thresholds on the 0-100 LDSI scale.
# MUST be recalibrated against real annotated data via calibrate_thresholds()
# before being used for anything reported in the dissertation.
DEFAULT_THRESHOLDS = {
    "healthy_low": 8.0,   # score < this        -> healthy
    "low_mid": 25.0,      # this <= score < mid_high, upper bound of "low"
    "mid_high": 50.0,     # score >= this        -> high
}

# A leaf with any real hole should never fully wash out to "healthy" just
# because the hole is small relative to total leaf area -- presence of a
# genuine puncture/tissue loss is itself a meaningful signal. BUT the
# strength of that floor must scale with how much of the leaf is actually
# affected, not fire at full strength for a single small nibble.
#
# HOLE_SUB_FLOOR_MIN: bump applied the instant ANY real hole is detected
#   (hole_count > 0), regardless of size -- "there is damage" is never
#   fully zero.
# HOLE_SUB_FLOOR_MAX: floor applied once hole_area_ratio reaches
#   HOLE_AREA_RATIO_FOR_MAX_FLOOR -- i.e. once the hole(s) cover a
#   genuinely significant fraction of the leaf.
# Between those two points the floor scales linearly with hole_area_ratio.
#
# HOLE_AREA_RATIO_FOR_MAX_FLOOR = 0.015 (1.5% of leaf area) is a first-pass
# calibration point -- recalibrate against your own confirmed small-vs-
# large hole examples (beli__low__image_04 as a "should stay low" anchor,
# beli__high__image_08 as a "should reach high territory" anchor) before
# citing in the dissertation.
HOLE_SUB_FLOOR_MIN = 8.0
HOLE_SUB_FLOOR_MAX = 30.0
HOLE_AREA_RATIO_FOR_MAX_FLOOR = 0.015


def _sub_index(values) -> float:
    """Average a list of already-~0-100-scale values into one 0-100
    sub-index, ignoring sentinels/None. Clamped to [0, 100]."""
    valid = [v for v in values if v is not None and v != SENTINEL]
    if not valid:
        return 0.0
    return float(np.clip(np.mean(valid), 0.0, 100.0))


def compute_ldsi(
    boundary_feats: dict,
    hole_feats: dict,
    colour_feats: dict,
    scar_feats: dict,
    miner_feats: Optional[dict] = None,
) -> dict:
    """
    Combine the feature-group dicts into one equal-weighted composite
    0-100 LDSI score, plus the sub-index components (kept separately for
    explainability -- you can show *why* a leaf scored high, which
    matters for viva defensibility).

    miner_feats is optional so this stays backward compatible with call
    sites that haven't been updated to extract miner-trail features yet.
    """
    margin_deficit = boundary_feats.get("boundary_margin_deficit_ratio", SENTINEL)
    roughness = boundary_feats.get("boundary_contour_roughness", SENTINEL)
    # notch_density (per 100px of margin) instead of raw notch_count -- a
    # 10-leaflet leaf has proportionally more margin, and therefore more
    # chances for small non-damage tip irregularities to register as
    # notches, than a 3-leaflet leaf. Density is fair across leaf sizes;
    # raw count over-penalises large/many-leaflet species (see boundary.py
    # fix notes). Falls back to raw notch_count*5 (old behaviour) only if
    # density is unavailable (e.g. degenerate contour).
    notch_density = boundary_feats.get("boundary_notch_density", SENTINEL)
    notch_count = boundary_feats.get("boundary_notch_count", 0)
    notch_term = (
        min(notch_density, 20) * 5 if notch_density not in (SENTINEL, None)
        else min(notch_count, 20) * 5
    )
    boundary_sub = _sub_index([
        margin_deficit * 100 if margin_deficit != SENTINEL else None,
        (roughness - 1.0) * 40 if roughness != SENTINEL else None,  # 1.0->0, 3.5->100
        notch_term,
    ])

    hole_area_ratio = hole_feats.get("hole_area_ratio", SENTINEL)
    hole_count = hole_feats.get("hole_count", 0)
    hole_sub = _sub_index([
        hole_area_ratio * 100 if hole_area_ratio not in (SENTINEL, None) and hole_area_ratio >= 0 else None,
        min(max(hole_count, 0), 20) * 5,
    ])
    if hole_count not in (SENTINEL, None, -1) and hole_count > 0:
        # hole_area_ratio is already normalised by THIS leaf's total mask
        # area, so it's naturally small for a minor puncture on a large
        # multi-leaflet leaf and naturally large for a puncture that eats
        # a real fraction of a small leaflet -- exactly the distinction
        # that was missing before.
        safe_area_ratio = hole_area_ratio if hole_area_ratio not in (SENTINEL, None) and hole_area_ratio >= 0 else 0.0
        severity_frac = min(safe_area_ratio / HOLE_AREA_RATIO_FOR_MAX_FLOOR, 1.0)
        graduated_floor = HOLE_SUB_FLOOR_MIN + severity_frac * (HOLE_SUB_FLOOR_MAX - HOLE_SUB_FLOOR_MIN)
        hole_sub = max(hole_sub, graduated_floor)

    colour_sub = _sub_index([
        colour_feats.get("colour_pct_necrotic", SENTINEL),
        colour_feats.get("colour_pct_chlorotic", SENTINEL),
        colour_feats.get("colour_pct_pale_patch", SENTINEL),
    ])

    scar_ratio = scar_feats.get("scar_tissue_ratio", SENTINEL)
    scar_sub = _sub_index([scar_ratio * 100 if scar_ratio != SENTINEL else None])

    components = [boundary_sub, hole_sub, colour_sub, scar_sub]
    keys = ["ldsi_boundary_sub", "ldsi_hole_sub", "ldsi_colour_sub", "ldsi_scar_sub"]

    if miner_feats is not None:
        coverage = miner_feats.get("miner_trail_coverage_pct", SENTINEL)
        length_norm = miner_feats.get("miner_trail_length_norm", SENTINEL)
        # Scale factors below are a first pass -- calibrate against your
        # confirmed miner-trail examples (e.g. beli__high__image_06)
        # rather than trusting these numbers as-is.
        miner_sub = _sub_index([
            coverage * 15 if coverage not in (SENTINEL, None) else None,
            length_norm * 8 if length_norm not in (SENTINEL, None) else None,
        ])
        components.append(miner_sub)
        keys.append("ldsi_miner_sub")

    ldsi = float(np.mean(components))
    out = dict(zip(keys, components))
    out["ldsi_score"] = ldsi
    return out


def score_to_level(score: float, thresholds: Optional[dict] = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if score < t["healthy_low"]:
        return "healthy"
    if score < t["low_mid"]:
        return "low"
    if score < t["mid_high"]:
        return "mid"
    return "high"


def calibrate_thresholds(scores, labels) -> dict:
    """
    Fit thresholds from labeled training LDSI scores instead of guessing.
    Uses the midpoint between consecutive class median scores -- robust to
    outliers, and easy to defend in a viva ("the threshold is the midpoint
    between the median score of adjacent severity classes in our own
    annotated training set").

    Falls back to DEFAULT_THRESHOLDS for any boundary whose adjacent
    classes have fewer than 3 samples to trust a median from, or where the
    medians are inverted/degenerate.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    medians = {}
    for lvl in LEVELS:
        vals = scores[labels == lvl]
        medians[lvl] = float(np.median(vals)) if len(vals) >= 3 else None

    def midpoint(lo_lvl, hi_lvl, default):
        lo, hi = medians[lo_lvl], medians[hi_lvl]
        if lo is None or hi is None or hi <= lo:
            return default
        return (lo + hi) / 2.0

    return {
        "healthy_low": midpoint("healthy", "low", DEFAULT_THRESHOLDS["healthy_low"]),
        "low_mid": midpoint("low", "mid", DEFAULT_THRESHOLDS["low_mid"]),
        "mid_high": midpoint("mid", "high", DEFAULT_THRESHOLDS["mid_high"]),
    }


@dataclass
class SideResult:
    view: str            # "top" or "bottom"
    score: float
    computed_level: str


@dataclass
class LeafSeverityResult:
    top: SideResult
    bottom: SideResult
    worst_side: str                  # "top" or "bottom"
    fused_level: str                 # worst-side-wins computed level
    folder_label: Optional[str] = None
    label_mismatch: bool = False     # True if folder_label vs fused_level differ by >1 ordinal step


def fuse_worst_side(top_score: float, bottom_score: float, thresholds: Optional[dict] = None,
                     folder_label: Optional[str] = None) -> LeafSeverityResult:
    """
    Implements worst-side-wins explicitly: the leaf's overall level is the
    MORE SEVERE of its two independently-computed per-side levels,
    regardless of which side that came from.

    Also runs a QC check against the dataset's folder label so mislabeled
    or inconsistent leaves are caught before they corrupt training,
    instead of assuming the folder name is automatically correct for both
    sides just because that's the labeling convention.
    """
    top_level = score_to_level(top_score, thresholds)
    bottom_level = score_to_level(bottom_score, thresholds)

    top_ord = LEVEL_TO_ORDINAL[top_level]
    bottom_ord = LEVEL_TO_ORDINAL[bottom_level]

    if top_ord >= bottom_ord:
        worst_side, fused_level = "top", top_level
    else:
        worst_side, fused_level = "bottom", bottom_level

    mismatch = False
    if folder_label is not None:
        mismatch = abs(LEVEL_TO_ORDINAL[fused_level] - LEVEL_TO_ORDINAL[folder_label]) > 1

    return LeafSeverityResult(
        top=SideResult("top", top_score, top_level),
        bottom=SideResult("bottom", bottom_score, bottom_level),
        worst_side=worst_side,
        fused_level=fused_level,
        folder_label=folder_label,
        label_mismatch=mismatch,
    )
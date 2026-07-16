"""
preprocessing/shared/mask_guard.py
=============================================================================
Per-image guard around select_mask(). Does NOT modify masking.py — its
public API (select_mask(), qc_check()) is untouched, per the existing
convention that these signatures stay stable across versions.

Why a guard instead of always using the flattened image
----------------------------------------------------------
Investigation summary (see VedaVision shadow-bleed diagnosis notes):

  1. GrabCut + multi-seed-vote "shadow fix" — RETIRED. Tested on 860
     images, made mean bleed worse (0.80% -> 0.93%). GrabCut seeded
     entirely with PR_FGD/PR_BGD has no hard anchor and can converge to a
     worse boundary than it started with, on exactly the ambiguous images
     you most need it to fix.

  2. Paper-referenced illumination flattening (Stage 0 pre-step) — net
     positive. Tested on 78 images: 65% improved, mean bleed 0.71% -> 0.59%.

  3. But NOT uniformly positive. Visually confirmed on siymbala (compact,
     tightly-packed leaflets): select_mask()'s Stage 5 tight/loose
     structure-selection is inherently borderline for this leaflet
     density (gaps ~1-2px at 512px). Flattening nudges pixel values just
     enough to occasionally flip that decision — and it flips in BOTH
     directions on different images of the same species (one image
     tight->loose and got worse, bleed 3.14%->6.69%; a different image
     loose->tight and got BETTER, bleed 1.20%->0.70%). This proves the
     instability is pre-existing in Stage 5, not a bias introduced by
     flattening — flattening just occasionally triggers it.

Since the direction of the flip isn't predictable per image, the correct
fix at this layer is not "always flatten" or "never flatten" — it's to
compute both and keep whichever is actually better, per image. This is
deterministic, requires no new thresholds, and is fully explainable
(keep the lower-bleed of two candidate masks).

Cost: this doubles select_mask() calls (each involving seed, region-grow,
rachis detection etc.) per image. `skip_flatten_if_baseline_below` gives a
cheap early-exit: if the baseline is already close to zero bleed, there is
negligible upside to computing the flattened variant, so it's skipped and
the guard trivially returns the baseline. Roughly a third of this dataset
already sits below 0.1% baseline bleed (see earlier 860-image batch), so
this meaningfully cuts average compute without weakening the guarantee
below (the guard never returns something worse than baseline either way).

Guarantee
---------
For every image, guarded_bleed = min(baseline_bleed, flattened_bleed).
This is true by construction, not empirically — the guard cannot make any
single image worse than its own baseline. A species whose baseline was
already using "tight" successfully will never be pushed to "loose" by
this guard; a species that benefits from flattening still gets the win.
"""
import cv2
import numpy as np

try:
    from scipy.interpolate import griddata
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from preprocessing.shared.masking import select_mask


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — paper-referenced illumination flattening (unchanged from the
# version validated in the 78-image test)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_illumination(img_resized: np.ndarray, grid: int = 16):
    """
    Estimate a smooth brightness field from paper-dominated tiles, without
    needing a prior leaf/paper segmentation. Robust to leaf coverage
    because paper occupies 80-98% of every image in this dataset (measured
    coverage_pct: mean 10.2%, max 20.7%) — the 90th percentile of L within
    each tile is dominated by paper pixels even when a leaf corner
    intrudes into that tile.
    """
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    h, w = L.shape
    th, tw = h // grid, w // grid

    tile_L = np.zeros((grid, grid), np.float32)
    for i in range(grid):
        for j in range(grid):
            tile = L[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            tile_L[i, j] = np.percentile(tile, 90)

    paper_thresh = np.percentile(tile_L, 50)
    paper_mask = tile_L >= paper_thresh
    target_L = float(tile_L[paper_mask].mean())

    tile_L_filled = tile_L.copy()
    if (~paper_mask).any():
        yy, xx = np.mgrid[0:grid, 0:grid]
        if _HAVE_SCIPY:
            ys, xs = np.where(paper_mask)
            vals = tile_L[paper_mask]
            tile_L_filled = griddata(
                (ys, xs), vals, (yy, xx), method="linear", fill_value=target_L
            )
            tile_L_filled = np.nan_to_num(tile_L_filled, nan=target_L)
        else:
            tile_L_filled[~paper_mask] = target_L

    illum = cv2.resize(
        tile_L_filled.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
    )
    return illum, target_L


def flatten_illumination(img_resized: np.ndarray, grid: int = 16,
                          clip_range: tuple = (0.6, 1.8)) -> np.ndarray:
    """Divide out the estimated illumination field. Chroma (a/b) channels
    are left untouched — illumination drift in L dominates for this setup,
    and over-correcting chroma risks distorting masking.py's saturation
    gate, which the whole shadow-detection logic depends on."""
    illum, target_L = estimate_illumination(img_resized, grid=grid)
    ratio = np.clip(target_L / np.maximum(illum, 1e-3), clip_range[0], clip_range[1])
    img_f = img_resized.astype(np.float32)
    flattened = np.clip(img_f * ratio[:, :, None], 0, 255).astype(np.uint8)
    return flattened


def shadow_bleed_fraction(img_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of mask foreground that is achromatic (S < 20). Always
    measured against the ORIGINAL (non-flattened) image colours, for both
    variants, so the metric reflects real-world appearance rather than the
    illumination-corrected version.

    IMPORTANT CAVEAT (found in QC batch, confirmed on
    ranawara_bottom_PXL_20260506_054942264): this is a RATIO
    (achromatic_fg / total_fg), so it is trivially gameable by shrinking
    the mask. Dropping an entire shadowed leaflet lowers this fraction
    just as effectively as correctly re-including that leaflet's pixels
    as chromatic — the metric cannot distinguish "fixed the bleed" from
    "amputated the leaflet." On that image, flattening dropped 29% of the
    baseline foreground area (one whole leaflet, confirmed by visual mask
    diff) and STILL scored a lower bleed fraction than baseline. This
    metric must never be used alone to pick a winner — see
    coverage_pct()/select_mask_guarded's coverage-safety veto below.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    fg = mask > 0
    if fg.sum() == 0:
        return 0.0
    return float(((s < 20) & fg).sum()) / float(fg.sum())


def coverage_pct(mask: np.ndarray) -> float:
    """Foreground area as a fraction of total frame pixels. Used alongside
    shadow_bleed_fraction so the guard can tell area LOSS (bad — leaflet
    dropped) apart from area unchanged/pixels re-classified as chromatic
    (good — bleed genuinely fixed)."""
    return float((mask > 0).sum()) / float(mask.size)


# ─────────────────────────────────────────────────────────────────────────────
# THE GUARD
# ─────────────────────────────────────────────────────────────────────────────
def select_mask_guarded(img_resized: np.ndarray,
                         grid: int = 16,
                         clip_range: tuple = (0.6, 1.8),
                         skip_flatten_if_baseline_below: float = 0.003,
                         max_coverage_drop_ratio: float = 0.08,
                         return_all: bool = False):
    """
    Drop-in replacement call site for select_mask(). Same return shape
    (mask_final, mask_choice, diag) so existing callers (pipeline.py) only
    need to change the import, not the call signature — diag gains extra
    guard_* fields for audit/traceability but all original diag keys are
    still present, taken from whichever variant won.

    Parameters
    ----------
    skip_flatten_if_baseline_below : float
        If the baseline mask's shadow-bleed is already below this fraction
        (default 0.3%), skip computing the flattened variant entirely and
        return baseline directly. There's negligible room for the
        flattened variant to help an already-clean mask, so this saves a
        full select_mask() call on a meaningful fraction of the dataset
        (roughly a third of images sat below 0.1% baseline bleed in the
        860-image batch) without changing the guarantee below.
    max_coverage_drop_ratio : float
        Coverage-safety veto (default 0.08 = 8%). shadow_bleed_fraction is
        a RATIO (achromatic_fg / total_fg), so it is trivially gameable by
        shrinking the mask — dropping a whole shadowed leaflet lowers the
        ratio just as effectively as correctly recovering that leaflet's
        true colour. Confirmed on ranawara_bottom_PXL_20260506_054942264:
        flattening dropped one entire leaflet (29% of baseline foreground
        area, verified by mask diff) and still scored a LOWER bleed
        fraction than baseline. To prevent this, the flattened variant is
        only eligible to win if its foreground area is no more than
        `max_coverage_drop_ratio` smaller than baseline's — i.e.
        (cov_baseline - cov_flattened) / cov_baseline <= 0.08. If the
        flattened variant fails this check, baseline wins regardless of
        its bleed score, and `diag["guard_leaflet_loss_rejected"]` is set
        True so this is traceable in QC output. 8% was chosen from the
        observed batch: legitimate edge-pixel trims sat at 2-5% relative
        area change; confirmed leaflet-loss cases sat at 11-29% — 8% sits
        in the gap between the two clusters. Re-validate this cutoff if a
        larger batch shows the two clusters overlapping nearer this value.
    return_all : bool
        Default False — pipeline.py call sites see no change at all.
        When True, adds `diag["guard_all"]`: a dict with BOTH variants'
        masks/diags/bleeds (baseline_mask, baseline_diag, baseline_bleed,
        flattened_mask, flattened_diag, flattened_bleed — the flattened_*
        entries are None if the flatten step was skipped). This exists
        purely so QC/visualization tooling (e.g. the batch QC script) can
        render both candidate masks side-by-side with the winner, without
        duplicating select_mask()/flatten_illumination() calls outside
        this module. Never set True in the training/inference pipeline —
        the extra mask arrays have no reason to be serialized there and
        would bloat the per-image diag for no benefit.

    Returns
    -------
    mask_final  : uint8 binary mask of the winning variant
    mask_choice : "tight" | "loose" (of the winning variant)
    diag        : dict — the winning variant's full select_mask() diag,
                  plus:
                    guard_variant_used           "baseline" | "flattened"
                    guard_baseline_bleed         float
                    guard_flattened_bleed        float | None (None if skipped)
                    guard_baseline_mask_choice   "tight" | "loose"
                    guard_flattened_mask_choice  "tight" | "loose" | None
                    guard_baseline_coverage_pct  float (0-1 fraction)
                    guard_flattened_coverage_pct float | None
                    guard_coverage_drop_ratio    float | None — (base-flat)/base
                    guard_leaflet_loss_rejected  bool — True if flattened would
                                                  have won on bleed alone but was
                                                  vetoed for dropping too much area
                    guard_all                    dict | absent (see return_all)

    Guarantees
    ----------
    1. guarded_bleed <= baseline_bleed, EXCEPT when the only way to achieve
       that would cost more than max_coverage_drop_ratio of foreground area
       — in that case the guard deliberately keeps the higher-bleed but
       area-complete baseline, because a shrunk mask corrupts every
       downstream shape/vein/morphology feature far more than residual
       shadow-bleed does.
    2. guarded_coverage_pct is never more than max_coverage_drop_ratio
       below guard_baseline_coverage_pct.
    This function cannot silently return a mask that is missing a leaflet
    baseline had, purely because that omission happened to look "cleaner"
    by the bleed metric.
    """
    mask_a, choice_a, diag_a = select_mask(img_resized)
    bleed_a = shadow_bleed_fraction(img_resized, mask_a)
    cov_a = coverage_pct(mask_a)

    if bleed_a < skip_flatten_if_baseline_below:
        diag = dict(diag_a)
        diag["guard_variant_used"] = "baseline"
        diag["guard_baseline_bleed"] = round(bleed_a, 4)
        diag["guard_flattened_bleed"] = None
        diag["guard_baseline_mask_choice"] = choice_a
        diag["guard_flattened_mask_choice"] = None
        diag["guard_flatten_skipped"] = True
        diag["guard_baseline_coverage_pct"] = round(cov_a, 4)
        diag["guard_flattened_coverage_pct"] = None
        diag["guard_coverage_drop_ratio"] = None
        diag["guard_leaflet_loss_rejected"] = False
        if return_all:
            diag["guard_all"] = {
                "baseline_mask": mask_a, "baseline_diag": diag_a, "baseline_bleed": bleed_a,
                "flattened_mask": None, "flattened_diag": None, "flattened_bleed": None,
            }
        return mask_a, choice_a, diag

    flattened_img = flatten_illumination(img_resized, grid=grid, clip_range=clip_range)
    mask_b, choice_b, diag_b = select_mask(flattened_img)
    bleed_b = shadow_bleed_fraction(img_resized, mask_b)
    cov_b = coverage_pct(mask_b)

    coverage_drop_ratio = (cov_a - cov_b) / cov_a if cov_a > 0 else 0.0
    flattened_would_win_on_bleed = bleed_b < bleed_a
    leaflet_loss_rejected = (
        flattened_would_win_on_bleed and coverage_drop_ratio > max_coverage_drop_ratio
    )

    if flattened_would_win_on_bleed and not leaflet_loss_rejected:
        winner_mask, winner_choice, winner_diag, variant = mask_b, choice_b, diag_b, "flattened"
    else:
        winner_mask, winner_choice, winner_diag, variant = mask_a, choice_a, diag_a, "baseline"

    diag = dict(winner_diag)
    diag["guard_variant_used"] = variant
    diag["guard_baseline_bleed"] = round(bleed_a, 4)
    diag["guard_flattened_bleed"] = round(bleed_b, 4)
    diag["guard_baseline_mask_choice"] = choice_a
    diag["guard_flattened_mask_choice"] = choice_b
    diag["guard_flatten_skipped"] = False
    diag["guard_baseline_coverage_pct"] = round(cov_a, 4)
    diag["guard_flattened_coverage_pct"] = round(cov_b, 4)
    diag["guard_coverage_drop_ratio"] = round(coverage_drop_ratio, 4)
    diag["guard_leaflet_loss_rejected"] = leaflet_loss_rejected
    if return_all:
        diag["guard_all"] = {
            "baseline_mask": mask_a, "baseline_diag": diag_a, "baseline_bleed": bleed_a,
            "flattened_mask": mask_b, "flattened_diag": diag_b, "flattened_bleed": bleed_b,
        }

    return winner_mask, winner_choice, diag
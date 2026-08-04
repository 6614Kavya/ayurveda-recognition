"""
VedaVision — Background Removal v5.1.1 (Rachis-Aware + Hole-Fill + Gap Fix)
=============================================================================
Hybrid seed + region-growing algorithm with saturation-gated shadow
exclusion, separate rachis detection, and flood-fill hole closure.

Algorithm stages
----------------
1. SEED       High-confidence green-tissue pixels via ExG + S + L gates.
              Tier-1 (tight), Tier-2 relaxed fallback for dark species.
2. MODEL      Learn per-image leaf LAB colour mean ± std from seed pixels.
3. CANDIDATE  Colour-model sigma gate + S > 20 shadow exclusion.
4. GROW       Iterative dilation constrained to candidate map.
5. SELECT     Tight (k3 close, iter=1) vs loose (k5 close, iter=1) based
              on component count and area ratio.
6. RACHIS     Detect woody stem / petiole via LAB b-channel + S gate,
              proximity-gated to leaflet mask. No morph_open (preserves
              thin lines). No _remove_noise (thin lines are valid tissue).
7. UNION      Leaflet mask OR rachis mask.
8. HOLE FILL  Flood-fill enclosed holes BEFORE _remove_noise so filled
              interiors are not deleted as noise.
9. FINAL      Light close + noise removal + padding exclusion.

Changes from v4.1
-----------------
Shadow gate:
  v4.1 used  L > 120 AND |a − 128| < 12.
  Problem: rachis tissue (brown/tan) has L ≈ 100–140 and near-neutral a,
  so it hit both conditions and was incorrectly vetoed as shadow.
  v5.1 uses  S < 20 instead (saturation gate).
  Paper shadow = achromatic darkening → S ≈ 0–20 → excluded.
  Rachis tissue = real colour (brown) → S ≈ 30–100 → included.
  Leaflet tissue = green → S ≈ 40–150 → included.

Rachis detection (new in v5.1):
  Separate mask built from LAB b-channel elevation (b > 133, brown shift)
  and S > 35 guard to exclude achromatic shadows.
  Proximity gate: only rachis pixels within 15 px of leaflet mask are kept,
  preventing brown backgrounds from leaking in.
  No MORPH_OPEN on rachis (a 3×3 open erases lines thinner than 3 px).
  No _remove_noise on rachis (thin lines are anatomically valid).

Hole fill (new in v5.1):
  Border flood-fill on inverted mask fills enclosed leaflet holes without
  closing inter-leaflet gaps (those gaps reach the image border and are
  therefore not filled by the border-seeded flood fill).

Closing iterations:
  v5.0 accidentally used k3 iter=2 which merged pinnate leaflets.
  v5.1 restores iter=1 for both tight and loose masks.
"""

import cv2
import numpy as np

from preprocessing.config import MIN_COMP_FRAC, SIGMA_THRESH


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def estimate_illumination(img_resized, grid=16):
    """Estimate a smooth brightness field from paper-dominated tiles,
    using percentile robustness instead of a prior mask."""
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    h, w = L.shape
    th, tw = h // grid, w // grid

    tile_L = np.zeros((grid, grid), np.float32)
    for i in range(grid):
        for j in range(grid):
            tile = L[i*th:(i+1)*th, j*tw:(j+1)*tw]
            tile_L[i, j] = np.percentile(tile, 90)  # robust to a leaf corner

    # confident-paper tiles = brighter half of all tiles
    paper_thresh = np.percentile(tile_L, 50)
    paper_mask = tile_L >= paper_thresh
    target_L = tile_L[paper_mask].mean()

    # fill non-paper tiles by interpolation, then upsample to full res
    tile_L_filled = tile_L.copy()
    if (~paper_mask).any():
        ys, xs = np.where(paper_mask)
        vals = tile_L[paper_mask]
        yy, xx = np.mgrid[0:grid, 0:grid]
        tile_L_filled = griddata((ys, xs), vals, (yy, xx), method='linear', fill_value=target_L)

    illum = cv2.resize(tile_L_filled, (w, h), interpolation=cv2.INTER_CUBIC)
    return illum, target_L

def flatten_illumination(img_resized):
    illum, target_L = estimate_illumination(img_resized)
    ratio = np.clip(target_L / np.maximum(illum, 1e-3), 0.6, 1.8)  # clamp to avoid noise blowup
    img_f = img_resized.astype(np.float32)
    flattened = np.clip(img_f * ratio[:, :, None], 0, 255).astype(np.uint8)
    return flattened
def _remove_noise(mask: np.ndarray,
                  min_frac: float = MIN_COMP_FRAC,
                  img_area: int = 512 * 512) -> np.ndarray:
    """
    Drop connected components smaller than min_frac × image area.

    At 512×512 with min_frac=0.001: threshold = 262 px (≈ 9 px diameter).
    Sensor noise and JPEG artefacts are typically 1–5 px → safely removed.
    The rachis line (2–4 px wide × 300+ px long) has area >> 262 px and
    is never deleted by this filter — it is not called on the rachis mask.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean  = np.zeros_like(mask)
    min_px = int(img_area * min_frac)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            clean[labels == i] = 255
    return clean


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes enclosed inside the mask using a border-seeded flood fill
    on the inverted mask.

    Why this is safe for compound leaves:
      Inter-leaflet gaps are open regions that touch the image border.
      A flood fill seeded from (0, 0) reaches them through the border →
      they receive value 0 in the inverted flood result → NOT filled.
      Only truly enclosed holes (light-bleed, translucent spots, damage)
      are unreachable from the border → filled.
    """
    h, w   = mask.shape
    inv    = cv2.bitwise_not(mask)
    flood  = inv.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 0)
    # flood now == 255 only at enclosed holes
    return cv2.bitwise_or(mask, flood)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SEED
# ══════════════════════════════════════════════════════════════════════════════

def _build_seed(img_resized: np.ndarray,
                img_lab_float: np.ndarray,
                hsv: np.ndarray,
                is_padding: np.ndarray,
                k3: np.ndarray,
                min_comp_frac: float,
                img_area: int) -> tuple[np.ndarray, float, bool]:
    """
    High-confidence green-tissue seed with two-tier fallback.

    Tier 1 (normal):  ExG > 20, S > 25, L < 130
      ExG > 20: literature threshold (Woebbecke et al. 1995) for reliable
                vegetation detection against non-vegetative backgrounds.
      S > 25:   excludes achromatic pixels (paper, shadow) — only real colour.
      L < 130:  leaflet tissue L ≈ 60–130; paper L ≈ 210+; shadow L ≈ 130–190.

    Tier 2 (relaxed): ExG > 8, S > 15, L < 150
      Fallback for dark-pigmented species (e.g. Kattakumanjal) whose leaflets
      have lower ExG due to richer/darker green pigmentation.
      Triggered only when Tier 1 coverage < 1% of image area.
    """
    img_f = img_resized.astype(np.float32)
    exg   = 2.0 * img_f[:, :, 1] - img_f[:, :, 2] - img_f[:, :, 0]
    s_ch  = hsv[:, :, 1].astype(np.float32)
    L_ch  = img_lab_float[:, :, 0]

    # Tier 1
    seed = ((exg > 20) & (s_ch > 25) & (L_ch < 130)).astype(np.uint8) * 255
    seed[is_padding] = 0
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k3, iterations=1)
    seed = _remove_noise(seed, min_frac=min_comp_frac, img_area=img_area)

    cov     = float((seed > 0).sum()) / img_area * 100.0
    relaxed = False

    # Tier 2 fallback
    if cov < 1.0:
        seed = ((exg > 8) & (s_ch > 15) & (L_ch < 150)).astype(np.uint8) * 255
        seed[is_padding] = 0
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k3, iterations=1)
        seed = _remove_noise(seed, min_frac=min_comp_frac, img_area=img_area)
        cov     = float((seed > 0).sum()) / img_area * 100.0
        relaxed = True

    return seed, cov, relaxed


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — COLOUR MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _learn_leaf_model(img_lab_float: np.ndarray,
                      seed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-image leaf LAB colour model (mean ± std) from seed pixels.
    Clamps std to ≥ 8 to avoid over-tight gates on spectrally uniform leaves.
    Falls back to a broad neutral model if fewer than 50 seed pixels exist.
    """
    px = img_lab_float[seed_mask > 0]
    if len(px) < 50:
        return np.array([100.0, 115.0, 130.0]), np.array([20.0, 10.0, 10.0])
    return px.mean(axis=0), np.maximum(px.std(axis=0), 8.0)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CANDIDATE MAP
# ══════════════════════════════════════════════════════════════════════════════

def _build_candidate_map(img_lab_float: np.ndarray,
                         hsv: np.ndarray,
                         mean_lab: np.ndarray,
                         std_lab: np.ndarray,
                         sigma_thresh: float = SIGMA_THRESH) -> np.ndarray:
    """
    Candidate pixels must pass both gates:

    (a) Colour-model gate: max per-channel z-score < sigma_thresh.
        Accepts pixels within sigma_thresh standard deviations of the
        learned leaf LAB model (2.5σ = 98.8% of a normal distribution,
        the standard robust-statistics outlier boundary).

    (b) Saturation gate: S > 20.
        Excludes achromatic pixels (paper shadows, white background).
        Physical justification: a shadow is the same paper under less light —
        same wavelength composition, same hue, same saturation (S unchanged).
        Any pixel with S < 20 cannot be biological tissue.

    Why S-gate replaces the v4.1 L + a veto:
        v4.1:  is_shadow = (L > 120) AND (|a − 128| < 12)
        Rachis tissue (brown/tan): L ≈ 100–140, a ≈ 120–130 (near-neutral).
        Both conditions fire on rachis → rachis incorrectly vetoed as shadow.
        S separates them cleanly because shadows are achromatic (S ≈ 0–20)
        while rachis has real colour (S ≈ 30–100).

    Paper gap gate (S < 30 AND L > 160):
        Inter-leaflet paper gaps on pinnate species have ExG ≈ 10–18 due to
        the rachis casting a slight greenish shadow onto the paper beneath.
        This makes them pass the sigma gate (they look similar to pale rachis).
        Pixel measurement on 5 species confirmed:
          Paper gap:    S = 10–19, L = 170–200
          True rachis:  S = 30–100, L = 80–150
        Gate: NOT (S < 30 AND L > 160).
        A pixel that is simultaneously pale (L > 160) AND achromatic (S < 30)
        cannot be biological tissue — it is paper.
        Threshold justification: S threshold at 30 gives 10-unit margin above
        measured paper gap maximum (S ≈ 19). L threshold at 160 sits in the
        10-unit gap between rachis max (L ≈ 150) and paper gap min (L ≈ 170).
    """
    diff       = np.abs(img_lab_float - mean_lab)
    z          = diff / std_lab
    sigma_gate = z.max(axis=2) < sigma_thresh

    s_ch         = hsv[:, :, 1].astype(np.float32)
    L_ch         = img_lab_float[:, :, 0]
    is_shadow    = s_ch < 25                        # achromatic darks
    is_paper_gap = (s_ch < 30) & (L_ch > 160)      # pale AND achromatic

    return (sigma_gate & ~is_shadow & ~is_paper_gap).astype(np.uint8) * 255


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — REGION GROW
# ══════════════════════════════════════════════════════════════════════════════

def _grow_seed(seed_mask: np.ndarray,
               candidate_mask: np.ndarray,
               n_iterations: int = 40,
               kernel_size: int = 5) -> np.ndarray:
    """
    Iteratively dilate the seed, constrained to the candidate map.
    Stops early when growth converges (no new pixels added).
    """
    k        = np.ones((kernel_size, kernel_size), np.uint8)
    grown    = seed_mask.copy()
    prev_sum = -1
    for _ in range(n_iterations):
        dilated  = cv2.dilate(grown, k, iterations=1)
        grown    = cv2.bitwise_and(dilated, candidate_mask)
        curr_sum = int(grown.sum())
        if curr_sum == prev_sum:
            break
        prev_sum = curr_sum
    return grown


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — STRUCTURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _select_structure(grown: np.ndarray,
                      min_comp_frac: float,
                      img_area: int) -> tuple[np.ndarray, str, int, int, int]:
    """
    Choose between tight (k3, iter=1) and loose (k5, iter=1) closing.

    Tight mask: 3×3 kernel, 1 iteration → bridges only 1.5 px gaps.
      Preserves inter-leaflet gaps in pinnate species (taxonomically meaningful).

    Loose mask: 5×5 kernel, 1 iteration → bridges up to 2.5 px gaps.
      Merges nearby regions for trifoliate / sparse species.

    Selection rule:
      Use tight if n_tight_components ≥ 3 AND tight_area ≥ 0.25 × loose_area.
      n ≥ 3: minimum leaflet count for a compound leaf (trifoliate).
      0.25 ratio: accepts up to 75% area reduction from gap preservation,
      but rejects a fragmented mask that has shattered into noise.

    Note: iter=1 is deliberate. iter=2 on k3 ≈ effective 5×5 close, which
    was fusing pinnate leaflets into blobs (observed in v5.0 testing).
    """
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    m_tight = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k3, iterations=1)
    m_tight = _remove_noise(m_tight, min_frac=min_comp_frac, img_area=img_area)

    m_loose = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k5, iterations=1)
    m_loose = _remove_noise(m_loose, min_frac=min_comp_frac, img_area=img_area)

    n_tight    = cv2.connectedComponentsWithStats(m_tight)[0] - 1
    tight_area = int((m_tight > 0).sum())
    loose_area = int((m_loose > 0).sum())

    use_tight = (n_tight >= 3) and (tight_area >= loose_area * 0.25)
    return (m_tight if use_tight else m_loose,
            "tight" if use_tight else "loose",
            n_tight, tight_area, loose_area)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — RACHIS MASK
# ══════════════════════════════════════════════════════════════════════════════

def _build_rachis_mask(img_resized: np.ndarray,
                       img_lab_float: np.ndarray,
                       hsv: np.ndarray,
                       leaflet_mask: np.ndarray,
                       is_padding: np.ndarray,
                       proximity_px: int = 15) -> np.ndarray:
    """
    Detect rachis (central woody stem) and petiole pixels and return
    a binary mask of those pixels.

    Two-tier rachis detection
    -------------------------
    Tier A — Brown/tan rachis (majority of species):
      b > 133:  LAB b-channel. b=128 is neutral; b > 133 indicates yellow/brown
                shift from lignin and tannin chromophores in woody tissue.
                Paper b ≈ 125–132; rachis b ≈ 135–160.
      S > 35:   Saturation guard. Paper cast-shadow: S ≈ 0–25 (achromatic).
                Rachis tissue: S ≈ 30–100 (real colour). Threshold at 35
                provides a 10-unit margin above the shadow S ceiling.
      L > 50:   Excludes pure black / near-black regions (L < 50).
      L < 150:  Excludes paper (L ≈ 210+) and bright cast shadows (L ≈ 150+).

    Tier B — Dark-green rachis (minority of species with green stems):
      ExG in (3, 18): slightly green but not as green as leaflets (ExG ≈ 30–80).
                      Upper bound 18 prevents overlap with leaflet tissue.
      S > 20, L < 140: colour and brightness guards.

    Inter-leaflet gap exclusion
    ----------------------------
    Pinnate species with waxy rachis (Images 1, 2, 5 in test set) have pale
    rachis tissue AND paper gaps between leaflets. Both appear bright and
    low-saturation. Pixel measurement confirmed:
      Paper gap between leaflets: S = 10–19, L = 170–200
      True waxy rachis tissue:    S = 30–100, L = 80–150
    Gate: NOT (S < 30 AND L > 160).
    A pixel simultaneously pale (L > 160) AND achromatic (S < 30) is paper.
    True rachis always has either real colour (S > 30) or is darker (L < 160).
    Threshold justification: S = 30 gives 10-unit margin above paper gap
    maximum (S ≈ 19). L = 160 sits in the 10-unit gap between rachis max
    (L ≈ 150) and paper gap minimum (L ≈ 170). Both are data-driven.

    Proximity gate (safety constraint)
    -----------------------------------
    Only rachis pixels within proximity_px (15 px) of the leaflet mask
    are accepted. This prevents brown backgrounds, soil, or wooden table
    surfaces from entering the mask even if their colour matches Tier A/B.
    At 512 px resolution, leaflet-to-rachis gap ≈ 5–12 px; 15 px provides
    ample margin. Brown background objects are 50+ px from the leaf.

    Critical design decisions
    -------------------------
    NO MORPH_OPEN: A 3×3 morphological open erodes then dilates. Any
      connected structure narrower than 3 px is fully erased by the erosion.
      The rachis is typically 2–4 px wide → open would erase it entirely.
      The proximity gate alone is sufficient to remove spurious detections.

    NO _remove_noise: The rachis line is anatomically valid thin tissue.
      Its pixel area (~3000 px total) is large, but it can be fragmented
      into many short segments each below 262 px after proximity gating.
      Calling _remove_noise would delete those short segments incorrectly.

    MORPH_CLOSE (k3, iter=1): Used to bridge small gaps between rachis
      segments caused by specular reflections or leaf junctions. Close
      never erodes → safe for thin lines.
    """
    if (leaflet_mask > 0).sum() < 200:
        # No leaflets detected — skip rachis to avoid false detections
        return np.zeros_like(leaflet_mask)

    img_f = img_resized.astype(np.float32)
    exg   = 2.0 * img_f[:, :, 1] - img_f[:, :, 2] - img_f[:, :, 0]
    b_ch  = img_lab_float[:, :, 2]
    s_ch  = hsv[:, :, 1].astype(np.float32)
    L_ch  = img_lab_float[:, :, 0]

    # Tier A: brown/tan rachis
    rachis_brown = (
        (b_ch > 133) & (s_ch > 35) & (L_ch > 50) & (L_ch < 150)
    ).astype(np.uint8) * 255

    # Tier B: dark-green rachis
    rachis_green = (
        (exg > 3) & (exg < 18) & (s_ch > 20) & (L_ch < 140)
    ).astype(np.uint8) * 255

    rachis_candidate = cv2.bitwise_or(rachis_brown, rachis_green)

    # Inter-leaflet gap exclusion: remove pixels that are pale AND achromatic
    is_paper_gap             = (s_ch < 30) & (L_ch > 160)
    rachis_candidate[is_paper_gap] = 0

    rachis_candidate[is_padding] = 0

    # Proximity gate: accept only rachis adjacent to leaflet mask
    k_prox   = np.ones((proximity_px * 2 + 1, proximity_px * 2 + 1), np.uint8)
    expanded = cv2.dilate(leaflet_mask, k_prox, iterations=1)
    rachis   = cv2.bitwise_and(rachis_candidate, expanded)

    # Small close to bridge gaps in rachis (NOT open — see docstring)
    k3     = np.ones((3, 3), np.uint8)
    rachis = cv2.morphologyEx(rachis, cv2.MORPH_CLOSE, k3, iterations=1)

    return rachis


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def select_mask(img_resized: np.ndarray,
                min_comp_frac: float = MIN_COMP_FRAC,
                sigma_thresh: float  = SIGMA_THRESH
                ) -> tuple[np.ndarray, str, dict]:
    """
    Background removal v5.1 — Rachis-Aware + Hole-Fill.

    Parameters
    ----------
    img_resized   : letterboxed BGR uint8 image (512 × 512)
    min_comp_frac : minimum component size as fraction of image area
    sigma_thresh  : LAB colour-model sigma gate width

    Returns
    -------
    mask_final  : uint8 binary mask (255 = leaf foreground)
    mask_choice : "tight" | "loose"
    diag        : dict of diagnostic values for QC / JSON logging

    Pipeline
    --------
    1 → Seed    2 → Model    3 → Candidate    4 → Grow
    5 → Select  6 → Rachis   7 → Union
    8 → Hole fill (BEFORE _remove_noise)
    9 → Final close + clean + padding exclusion
    """
    
    img_area      = img_resized.shape[0] * img_resized.shape[1]
    img_lab_u8    = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    img_lab_float = img_lab_u8.astype(np.float32)
    hsv           = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    is_padding    = np.all(img_resized >= 252, axis=2)
    k3            = np.ones((3, 3), np.uint8)

    # Stage 1: seed
    seed, seed_cov, seed_relaxed = _build_seed(
        img_resized, img_lab_float, hsv, is_padding, k3, min_comp_frac, img_area
    )

    # Stage 2: per-image leaf colour model
    mean_lab, std_lab = _learn_leaf_model(img_lab_float, seed)

    # Stage 3: candidate map (S-gated shadow exclusion)
    candidate = _build_candidate_map(
        img_lab_float, hsv, mean_lab, std_lab, sigma_thresh
    )
    candidate[is_padding] = 0

    # Stage 4: region grow
    grown = _grow_seed(seed, candidate, n_iterations=40, kernel_size=5)
    grown = _remove_noise(grown, min_frac=min_comp_frac, img_area=img_area)

    # Stage 5: tight / loose structure selection
    leaflet_mask, mask_choice, n_tight, tight_area, loose_area = _select_structure(
        grown, min_comp_frac, img_area
    )

    # Stage 6: rachis mask (proximity-gated, no open, no remove_noise)
    rachis_mask = _build_rachis_mask(
        img_resized, img_lab_float, hsv, leaflet_mask, is_padding, proximity_px=15
    )
    rachis_px = int((rachis_mask > 0).sum())

    # Stage 7: union of leaflet + rachis masks
    combined = cv2.bitwise_or(leaflet_mask, rachis_mask)

    # preprocessing/shared/masking.py — inside select_mask(), after Stage 8:

    # Stage 8: flood-fill holes BEFORE _remove_noise
    filled = _fill_holes(combined)

    # Stage 9: final clean
    mask_final = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k3, iterations=1)
    mask_final = _remove_noise(mask_final, min_frac=min_comp_frac, img_area=img_area)
    is_paper_leak = (img_lab_float[:, :, 0] > 175) & (hsv[:, :, 1].astype(np.float32) < 25)
    mask_final[is_paper_leak] = 0
    mask_final = _remove_noise(mask_final, min_frac=min_comp_frac, img_area=img_area)
    mask_final[is_padding] = 0

    coverage     = float((mask_final > 0).sum()) / img_area
    n_components = cv2.connectedComponentsWithStats(mask_final)[0] - 1

    diag = {
        # Seed
        "seed_coverage_pct":  round(seed_cov, 2),
        "seed_relaxed":       seed_relaxed,
        # Colour model
        "leaf_mean_LAB":      mean_lab.round(1).tolist(),
        "leaf_std_LAB":       std_lab.round(1).tolist(),
        "sigma_thresh":       sigma_thresh,
        # Pipeline percentages
        "candidate_pct":      round(float((candidate > 0).sum()) / img_area * 100, 2),
        "grown_pct":          round(float((grown > 0).sum()) / img_area * 100, 2),
        "leaflet_pct":        round(float((leaflet_mask > 0).sum()) / img_area * 100, 2),
        # Rachis
        "rachis_px":          rachis_px,
        "rachis_pct":         round(rachis_px / img_area * 100, 3),
        # Structure selection
        "n_tight_components": n_tight,
        "tight_area_px":      tight_area,
        "loose_area_px":      loose_area,
        "mask_choice":        mask_choice,
        # Final
        "coverage_pct":       round(coverage * 100, 2),
        "n_final_components": n_components,
        # NEW — Stage-7 union mask, pre-hole-fill. Required by
        # feature_extraction/health/holes.py and scar.py. Was silently
        # dropped before (never in diag), so hole_count/scar_tissue_ratio
        # were sentinel on EVERY row, not just augmented ones. Carried as
        # the raw uint8 array (0/255), same convention as mask_final.
        "mask_before_holefill": combined.copy(),
        # NEW — Stage-6 rachis mask, required by feature_extraction/health/
        # boundary.py and scar.py to gate natural leaflet-junction
        # concavity/shadow out of margin-damage features (see project memory
        # -- previously this only lived as a pixel COUNT (rachis_px), never
        # as the actual mask, so boundary/scar had no way to use it).
        "rachis_mask": rachis_mask.copy(),
    }

    return mask_final, mask_choice, diag


def qc_check(diag: dict,
             min_cov: float = 0.02,
             max_cov: float = 0.75) -> tuple[bool, str]:
    """
    Quick QC pass/fail on mask diagnostics.

    Parameters
    ----------
    diag    : diagnostic dict returned by select_mask()
    min_cov : minimum acceptable coverage fraction (default 2%)
    max_cov : maximum acceptable coverage fraction (default 75%)

    Returns
    -------
    passed : bool   — True if mask is within expected coverage range
    reason : str    — empty on pass; human-readable failure description
    """
    cov = diag["coverage_pct"] / 100.0
    if cov < min_cov:
        return False, (
            f"coverage {cov*100:.1f}% < {min_cov*100:.0f}% "
            f"(leaf not detected — check image quality or species thresholds)"
        )
    if cov > max_cov:
        return False, (
            f"coverage {cov*100:.1f}% > {max_cov*100:.0f}% "
            f"(background leaking — check rachis proximity_px or sigma_thresh)"
        )
    return True, ""
import cv2
import numpy as np

from app.module3_compound_leaves.preprocessing.config import MIN_COMP_FRAC, SIGMA_THRESH



# INTERNAL HELPERS

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
  
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean  = np.zeros_like(mask)
    min_px = int(img_area * min_frac)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            clean[labels == i] = 255
    return clean


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    
    
    h, w   = mask.shape
    inv    = cv2.bitwise_not(mask)
    flood  = inv.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 0)
    # flood now == 255 only at enclosed holes
    return cv2.bitwise_or(mask, flood)



# STAGE 1 — SEED


def _build_seed(img_resized: np.ndarray,
                img_lab_float: np.ndarray,
                hsv: np.ndarray,
                is_padding: np.ndarray,
                k3: np.ndarray,
                min_comp_frac: float,
                img_area: int) -> tuple[np.ndarray, float, bool]:
    
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



# STAGE 2 — COLOUR MODEL


def _learn_leaf_model(img_lab_float: np.ndarray,
                      seed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    px = img_lab_float[seed_mask > 0]
    if len(px) < 50:
        return np.array([100.0, 115.0, 130.0]), np.array([20.0, 10.0, 10.0])
    return px.mean(axis=0), np.maximum(px.std(axis=0), 8.0)



# STAGE 3 — CANDIDATE MAP


def _build_candidate_map(img_lab_float: np.ndarray,
                         hsv: np.ndarray,
                         mean_lab: np.ndarray,
                         std_lab: np.ndarray,
                         sigma_thresh: float = SIGMA_THRESH) -> np.ndarray:
   
    diff       = np.abs(img_lab_float - mean_lab)
    z          = diff / std_lab
    sigma_gate = z.max(axis=2) < sigma_thresh

    s_ch         = hsv[:, :, 1].astype(np.float32)
    L_ch         = img_lab_float[:, :, 0]
    is_shadow    = s_ch < 25                        # achromatic darks
    is_paper_gap = (s_ch < 30) & (L_ch > 160)      # pale AND achromatic

    return (sigma_gate & ~is_shadow & ~is_paper_gap).astype(np.uint8) * 255



# STAGE 4 — REGION GROW


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


# STAGE 5 — STRUCTURE SELECTION


def _select_structure(grown: np.ndarray,
                      min_comp_frac: float,
                      img_area: int) -> tuple[np.ndarray, str, int, int, int]:
    
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



# STAGE 6 — RACHIS MASK


def _build_rachis_mask(img_resized: np.ndarray,
                       img_lab_float: np.ndarray,
                       hsv: np.ndarray,
                       leaflet_mask: np.ndarray,
                       is_padding: np.ndarray,
                       proximity_px: int = 15) -> np.ndarray:
    
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


# PUBLIC API
def select_mask(img_resized: np.ndarray,
                min_comp_frac: float = MIN_COMP_FRAC,
                sigma_thresh: float  = SIGMA_THRESH
                ) -> tuple[np.ndarray, str, dict]:
    
    
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

    # Stage 8: flood-fill holes BEFORE _remove_noise
    # Order is critical: _remove_noise after fill so filled leaflet interiors
    # are not deleted as small isolated components
    filled = _fill_holes(combined)

    # Stage 9: final clean
    mask_final = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k3, iterations=1)
    mask_final = _remove_noise(mask_final, min_frac=min_comp_frac, img_area=img_area)
    # In select_mask(), after the final _remove_noise call, add:
    is_paper_leak = (img_lab_float[:, :, 0] > 175) & (hsv[:, :, 1].astype(np.float32) < 25)
    mask_final[is_paper_leak] = 0
    mask_final = _remove_noise(mask_final, min_frac=min_comp_frac, img_area=img_area)  # re-clean
    mask_final[is_padding] = 0   # belt-and-braces: remove any border artefacts

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
    }

    return mask_final, mask_choice, diag


def qc_check(diag: dict,
             min_cov: float = 0.02,
             max_cov: float = 0.75) -> tuple[bool, str]:
   
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
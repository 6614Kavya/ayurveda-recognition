"""
VedaVision — Background Removal v4.1 (Shadow-Aware)
=====================================================
Hybrid seed + region-growing algorithm with shadow veto.

Algorithm stages
----------------
1. SEED       High-confidence leaf pixels via ExG + Saturation + L-channel veto.
              Tier-1 (tight), Tier-2 relaxed fallback for dark species.
2. MODEL      Learn per-image leaf LAB colour mean ± std from seed pixels.
3. CANDIDATE  Colour-model gate + shadow fingerprint veto → candidate pixel map.
4. GROW       Iterative dilation constrained to candidate map.
5. SELECT     Tight (k3 close) vs loose (k5 close) based on component count & area.

Shadow fingerprint (v4.1 fix):
  Shadow on white paper = high L (bright) AND near-neutral a channel.
  Leaf tissue           = lower L AND green-shifted a (< 128 in OpenCV scale).
  Both conditions must hold simultaneously to veto (AND gate, not OR).
"""

import cv2
import numpy as np
from preprocessing.config import MIN_COMP_FRAC, SIGMA_THRESH


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _remove_noise(mask: np.ndarray,
                  min_frac: float = MIN_COMP_FRAC,
                  img_area: int = 512 * 512) -> np.ndarray:
    """Drop connected components smaller than min_frac of image area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean   = np.zeros_like(mask)
    min_px  = int(img_area * min_frac)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            clean[labels == i] = 255
    return clean


def _learn_leaf_model(img_lab_float: np.ndarray,
                      seed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-image leaf LAB colour model (mean ± std) from seed pixels.
    Clamps std to ≥ 8 to avoid over-tight gates on uniform leaves.
    """
    px = img_lab_float[seed_mask > 0]
    if len(px) < 50:
        return np.array([128.0, 128.0, 128.0]), np.array([40.0, 40.0, 40.0])
    mean = px.mean(axis=0)
    std  = np.maximum(px.std(axis=0), 8.0)
    return mean, std


def _build_candidate_map(img_lab_float: np.ndarray,
                         mean_lab: np.ndarray,
                         std_lab: np.ndarray,
                         sigma_thresh: float = SIGMA_THRESH) -> np.ndarray:
    """
    Candidate pixels: within sigma_thresh of leaf model AND not a shadow.

    Shadow fingerprint (OpenCV uint8 → float32 scale):
      L > 120   (brighter than typical leaf tissue)
      |a - 128| < 12   (too neutral to be green leaf)
    Both must be true simultaneously (AND gate).
    """
    diff        = np.abs(img_lab_float - mean_lab)
    z           = diff / std_lab
    sigma_gate  = z.max(axis=2) < sigma_thresh

    L_ch        = img_lab_float[:, :, 0]
    a_ch        = img_lab_float[:, :, 1]
    is_shadow   = (L_ch > 120) & (np.abs(a_ch - 128) < 12)

    candidate   = (sigma_gate & ~is_shadow).astype(np.uint8) * 255
    return candidate


def _grow_seed(seed_mask: np.ndarray,
               candidate_mask: np.ndarray,
               n_iterations: int = 40,
               kernel_size: int = 5) -> np.ndarray:
    """Iteratively dilate the seed, constrained to the candidate map."""
    k     = np.ones((kernel_size, kernel_size), np.uint8)
    grown = seed_mask.copy()
    prev_sum = -1
    for _ in range(n_iterations):
        dilated  = cv2.dilate(grown, k, iterations=1)
        grown    = cv2.bitwise_and(dilated, candidate_mask)
        curr_sum = int(grown.sum())
        if curr_sum == prev_sum:
            break
        prev_sum = curr_sum
    return grown


def _build_seed(img_resized: np.ndarray,
                img_lab_float: np.ndarray,
                is_padding: np.ndarray,
                k3: np.ndarray,
                min_comp_frac: float,
                img_area: int) -> tuple[np.ndarray, float, bool]:
    """
    High-confidence seed with two-tier fallback for dark species.

    Tier 1 (normal):  ExG > 20, S > 20, L < 115
    Tier 2 (relaxed): ExG > 8,  S > 15, L < 145   ← only if tier 1 < 1% coverage
    """
    img_f = img_resized.astype(np.float32)
    exg   = 2.0 * img_f[:, :, 1] - img_f[:, :, 2] - img_f[:, :, 0]
    hsv   = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    s_ch  = hsv[:, :, 1].astype(np.float32)
    L_ch  = img_lab_float[:, :, 0]

    # Tier 1
    seed = ((exg > 20) & (s_ch > 20) & (L_ch < 115)).astype(np.uint8) * 255
    seed[is_padding] = 0
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k3, iterations=1)
    seed = _remove_noise(seed, min_frac=min_comp_frac, img_area=img_area)

    seed_coverage     = float((seed > 0).sum()) / img_area * 100.0
    seed_relaxed_flag = False

    # Tier 2 fallback (dark-species: kattakumanjal, etc.)
    if seed_coverage < 1.0:
        seed = ((exg > 8) & (s_ch > 15) & (L_ch < 145)).astype(np.uint8) * 255
        seed[is_padding] = 0
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, k3, iterations=1)
        seed = _remove_noise(seed, min_frac=min_comp_frac, img_area=img_area)
        seed_coverage     = float((seed > 0).sum()) / img_area * 100.0
        seed_relaxed_flag = True

    return seed, seed_coverage, seed_relaxed_flag


# ─── Public API ───────────────────────────────────────────────────────────────

def select_mask(img_resized: np.ndarray,
                min_comp_frac: float = MIN_COMP_FRAC,
                sigma_thresh: float  = SIGMA_THRESH
                ) -> tuple[np.ndarray, str, dict]:
    """
    Background removal v4.1 — shadow-aware hybrid seed + region growing.

    Parameters
    ----------
    img_resized   : letterboxed BGR uint8 image (512 × 512)
    min_comp_frac : minimum component size fraction (default from config)
    sigma_thresh  : LAB colour-model sigma gate (default from config)

    Returns
    -------
    mask_final  : uint8 binary mask (255 = foreground leaf)
    mask_choice : "tight" | "loose"
    diag        : dict of diagnostic values for QC/JSON logging
    """
    img_area      = img_resized.shape[0] * img_resized.shape[1]
    img_lab_u8    = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    img_lab_float = img_lab_u8.astype(np.float32)
    is_padding    = np.all(img_resized >= 252, axis=2)

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    # Stage 1: seed
    seed, seed_coverage, seed_relaxed = _build_seed(
        img_resized, img_lab_float, is_padding, k3, min_comp_frac, img_area
    )

    # Stage 2: per-image leaf model
    mean_lab, std_lab = _learn_leaf_model(img_lab_float, seed)

    # Stage 3: candidate + grow
    candidate = _build_candidate_map(img_lab_float, mean_lab, std_lab, sigma_thresh)
    candidate[is_padding] = 0
    grown = _grow_seed(seed, candidate, n_iterations=40, kernel_size=5)
    grown = _remove_noise(grown, min_frac=min_comp_frac, img_area=img_area)

    # Stage 4: tight vs loose structure selection
    mask_tight = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k3, iterations=1)
    mask_tight = _remove_noise(mask_tight, min_frac=min_comp_frac, img_area=img_area)
    mask_loose = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k5, iterations=1)
    mask_loose = _remove_noise(mask_loose, min_frac=min_comp_frac, img_area=img_area)

    n_tight    = cv2.connectedComponentsWithStats(mask_tight)[0] - 1
    tight_area = int((mask_tight > 0).sum())
    loose_area = int((mask_loose > 0).sum())

    # Tight if: leaflets clearly separated (≥3 components) AND tight keeps ≥25% of loose
    use_tight   = (n_tight >= 3) and (tight_area >= loose_area * 0.25)
    mask_final  = mask_tight if use_tight else mask_loose
    mask_choice = "tight" if use_tight else "loose"

    coverage = float((mask_final > 0).sum()) / img_area

    diag = {
        "seed_coverage_pct"   : round(seed_coverage, 2),
        "seed_relaxed"        : seed_relaxed,
        "leaf_mean_LAB"       : mean_lab.round(1).tolist(),
        "leaf_std_LAB"        : std_lab.round(1).tolist(),
        "sigma_thresh"        : sigma_thresh,
        "candidate_pct"       : round(float((candidate > 0).sum()) / img_area * 100, 2),
        "grown_pct"           : round(float((grown > 0).sum()) / img_area * 100, 2),
        "n_tight_components"  : n_tight,
        "tight_area_px"       : tight_area,
        "loose_area_px"       : loose_area,
        "mask_choice"         : mask_choice,
        "coverage_pct"        : round(coverage * 100, 2),
    }

    return mask_final, mask_choice, diag


def qc_check(diag: dict,
             min_cov: float = 0.02,
             max_cov: float = 0.75) -> tuple[bool, str]:
    """
    Quick QC pass/fail on mask diagnostics.

    Returns (passed: bool, reason: str).
    'reason' is empty string on pass.
    """
    cov = diag["coverage_pct"] / 100.0
    if cov < min_cov:
        return False, f"coverage {cov*100:.1f}% < {min_cov*100:.0f}% (leaf not detected)"
    if cov > max_cov:
        return False, f"coverage {cov*100:.1f}% > {max_cov*100:.0f}% (background leak)"
    return True, ""

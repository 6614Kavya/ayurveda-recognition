
import cv2
import numpy as np

try:
    from scipy.interpolate import griddata
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from app.module3_compound_leaves.preprocessing.shared.masking import select_mask



# Stage 0 — paper-referenced illumination flattening (unchanged from the

def estimate_illumination(img_resized: np.ndarray, grid: int = 16):
   
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

    illum, target_L = estimate_illumination(img_resized, grid=grid)
    ratio = np.clip(target_L / np.maximum(illum, 1e-3), clip_range[0], clip_range[1])
    img_f = img_resized.astype(np.float32)
    flattened = np.clip(img_f * ratio[:, :, None], 0, 255).astype(np.uint8)
    return flattened


def shadow_bleed_fraction(img_bgr: np.ndarray, mask: np.ndarray) -> float:
  
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    fg = mask > 0
    if fg.sum() == 0:
        return 0.0
    return float(((s < 20) & fg).sum()) / float(fg.sum())


def coverage_pct(mask: np.ndarray) -> float:
  
    return float((mask > 0).sum()) / float(mask.size)



# THE GUARD

def select_mask_guarded(img_resized: np.ndarray,
                         grid: int = 16,
                         clip_range: tuple = (0.6, 1.8),
                         skip_flatten_if_baseline_below: float = 0.003,
                         max_coverage_drop_ratio: float = 0.08,
                         return_all: bool = False):
  
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
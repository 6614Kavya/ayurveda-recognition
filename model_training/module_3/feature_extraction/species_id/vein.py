"""
VedaVision — Vein Features  (small-leaf-stable revision)
=========================================================
Skeleton-based vein descriptors, all normalised by leaf area or perimeter.

Previous revisions fixed
------------------------
  v1 — original
  v2 — shadow-robust: CLAHE before top-hat, top-hat on full image, mask after

This revision fixes
-------------------
PROBLEM 4 — Small leaf area in frame (the dominant failure mode)

  ROOT CAUSE:
  Most images have the compound leaf occupying a small fraction of the
  512×512 frame.  Three things break simultaneously:

  (a) CLAHE tile mismatch
      Fixed 8×8 tile grid → 64×64 px tiles at 512px image size.
      When the leaf ROI is, say, 120×80 px, every tile is LARGER than the
      leaf itself.  CLAHE equalises background pixels in each tile, not
      leaf tissue.  Shadow normalisation fails completely.

  (b) Top-hat kernel too large relative to the leaf
      The 15px ellipse kernel is calibrated for vein width at 512px scale.
      On a 120px-wide leaf, 15px equals ~6% of leaf width — far too large.
      It bridges across inter-vein lamina, suppressing the vein signal.

  (c) Skeleton pixel counts collapse to near zero
      Tiny skeleton → vein_density ≈ 0, branch_density ≈ 0.
      These look like "no veins" to the classifier, which is wrong.
      All species then appear identical on vein features — removing their
      most discriminative signal.

  FIX — ROI-upscale pipeline (unconditional, not a fallback gate):
  ----------------------------------------------------------------
  1. Crop tightly to the leaf bounding box (+ small pad).
  2. Upscale the crop to a fixed working resolution (WORK_SIZE = 512px
     longest side) using INTER_CUBIC.
  3. Run CLAHE + top-hat + adaptive threshold + skeletonise on the
     upscaled crop — all kernels now see the leaf at a consistent scale.
  4. Downscale the binary vein map back to the original crop size.
  5. Place results back into the full-frame output arrays.
  6. Compute density ratios using the ORIGINAL mask pixel count
     (not the upscaled count) to keep units consistent with other
     feature files.

  Why upscale instead of just crop?
  - Cropping alone does not fix (a) or (b): if the leaf is 120px wide
    the tiles are still too large and the kernel is still too coarse.
  - Upscaling to a fixed resolution decouples the algorithm parameters
    (tile size, kernel size, adaptive block size) from the physical size
    of the leaf in the image.  The same hyperparameters work for every
    image regardless of how close or far the leaf was photographed.
  - This is exactly what MobileNetV2 does: resize to 224×224 so filters
    always see features at the same scale.  We do the same for vein
    extraction.

  WORK_SIZE choice (512px):
  - Matches the main pipeline resolution.
  - Large enough for secondary veins (target ≈15px wide at this scale)
    to be detected reliably.
  - INTER_CUBIC upscale preserves sub-pixel vein edges better than
    INTER_LINEAR for the small→large magnification needed here.

Prior fixes preserved
---------------------
  PROBLEM 1 — CLAHE before top-hat for shadow normalisation  ✓ preserved
  PROBLEM 2 — Foreground area as denominator (not 512²)       ✓ preserved
  PROBLEM 3 — Top-hat on unmasked then clamp                  ✓ preserved
              (applied to the upscaled crop without masking,
               then clamped — same principle, new coordinate space)

Kept features (dimensionless, scale-invariant):
    vein_density            skeleton_px  / leaf_area_px   (original coords)
    vein_length_ratio       skeleton_px  / contour_perimeter
    vein_branch_density     branch_pts   / leaf_area_px
    vein_end_point_density  end_pts      / leaf_area_px

New diagnostic output columns (written to CSV, not used by classifier):
    vein_coverage_pct       leaf_area_px / (H * W)  — audit use only
    vein_roi_scale          upscale factor applied   — audit use only
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from preprocessing.config import GLCM_DIST, GLCM_ANGLES   # kept for config parity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All vein processing is done at this resolution (longest side).
# Keeps kernel sizes, CLAHE tiles and adaptive block sizes consistent
# across images regardless of how small the leaf is in the frame.
WORK_SIZE   = 512

# Padding around the bounding box before upscale.
# Prevents edge-halo artefacts from top-hat at the crop boundary.
ROI_PAD_PX  = 12

# Minimum linear dimension (px) of the crop before upscale.
# Crops smaller than this have no usable vein detail — return zeros.
MIN_CROP_PX = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_padded_bbox(leaf_mask: np.ndarray,
                     pad: int = ROI_PAD_PX
                     ) -> tuple[int, int, int, int] | None:
    """
    Return (x1, y1, x2, y2) of the foreground bounding box,
    expanded by `pad` pixels on all sides and clamped to image bounds.
    Returns None if no foreground found.
    """
    coords = cv2.findNonZero(leaf_mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    H, W = leaf_mask.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return x1, y1, x2, y2


def _upscale_to_work_size(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Scale the longest side of `img` to WORK_SIZE using INTER_CUBIC.
    Returns (upscaled_img, scale_factor).
    scale_factor > 1 means we upscaled (small leaf → work size).
    scale_factor < 1 means the crop was already larger (downscaled).
    """
    h, w = img.shape[:2]
    scale = WORK_SIZE / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp), scale


def _build_vein_map(gray_work: np.ndarray,
                    mask_work: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full CLAHE → top-hat → adaptive threshold → skeletonise
    pipeline on `gray_work` (already at WORK_SIZE resolution).

    `mask_work` is the upscaled binary mask (uint8, 255=fg).

    Returns (vein_skel_work, vein_binary_work) both at WORK_SIZE.

    All hyperparameters are calibrated for WORK_SIZE=512:
      - CLAHE tile grid  : 8×8  → 64×64 px tiles
      - Top-hat kernel   : 15px ellipse  (secondary vein width ≈ 10-15px)
      - Adaptive block   : max(11, H//20) rounded to odd
    These stay constant because the image is always upscaled to WORK_SIZE
    before this function is called.
    """
    # Step A: CLAHE on full grayscale (no mask) — shadow normalisation
    # Applied BEFORE masking so each tile has a realistic local histogram
    # that includes both vein and lamina pixels.
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq  = clahe.apply(gray_work)

    # Step B: Black top-hat on FULL equalised image (no mask yet)
    # Applied to unmasked image so boundary sees real image content,
    # not the artificial zero-edge that masking would create.
    k_bthat      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    black_tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, k_bthat)

    # Step C: Clamp to foreground AFTER top-hat
    black_tophat = cv2.bitwise_and(black_tophat, black_tophat, mask=mask_work)

    # Step D: Adaptive threshold
    blk = max(11, (gray_work.shape[0] // 20) | 1)
    vein_binary = cv2.adaptiveThreshold(
        black_tophat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blk, -2,
    )
    vein_binary = cv2.bitwise_and(vein_binary, vein_binary, mask=mask_work)

    # Step E: Skeletonise → 1-px vein centrelines
    vein_skel = skeletonize(vein_binary > 0).astype(np.uint8) * 255

    return vein_skel, vein_binary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_vein_features(img_sharp_bgr: np.ndarray,
                          leaf_mask: np.ndarray
                          ) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Extract shadow-robust, small-leaf-stable vein features.

    Parameters
    ----------
    img_sharp_bgr : enhanced BGR uint8 image (output of enhance.py), 512×512
    leaf_mask     : uint8 binary mask (255 = foreground), 512×512

    Returns
    -------
    feats       : dict — 4 vein features + 2 diagnostic columns
    vein_skel   : uint8 skeleton image in ORIGINAL 512×512 frame
    vein_binary : uint8 thresholded vein map in ORIGINAL 512×512 frame

    Pipeline summary
    ----------------
    1. Crop to padded leaf bounding box.
    2. Upscale crop to WORK_SIZE (512px longest side) with INTER_CUBIC.
       → All subsequent ops see the leaf at a consistent scale regardless
         of how small it was in the original frame.
    3. CLAHE (8×8 tiles = 64px each at WORK_SIZE) on full grayscale.
    4. Black top-hat (15px ellipse) on full equalised image, then mask.
    5. Adaptive threshold + skeletonise.
    6. Downscale vein maps back to original crop size, place in full frame.
    7. Compute density ratios using ORIGINAL mask pixel count (not upscaled)
       so that features are consistent with colour/texture/shape features
       which all use the original 512×512 coordinate space.
    """
    gray    = cv2.cvtColor(img_sharp_bgr, cv2.COLOR_BGR2GRAY)
    H, W    = gray.shape[:2]
    px_mask = leaf_mask > 0

    # Guard: completely empty mask
    if px_mask.sum() < 100:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {
            "vein_density": 0.0, "vein_length_ratio": 0.0,
            "vein_branch_density": 0.0, "vein_end_point_density": 0.0,
            "vein_coverage_pct": 0.0, "vein_roi_scale": 1.0,
        }, empty, empty

    # ── Diagnostic: coverage in original frame ────────────────────────────
    leaf_area_px    = float(px_mask.sum())
    coverage_pct    = leaf_area_px / float(H * W)

    # ── Step 1: Crop to padded bounding box ───────────────────────────────
    bbox = _get_padded_bbox(leaf_mask, pad=ROI_PAD_PX)
    if bbox is None:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {
            "vein_density": 0.0, "vein_length_ratio": 0.0,
            "vein_branch_density": 0.0, "vein_end_point_density": 0.0,
            "vein_coverage_pct": round(coverage_pct, 4), "vein_roi_scale": 1.0,
        }, empty, empty

    x1, y1, x2, y2 = bbox
    gray_crop = gray[y1:y2, x1:x2]
    mask_crop = leaf_mask[y1:y2, x1:x2]

    crop_h, crop_w = gray_crop.shape[:2]

    # Guard: crop too small to extract any vein detail
    if min(crop_h, crop_w) < MIN_CROP_PX:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {
            "vein_density": 0.0, "vein_length_ratio": 0.0,
            "vein_branch_density": 0.0, "vein_end_point_density": 0.0,
            "vein_coverage_pct": round(coverage_pct, 4), "vein_roi_scale": 0.0,
        }, empty, empty

    # ── Step 2: Upscale crop to WORK_SIZE ─────────────────────────────────
    # This is the key fix: every leaf is processed at the same effective
    # resolution, so CLAHE tiles, top-hat kernel and adaptive block size
    # always see veins at a consistent pixel scale.
    gray_work, roi_scale = _upscale_to_work_size(gray_crop)
    mask_work, _         = _upscale_to_work_size(mask_crop)
    # Re-binarise mask after interpolation artefacts from resize
    mask_work = (mask_work > 127).astype(np.uint8) * 255

    # ── Steps 3-5: CLAHE → top-hat → threshold → skeleton ────────────────
    vein_skel_work, vein_binary_work = _build_vein_map(gray_work, mask_work)

    # ── Step 6: Downscale results back to original crop size ──────────────
    # INTER_NEAREST preserves binary structure (no new grey values created).
    vein_skel_crop   = cv2.resize(vein_skel_work,   (crop_w, crop_h),
                                  interpolation=cv2.INTER_NEAREST)
    vein_binary_crop = cv2.resize(vein_binary_work, (crop_w, crop_h),
                                  interpolation=cv2.INTER_NEAREST)

    # Place crop results back into full 512×512 output frames
    vein_skel   = np.zeros((H, W), dtype=np.uint8)
    vein_binary = np.zeros((H, W), dtype=np.uint8)
    vein_skel[y1:y2, x1:x2]   = vein_skel_crop
    vein_binary[y1:y2, x1:x2] = vein_binary_crop

    # ── Step 7: Density features — original coordinate denominators ───────
    # leaf_area_px is from the original mask (512×512 space).
    # skel_px is counted in the full-frame vein_skel (same space).
    # This keeps vein features in the same units as colour/texture/shape.
    skel_px = float((vein_skel > 0).sum())

    feats: dict = {}
    feats["vein_density"] = skel_px / leaf_area_px if leaf_area_px > 0 else 0.0

    # ── Length ratio (normalised by contour perimeter) ─────────────────────
    # Perimeter from original mask — consistent with shape features.
    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = (float(cv2.arcLength(max(cnts, key=cv2.contourArea), True))
                 if cnts else 1.0)
    feats["vein_length_ratio"] = skel_px / perimeter if perimeter > 0 else 0.0

    # ── Branch & end-point densities ──────────────────────────────────────
    k_n    = np.ones((3, 3), np.uint8);  k_n[1, 1] = 0
    skel_b = (vein_skel > 0).astype(np.uint8)
    nbr    = cv2.filter2D(skel_b.astype(np.float32), -1, k_n.astype(np.float32))
    nbr    = (nbr * skel_b).astype(np.uint8)

    branch_pts = int((nbr >= 3).sum())
    end_pts    = int((nbr == 1).sum())

    feats["vein_branch_density"]    = branch_pts / leaf_area_px if leaf_area_px > 0 else 0.0
    feats["vein_end_point_density"] = end_pts    / leaf_area_px if leaf_area_px > 0 else 0.0

    # ── Diagnostic columns (not used by classifier, for audit CSV only) ───
    feats["vein_coverage_pct"] = round(coverage_pct, 4)
    feats["vein_roi_scale"]    = round(roi_scale, 4)

    return feats, vein_skel, vein_binary
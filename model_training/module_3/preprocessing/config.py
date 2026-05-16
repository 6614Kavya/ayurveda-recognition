"""
VedaVision — Preprocessing Configuration
=========================================
Single source of truth for all pipeline parameters.
Import this in every module; never hardcode magic numbers elsewhere.
"""

from pathlib import Path

# ── Resolution ────────────────────────────────────────────────────────────────
TARGET_LONG   = 512       # Longest side px. Power-of-two; preserves vein detail
                          # on small Moringa leaflets (224 would under-resolve).
CLASSIFIER_RES = 224      # CNN input resolution (resize after feature extraction)

# ── Background removal ────────────────────────────────────────────────────────
MIN_COMP_FRAC  = 0.001    # Minimum connected component as fraction of image area.
                          # 0.1% of 512² = 262 px² — removes dust/speckle noise.
SIGMA_THRESH   = 2.5      # LAB colour-model gate: pixels > N sigma from leaf mean
                          # are excluded as background.

# ── Texture features ──────────────────────────────────────────────────────────
LBP_RADIUS    = 3         # LBP neighbourhood radius (px)
LBP_POINTS    = 24        # LBP sample points = 8 × radius (standard)
GLCM_DIST     = [1, 3]    # GLCM distances: 1=immediate neighbours, 3=medium range
GLCM_ANGLES   = [0, 1.5707963, 0.7853981, 2.3561944]  # 0°, 90°, 45°, 135°

# ── Enhancement (species-ID branch only) ─────────────────────────────────────
BILATERAL_D          = 9
BILATERAL_SIGMA      = 75
CLAHE_CLIP           = 2.5
CLAHE_TILE           = (8, 8)
UNSHARP_SIGMA        = 3
UNSHARP_STRENGTH     = 1.5

# ── QC thresholds ─────────────────────────────────────────────────────────────
QC_MIN_COVERAGE  = 0.02   # < 2%  → leaf not detected → flag FAIL
QC_MAX_COVERAGE  = 0.75   # > 75% → background leaking → flag FAIL
QC_MIN_AREA_FRAC = 0.001  # component filter (same as MIN_COMP_FRAC)

# ── Dataset folder layout ─────────────────────────────────────────────────────
# dataset/raw/<species>/top/*.jpg
# dataset/raw/<species>/bottom/*.jpg
VIEWS = ["top", "bottom"]
IMG_EXTS = ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]

# ── Output folder layout ──────────────────────────────────────────────────────
# processed/<species>/top/enhanced/    ← CNN input
# processed/<species>/top/masked_raw/  ← Health input
# processed/<species>/top/summaries/   ← QC grid images
# features/                            ← CSVs
# diagnostics/                         ← per-image JSON + failure log

# ── Augmentation ──────────────────────────────────────────────────────────────
AUG_PER_IMAGE = 6         # Augmented variants per raw image (offline)
                          # 720 raw × 7 (orig + 6 aug) = 5040 CSV rows

# ── Batch processor ───────────────────────────────────────────────────────────
CHECKPOINT_EVERY = 50     # Save partial CSV every N images

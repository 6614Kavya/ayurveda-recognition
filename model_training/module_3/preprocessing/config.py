# =============================================================================
# VedaVision — Preprocessing Config
# Covers: Shared base + Species ID branch (diagram steps 1–4, 5a–9a)
# Health branch config will be added separately
# =============================================================================

# ── Image Settings ────────────────────────────────────────────────────────────
IMG_SIZE            = (512, 512)        # From diagram: 512×512 RGB PNG
IMG_EXTENSIONS      = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# ── Quality Filter ────────────────────────────────────────────────────────────
BLUR_THRESHOLD      = 80               # Laplacian variance — reject below this

# ── GrabCut + Saliency ────────────────────────────────────────────────────────
# Saliency map is used to find where the leaf actually is (handles off-center)
# GrabCut rect is derived from saliency bounding box, not fixed centre margin
GRABCUT_ITERATIONS  = 5
SALIENCY_BLUR_K     = 51              # Gaussian blur kernel for saliency
SALIENCY_THRESH     = 0.3            # Fraction of max saliency to threshold at

# ── Bilateral Filter (Step 4) ─────────────────────────────────────────────────
# Edge-preserving noise reduction — from diagram
BILATERAL_D         = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# ── CLAHE — Species ID branch (Step 5a) ───────────────────────────────────────
# From diagram: clipLimit=2.0, tile 8×8, LAB space
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_SIZE     = (8, 8)

# ── Edge Sharpening (Step 6a) ─────────────────────────────────────────────────
# Laplacian sharpening kernel — from diagram
SHARPEN_KERNEL      = [
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
]

# ── Normalization ─────────────────────────────────────────────────────────────
# ImageNet stats — for pretrained CNN input
IMAGENET_MEAN       = (0.485, 0.456, 0.406)
IMAGENET_STD        = (0.229, 0.224, 0.225)

# ── Augmentation ─────────────────────────────────────────────────────────────
AUG_PER_IMAGE       = 8

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_RAW         = '../dataset/raw'
PROCESSED_DIR       = '../dataset/species_id/processed'
REJECTED_DIR        = '../dataset/species_id/rejected'
AUGMENTED_DIR       = '../dataset/species_id/augmented'
REFERENCE_IMG_PATH  = '../dataset/reference.jpg'   # for histogram matching (future)

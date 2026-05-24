"""
VedaVision — augmentation.py
==============================
Offline data augmentation applied to RAW BGR images BEFORE any preprocessing.

Design rules (from project spec):
  • Applied to training images only — test images are NEVER augmented.
  • Applied to the raw image BEFORE resize → mask → enhance, so that
    geometric and photometric variations flow through the full feature
    extraction pipeline, producing a more robust feature distribution.
  • Augmented images are NOT saved to disk (fast to regenerate).
    Only the extracted feature rows are saved to CSV.
  • Default N_AUGMENTATIONS = 6 variants per original image.
    With 30 train images × 2 sides × 12 species × 6 aug = ~4,320 CSV rows
    (plus the 720 original rows = ~5,040 total).

Excluded transforms (with reasons):
  • RandomCrop / ElasticTransform — destroys leaflet structure and
    whole-leaf shape features (aspect ratio, convexity, Hu moments).
  • CoarseDropout — creates fake lesions that corrupt the health branch.
  • Strong colour shifts (>±20%) — would fabricate false yellowing /
    browning signals that confuse both species ID and health features.

Included transforms (all mild, field-realistic):
  • HorizontalFlip, VerticalFlip         — orientation invariance
  • Rotate ±30° (white border fill)      — matches dataset background
  • BrightnessContrast ±15%              — field lighting variation
  • HueSaturationValue (mild)            — colour cast variation
  • GaussianBlur (kernel 3–5 px)         — slight motion / focus blur
  • GaussNoise                           — sensor noise
  • RandomShadow                         — partial shadow simulation

Usage:
    from preprocessing.augmentation import augment_raw, N_AUGMENTATIONS

    img_bgr = cv2.imread(str(img_path))
    variants = augment_raw(img_bgr)
    # variants is a list of N_AUGMENTATIONS BGR numpy arrays
    # The original image is NOT included — add it separately if needed.
"""

import cv2
import numpy as np

try:
    import albumentations as A
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────

N_AUGMENTATIONS = 6   # synthetic variants per original image

# ── Transform pipeline ────────────────────────────────────────────────────────

def _build_transform() -> "A.Compose":
    """
    Build the Albumentations transform pipeline.

    Each call to the transform produces one augmented variant.
    All parameters are chosen to be field-realistic and mild enough
    not to fabricate false health signals or destroy leaf geometry.
    """
    if not _ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required for augmentation.\n"
            "Install with:  pip install albumentations"
        )

    return A.Compose([

        # ── Geometry (orientation only — no crop, no elastic) ──────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=30,                   # ±30 degrees
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,                   # white fill matches dataset background
            p=0.7,
        ),

        # ── Photometry (mild — must not fabricate health signals) ──────────
        A.RandomBrightnessContrast(
            brightness_limit=0.15,      # ±15% brightness
            contrast_limit=0.15,        # ±15% contrast
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,          # very mild hue shift (±8°)
            sat_shift_limit=15,         # ±15 saturation
            val_shift_limit=10,         # ±10 value
            p=0.5,
        ),

        # ── Blur / Noise (sensor and motion simulation) ────────────────────
        A.GaussianBlur(
            blur_limit=(3, 5),          # kernel size 3 or 5 px only
            p=0.3,
        ),
        A.GaussNoise(
            std_range=(0.01, 0.05),     # mild noise (1–5% of range)
            p=0.3,
        ),

        # ── Shadow simulation (partial occlusion from field conditions) ────
        A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),   # shadow can appear anywhere
            num_shadows_limit=(1, 2),
            shadow_dimension=4,
            p=0.3,
        ),

    ])


# Build once at module load (reused across all images for efficiency)
_TRANSFORM = None

def _get_transform() -> "A.Compose":
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = _build_transform()
    return _TRANSFORM


# ── Public API ────────────────────────────────────────────────────────────────

def augment_raw(img_bgr: np.ndarray,
                n: int = N_AUGMENTATIONS) -> list[np.ndarray]:
    """
    Generate N augmented variants of a raw BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Raw BGR image as loaded by cv2.imread().
        Must be called BEFORE resize, mask, or enhance steps.
    n : int
        Number of augmented variants to generate (default: N_AUGMENTATIONS=6).

    Returns
    -------
    list of np.ndarray
        N augmented BGR images. The original image is NOT included in this list.
        The caller decides whether to include it (batch_processor always does).

    Notes
    -----
    - Albumentations works in RGB internally; conversion is handled here.
    - Each call applies a random selection of transforms, so variants differ.
    - Never call this on test images.
    """
    if not _ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is not installed.\n"
            "Install with:  pip install albumentations"
        )

    transform = _get_transform()

    # Albumentations expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    variants = []
    for _ in range(n):
        augmented_rgb = transform(image=img_rgb)["image"]
        # Convert back to BGR for OpenCV downstream pipeline
        variants.append(cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR))

    return variants


def augment_raw_with_original(img_bgr: np.ndarray,
                               n: int = N_AUGMENTATIONS) -> list[np.ndarray]:
    """
    Return [original] + N augmented variants.

    Convenience wrapper used by the batch processor so the original image
    always goes through the pipeline alongside its augmented siblings.

    Returns
    -------
    list of np.ndarray
        Length = n + 1.  First element is always the unmodified original.
    """
    return [img_bgr] + augment_raw(img_bgr, n=n)

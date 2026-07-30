"""
VedaVision — augmentation.py
==============================
Offline data augmentation applied to RAW BGR images BEFORE any preprocessing.

Verified for albumentations==2.0.8 (the installed version).

Design rules (from project spec):
  • Applied to training images only — test images are NEVER augmented.
  • Applied to the raw image BEFORE resize → mask → enhance, so that
    geometric and photometric variations flow through the full feature
    extraction pipeline, producing a more robust feature distribution.
  • Augmented images are NOT saved to disk (fast to regenerate).
    Only the extracted feature rows are saved to CSV.
  • Default N_AUGMENTATIONS = 6 variants per original image.

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
    from preprocessing.shared.augmentation import augment_raw, N_AUGMENTATIONS

    img_bgr = cv2.imread(str(img_path))
    variants = augment_raw(img_bgr)
    # variants is a list of N_AUGMENTATIONS BGR numpy arrays
    # The original is NOT included — use augment_raw_with_original() for both.
"""

import cv2
import numpy as np
import albumentations as A

# ── Configuration ─────────────────────────────────────────────────────────────

N_AUGMENTATIONS = 6   # synthetic variants per original image

# ── Transform pipeline ────────────────────────────────────────────────────────

def _build_transform() -> A.Compose:
    """
    Build the Albumentations transform pipeline for albumentations==2.0.8.

    Parameter names verified against A.GaussNoise, A.Rotate, A.RandomShadow
    signatures in 2.0.8 — do not change without re-checking signatures.
    """
    return A.Compose([

        # ── Geometry (orientation only — no crop, no elastic) ──────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=30,                        # ±30 degrees
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,                        # white fill matches dataset background
            p=0.7,
        ),

        # ── Photometry (mild — must not fabricate health signals) ──────────
        A.RandomBrightnessContrast(
            brightness_limit=0.15,           # ±15% brightness
            contrast_limit=0.15,             # ±15% contrast
            p=0.7,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,               # very mild hue shift (±8°)
            sat_shift_limit=15,              # ±15 saturation
            val_shift_limit=10,              # ±10 value
            p=0.5,
        ),

        # ── Blur / Noise ───────────────────────────────────────────────────
        A.GaussianBlur(
            blur_limit=(3, 5),               # kernel size 3 or 5 px only
            p=0.3,
        ),
        # GaussNoise in 2.0.8: std_range is (0.0–1.0) normalised float range
        A.GaussNoise(
            std_range=(0.01, 0.05),          # 1–5% of pixel range — mild sensor noise
            p=0.3,
        ),

        # ── Shadow simulation ──────────────────────────────────────────────
        # RandomShadow in 2.0.8: shadow_dimension is valid (int, default=5)
        A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            num_shadows_limit=(1, 2),
            shadow_dimension=4,
            p=0.3,
        ),

    ])


# Build once at module load — reused across all images for efficiency
_TRANSFORM: A.Compose | None = None

def _get_transform() -> A.Compose:
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
        N augmented BGR images. The original is NOT included.
        Use augment_raw_with_original() to include it.

    Notes
    -----
    - Albumentations expects RGB; BGR↔RGB conversion is handled here.
    - Never call this on test images.
    """
    transform = _get_transform()

    # Albumentations works in RGB internally
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    variants = []
    for _ in range(n):
        augmented_rgb = transform(image=img_rgb)["image"]
        variants.append(cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR))

    return variants


def augment_raw_with_original(img_bgr: np.ndarray,
                               n: int = N_AUGMENTATIONS) -> list[np.ndarray]:
    """
    Return [original] + N augmented variants.

    The original image is always variants[0].
    Augmented variants are variants[1..N].

    Returns
    -------
    list of np.ndarray — length = n + 1
    """
    return [img_bgr] + augment_raw(img_bgr, n=n)



def _build_transform_with_mask() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,          # image border -> white (matches dataset background)
            fill_mask=0,       # mask border -> background (never leaf)
            mask_interpolation=cv2.INTER_NEAREST,  # keep mask strictly binary, no grey edges
            p=0.7,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            num_shadows_limit=(1, 2),
            shadow_dimension=4,
            p=0.3,
        ),
    ])


_TRANSFORM_WITH_MASK: A.Compose | None = None

def _get_transform_with_mask() -> A.Compose:
    global _TRANSFORM_WITH_MASK
    if _TRANSFORM_WITH_MASK is None:
        _TRANSFORM_WITH_MASK = _build_transform_with_mask()
    return _TRANSFORM_WITH_MASK


def augment_resized_with_mask(img_resized_bgr: np.ndarray,
                               mask_final: np.ndarray,
                               n: int = N_AUGMENTATIONS
                               ) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate N (augmented_image, augmented_mask) pairs from an ALREADY
    letterbox-resized image and its ALREADY-computed mask_final.

    Parameters
    ----------
    img_resized_bgr : 512x512 BGR image, output of letterbox_resize()
    mask_final       : 512x512 uint8 binary mask, output of select_mask()
                        run on img_resized_bgr (the CLEAN, un-augmented one)
    n                 : number of augmented variants (default N_AUGMENTATIONS)

    Returns
    -------
    list of (aug_img, aug_mask) tuples, length n. Original NOT included --
    use augment_resized_with_mask_and_original() for [orig] + n variants.
    """
    transform = _get_transform_with_mask()
    img_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)

    pairs = []
    for _ in range(n):
        out = transform(image=img_rgb, mask=mask_final)
        aug_img = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
        aug_mask = out["mask"]
        pairs.append((aug_img, aug_mask))
    return pairs


def augment_resized_with_mask_and_original(img_resized_bgr: np.ndarray,
                                            mask_final: np.ndarray,
                                            n: int = N_AUGMENTATIONS
                                            ) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return [(original_img, original_mask)] + N augmented (img, mask) pairs.
    variants[0] = original, variants[1..N] = augmented.
    """
    return [(img_resized_bgr, mask_final)] + augment_resized_with_mask(
        img_resized_bgr, mask_final, n=n
    )
# --- append to preprocessing/shared/augmentation.py --------------------------

def _build_transform_geo_only() -> A.Compose:
    """
    Health-branch augmentation: geometry ONLY, no photometric transforms.

    Rationale (see project memory): colour_health.py does hard per-pixel
    LAB/HSV threshold classification (necrotic/chlorotic/pale) and that
    classification IS the primary training signal for the health branch
    -- unlike species-ID, where colour features are a demonstrated weak
    discriminator. HueSaturationValue/BrightnessContrast/RandomShadow can
    silently nudge a borderline pixel across a threshold and fabricate a
    false severity shift. Geometric transforms (flip/rotate) cannot alter
    per-pixel colour classification at all, so they carry zero risk to
    colour_* features while still multiplying leaf coverage for the small
    (~15 img/level) health dataset.

    fill_mask=0 on both mask targets: rotation border must resolve to
    "not leaf" / "not hole-region", never leaf, for both masks.

    NEW: rachis_mask carried through the same warp too (registered as a
    third additional_targets entry) -- needed so boundary.py/scar.py's
    rachis-proximity gating stays geometrically correct on augmented rows,
    not just on the originals.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(
            limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,                    # image border -> white background
            fill_mask=0,                 # ALL mask targets -> background
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.7,
        ),
    ], additional_targets={"mask_before_holefill": "mask", "rachis_mask": "mask"})


_TRANSFORM_GEO_ONLY: A.Compose | None = None

def _get_transform_geo_only() -> A.Compose:
    global _TRANSFORM_GEO_ONLY
    if _TRANSFORM_GEO_ONLY is None:
        _TRANSFORM_GEO_ONLY = _build_transform_geo_only()
    return _TRANSFORM_GEO_ONLY


def augment_health_resized_with_masks(img_resized_bgr: np.ndarray,
                                       mask_final: np.ndarray,
                                       mask_before_holefill: np.ndarray,
                                       rachis_mask: np.ndarray,
                                       n: int = N_AUGMENTATIONS
                                       ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Health-branch augmentation. Generates N geometric-only variants,
    warping img + mask_final + mask_before_holefill + rachis_mask TOGETHER
    with the identical transform so hole_count/scar_tissue_ratio AND the
    boundary/scar rachis-gating stay valid on augmented rows (unlike the
    species-ID path, geometry-only warping doesn't change which pixels are
    "hole"/"rachis" vs not -- only where they sit -- so carrying all three
    masks through the same warp is safe, which is NOT true for photometric
    transforms).

    Parameters
    ----------
    img_resized_bgr        : 512x512 BGR, output of letterbox_resize()
    mask_final              : 512x512 uint8, from select_mask() on the
                               CLEAN (unaugmented) image
    mask_before_holefill    : 512x512 uint8, diag["mask_before_holefill"]
                               from the SAME select_mask() call
    rachis_mask             : 512x512 uint8, diag["rachis_mask"] from the
                               SAME select_mask() call

    Returns
    -------
    list of (aug_img, aug_mask_final, aug_mask_before_holefill,
    aug_rachis_mask), length n. Original NOT included -- see
    *_with_original() below.
    """
    transform = _get_transform_geo_only()
    img_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)

    quads = []
    for _ in range(n):
        out = transform(
            image=img_rgb,
            mask=mask_final,
            mask_before_holefill=mask_before_holefill,
            rachis_mask=rachis_mask,
        )
        aug_img = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
        quads.append((aug_img, out["mask"], out["mask_before_holefill"], out["rachis_mask"]))
    return quads


def augment_health_resized_with_masks_and_original(img_resized_bgr, mask_final,
                                                     mask_before_holefill, rachis_mask,
                                                     n: int = N_AUGMENTATIONS):
    """[(original quad)] + N augmented quads. variants[0] = original."""
    return [(img_resized_bgr, mask_final, mask_before_holefill, rachis_mask)] + \
        augment_health_resized_with_masks(img_resized_bgr, mask_final,
                                           mask_before_holefill, rachis_mask, n=n)
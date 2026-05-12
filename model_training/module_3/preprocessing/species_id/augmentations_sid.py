"""
VedaVision — Augmentation for Species ID Branch
For Classical CV + ML pipeline (SVM / Random Forest / XGBoost)

Key difference from DL augmentation:
    DL: augmented images fed directly to CNN → CNN learns invariance
    Classical: augmented images → features extracted from each → appended to feature matrix

So augmentation here:
    1. Creates more training images (offline, saved to disk)
    2. Feature extraction (step 8a) is run on EACH augmented image
    3. All feature vectors go into the training matrix
    → Classifier sees more varied feature distributions → better generalisation

Field-simulation transforms included to close the domain gap.
"""

import albumentations as A
import cv2
import numpy as np


# =============================================================================
# TRAINING AUGMENTATION — field-simulation
# =============================================================================

train_augment = A.Compose([

    # ── Geometric ─────────────────────────────────────────────────────────────
    # Compound leaves appear at any orientation in field photos
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.6),
    A.ShiftScaleRotate(
        shift_limit  = 0.08,
        scale_limit  = 0.12,
        rotate_limit = 0,
        border_mode  = cv2.BORDER_CONSTANT,
        p=0.4
    ),
    # Off-centre / angled capture simulation
    A.Perspective(scale=(0.04, 0.08), p=0.3),

    # ── Lighting / colour ─────────────────────────────────────────────────────
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20, p=0.5),
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, p=0.3),

    # ── Noise / compression ───────────────────────────────────────────────────
    A.GaussNoise(var_limit=(10.0, 35.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.ImageCompression(quality_lower=65, quality_upper=95, p=0.3),
])


# =============================================================================
# OFFLINE AUGMENTATION RUNNER
# =============================================================================

def augment_and_save(
    img_bgr: np.ndarray,
    save_dir,
    stem: str,
    n: int = 8
) -> list:
    """
    Creates n augmented versions of img_bgr and saves to save_dir.
    Returns list of saved file paths.

    Usage:
        paths = augment_and_save(img, Path('dataset/augmented/neem'), 'leaf_001', n=8)
    """
    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    saved    = []

    # Save original (no augmentation)
    orig_path = save_dir / f'{stem}_orig.png'
    cv2.imwrite(str(orig_path), img_bgr)
    saved.append(str(orig_path))

    # Save n augmented versions
    for i in range(n):
        aug    = train_augment(image=img_rgb)['image']
        aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
        path   = save_dir / f'{stem}_aug{i:02d}.png'
        cv2.imwrite(str(path), aug_bgr)
        saved.append(str(path))

    return saved

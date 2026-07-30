import numpy as np

from app.module2_single_leaves.preprocessing.Identification.enhancement import (
    apply_clahe, apply_light_denoise
)
from app.module2_single_leaves.preprocessing.Identification.segmentation import (
    segment_leaf, crop_to_leaf_bbox, remove_background, resize_with_padding
)
from app.module2_single_leaves.preprocessing.Identification.shadow_removal import correct_leaf_shadow

from app.module2_single_leaves.config import WORK_SIZE, TARGET_SIZE


def preprocess_image(image_bgr, use_rembg=False):
    if image_bgr is None:
        return None

    img = resize_with_padding(image_bgr, WORK_SIZE)
    mask = segment_leaf_rembg(img) if use_rembg else segment_leaf(img)

    coverage = np.sum(mask > 0) / mask.size
    if coverage < 0.02 or coverage > 0.95:
        return None

    img, mask = crop_to_leaf_bbox(img, mask)
    img       = correct_leaf_shadow(img, mask)
    no_bg     = remove_background(img, mask)
    clahe_img = apply_clahe(no_bg)
    denoised  = apply_light_denoise(clahe_img, mask)

    final = resize_with_padding(denoised, TARGET_SIZE)
    return final
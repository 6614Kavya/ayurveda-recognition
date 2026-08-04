import cv2
import numpy as np
from app.module2_single_leaves.preprocessing.health.segmentation import segment_leaf, crop_to_leaf_bbox
from app.module2_single_leaves.preprocessing.health.enhancement import remove_background, apply_clahe, apply_light_denoise

WORK_SIZE = (512, 512)
TARGET_SIZE = (384, 384)   
def preprocess_image(image_bgr):
    if image_bgr is None:
        return None

    img  = cv2.resize(image_bgr, WORK_SIZE)
    mask = segment_leaf(img)

    coverage = np.sum(mask > 0) / mask.size
    if coverage < 0.02 or coverage > 0.95:
        return None

    img, mask = crop_to_leaf_bbox(img, mask)
    no_bg     = remove_background(img, mask)
    clahe_img = apply_clahe(no_bg)
    denoised  = apply_light_denoise(clahe_img, mask)

    final = cv2.resize(denoised, TARGET_SIZE)
    return final
 
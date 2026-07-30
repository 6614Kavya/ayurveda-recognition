import numpy as np

def get_leaf_mask(image_bgr, white_thresh=245):
    is_bg = np.all(image_bgr >= white_thresh, axis=2)
    mask = (~is_bg).astype(np.uint8) * 255
    return mask
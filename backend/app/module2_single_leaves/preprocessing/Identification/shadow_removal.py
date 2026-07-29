import cv2
import numpy as np

def correct_leaf_shadow(image, leaf_mask, blur_ksize=61, max_gain=2.2):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)

    leaf_area_mask = leaf_mask > 0
    leaf_l = l[leaf_area_mask]
    if len(leaf_l) < 100:
        return image

    l_median = np.median(leaf_l)
    l_std = np.std(leaf_l)
    s_median = np.median(s[leaf_area_mask])

    dark_threshold = np.clip(l_median - 2.5 * l_std, 40, 110)
    is_dark = l < dark_threshold
    is_not_green = a > 120
    is_low_sat = s < s_median * 0.40
    shadow_pixels = is_dark & is_not_green & is_low_sat & leaf_area_mask

    shadow_area = shadow_pixels.sum()
    leaf_area = leaf_area_mask.sum()
    shadow_percent = (shadow_area / leaf_area * 100) if leaf_area > 0 else 0
    if shadow_area == 0 or shadow_percent > 15:
        return image

    k = blur_ksize | 1
    illum = cv2.GaussianBlur(l, (k, k), 0)
    illum = np.clip(illum, 1, 255)
    target_illum = np.median(illum[leaf_area_mask])
    gain = np.clip(target_illum / illum, 1.0, max_gain)

    l_corrected = np.clip(l * gain, 0, 255)
    l_final = np.where(shadow_pixels, l_corrected, l)
    lab[:, :, 0] = l_final.astype(np.uint8)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    result[~leaf_area_mask] = [255, 255, 255]
    return result

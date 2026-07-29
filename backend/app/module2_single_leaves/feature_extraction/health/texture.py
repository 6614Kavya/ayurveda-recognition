import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import entropy as scipy_entropy

def extract_texture_features(image_bgr, mask, distances=(1, 3), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1]

    # Zero-out background inside the crop so it doesn't pollute GLCM/LBP with the
    # white background; use median leaf gray value as a neutral fill instead of 0.
    gray_filled = gray.copy()
    leaf_median = int(np.median(gray[sub_mask > 0])) if np.any(sub_mask > 0) else 128
    gray_filled[sub_mask == 0] = leaf_median

    levels = 32
    gray_q = (gray_filled.astype(np.float32) / 256 * levels).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=list(distances), angles=list(angles),
                         levels=levels, symmetric=True, normed=True)

    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))

    lbp = local_binary_pattern(gray_filled, P=8, R=1, method='uniform')
    lbp_leaf = lbp[sub_mask > 0]
    hist, _ = np.histogram(lbp_leaf, bins=np.arange(0, 11), density=True)
    lbp_entropy = float(scipy_entropy(hist + 1e-8))
    lbp_uniformity = float(np.max(hist))

    # Edge density: fraction of leaf pixels that are edge pixels (Canny). Complements
    # GLCM/LBP for fine reticulated/mottled/net-like patterns (e.g. interveinal necrosis
    # following the vein structure) -- smooth healthy tissue has very few edges; a
    # marbled necrotic pattern has many fine ones. Different signal from GLCM's
    # co-occurrence statistics, not a duplicate of it.
    edges = cv2.Canny(gray_filled, threshold1=50, threshold2=150)
    leaf_pixel_count = int(np.sum(sub_mask > 0))
    edge_density = float(np.sum((edges > 0) & (sub_mask > 0)) / leaf_pixel_count) if leaf_pixel_count > 0 else 0.0

    return {
        'glcm_contrast': contrast,
        'glcm_homogeneity': homogeneity,
        'glcm_energy': energy,
        'glcm_correlation': correlation,
        'lbp_entropy': lbp_entropy,
        'lbp_uniformity': lbp_uniformity,
        'edge_density': edge_density,
    }

 
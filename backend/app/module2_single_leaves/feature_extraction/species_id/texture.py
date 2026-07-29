import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor

# 1. GLCM — Texture Features 
def extract_glcm(gray_img):
    distances  = [1, 2, 3, 4]
    angles     = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ['contrast', 'dissimilarity', 'homogeneity',
                  'energy', 'correlation', 'ASM']
    glcm = graycomatrix(
        gray_img, distances=distances, angles=angles,
        levels=256, symmetric=True, normed=True)
    features = {}
    for prop in properties:
        features[f'glcm_{prop}'] = graycoprops(glcm, prop).mean()
    return features  # 6 values


# 2. Gabor — Vein Frequency Features 
def extract_gabor(gray_img):
    frequencies  = [0.1, 0.2, 0.3, 0.4, 0.5]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    img_float    = gray_img.astype(np.float32) / 255.0
    features     = {}
    for fi, freq in enumerate(frequencies):
        for ti, theta in enumerate(orientations):
            real, _ = gabor(img_float, frequency=freq, theta=theta)
            features[f'gabor_f{fi}_t{ti}_mean'] = real.mean()
            features[f'gabor_f{fi}_t{ti}_std']  = real.std()
    return features  # 40 values

# 7. LBP — Local Binary Pattern 
def extract_lbp(gray_img):
    radius   = 3
    n_points = 8 * radius
    lbp      = local_binary_pattern(
        gray_img, n_points, radius, method='uniform')
    hist, _  = np.histogram(
        lbp.ravel(),
        bins=n_points + 2,
        range=(0, n_points + 2))
    hist     = hist.astype(float)
    hist    /= (hist.sum() + 1e-7)  # normalize
    return {f'lbp_{i}': v for i, v in enumerate(hist)}  # 26 values

def extract_vein_edge_density(gray_img, mask):
    edges = cv2.Canny(gray_img, 40, 120)
    leaf_px = mask > 0
    total = leaf_px.sum()
    if total == 0:
        return {'vein_edge_density': 0.0}
    density = (edges[leaf_px] > 0).sum() / total * 100
    return {'vein_edge_density': density}

def extract_surface_relief_features(image_bgr, leaf_mask, erode_px=15):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
    interior_mask = cv2.erode(leaf_mask, kernel)
    leaf_pixels = interior_mask > 0
    if leaf_pixels.sum() < 50:
        leaf_pixels = leaf_mask > 0

    fine_blur   = cv2.GaussianBlur(gray, (3, 3), 0)
    coarse_blur = cv2.GaussianBlur(gray, (9, 9), 0)
    lap_fine   = cv2.Laplacian(fine_blur, cv2.CV_64F, ksize=3)
    lap_coarse = cv2.Laplacian(coarse_blur, cv2.CV_64F, ksize=5)
    relief_fine   = float(np.var(lap_fine[leaf_pixels]))
    relief_coarse = float(np.var(lap_coarse[leaf_pixels]))
    relief_ratio  = relief_fine / (relief_coarse + 1e-6)

    return {
        'relief_laplacian_var_fine': relief_fine,
        'relief_laplacian_var_coarse': relief_coarse,
        'relief_ratio': relief_ratio,
    }

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor

# ── 1. GLCM — Texture Features ────────────────────────────
def extract_glcm(gray_img):
    """
    Gray-Level Co-occurrence Matrix (GLCM).
    Captures HOW PIXELS RELATE to their neighbours.
    Why for leaves:
    → Gotukola has rough bumpy venation → high contrast GLCM
    → Long narrow leaf has smooth parallel texture → low contrast
    → Different species have measurably different texture values
    Returns 6 values: contrast, dissimilarity, homogeneity,
                      energy, correlation, ASM
    """
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


# ── 2. Gabor — Vein Frequency Features ────────────────────
def extract_gabor(gray_img):
    """
    Gabor filter bank responses.
    Captures TEXTURE at different FREQUENCIES and ORIENTATIONS.
    Why for leaves:
    → Long narrow leaves have veins running vertically
      → Gabor captures horizontal frequency response
    → Gotukola has fan-shaped radiating veins
      → Gabor captures multiple direction responses
    → Different vein patterns = different Gabor values
    Returns 40 values: 5 freq × 4 orientations × (mean + std)
    """
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

# ── 7. LBP — Local Binary Pattern ─────────────────────────
def extract_lbp(gray_img):
    """
    Local Binary Pattern (LBP) — local surface texture.
    Captures micro-texture patterns around each pixel.
    Why for leaves:
    → Gotukola has very rough bumpy surface texture
    → Long narrow leaves have smooth parallel texture
    → Dark oval leaf has very smooth waxy texture
    → LBP captures these micro-texture differences ✅
    → Very useful for morphologically similar leaves
       that differ in surface texture not shape
    Returns 26 histogram values.
    """
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
    """
    Canny-edge density within the leaf, as a direct vein-visibility
    proxy -- distinct from GLCM/LBP (which measure general texture
    statistics, not specifically linear vein structure).

    Why: measured directly -- e.g. Kora kaha (smooth, glossy) at 11.6
    vs Walikaha (visibly veined/textured) at 34.1 -- one of the largest
    gaps found in this whole analysis. Also separates Diya NA (18.5)
    from Na (28.5) clearly.
    """
    edges = cv2.Canny(gray_img, 40, 120)
    leaf_px = mask > 0
    total = leaf_px.sum()
    if total == 0:
        return {'vein_edge_density': 0.0}
    density = (edges[leaf_px] > 0).sum() / total * 100
    return {'vein_edge_density': density}

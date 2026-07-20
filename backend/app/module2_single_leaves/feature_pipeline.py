from feature_extraction.species_id.texture import (
    extract_glcm, extract_gabor, extract_lbp, extract_vein_edge_density
)

from feature_extraction.species_id.shape import (
    extract_hog, extract_hu_moments, extract_contour_shape,
    get_leaf_mask_and_contour, extract_notch_features,
    extract_margin_features, extract_principal_axis_features
)

from feature_extraction.species_id.colour import extract_hsv_color

# ── Master function — extract ALL features ─────────────────
def extract_all_features(image_bgr):
    """
    Extract all 105 features from one image (97 generic + 8 handcrafted).
    Input : BGR image array (preprocessed, 224×224)
    Output: dict with 105 feature key-value pairs
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    leaf_mask = (gray < 250).astype(np.uint8)
    coords = cv2.findNonZero(leaf_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        gray_for_texture = gray[y:y+h, x:x+w]
    else:
        gray_for_texture = gray

    features = {}
    features.update(extract_glcm(gray_for_texture))            #  6 features
    features.update(extract_gabor(gray_for_texture))            # 40 features
    features.update(extract_hog(gray_for_texture))              #  5 features
    features.update(extract_hu_moments(gray))       #  7 features
    features.update(extract_hsv_color(image_bgr))   #  6 features
    features.update(extract_contour_shape(gray))    #  7 features
    features.update(extract_lbp(gray_for_texture))              # 26 features

    # ── Handcrafted (botanically-informed) features ────────
    mask, cnt = get_leaf_mask_and_contour(image_bgr)
    if cnt is not None:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        leaf_length = max(w, h)

        features.update(extract_notch_features(cnt, leaf_length))   # 2 features
        features.update(extract_margin_features(cnt, area, peri))   # 2 features
        features.update(extract_principal_axis_features(cnt))       # 3 features
        features.update(extract_vein_edge_density(gray, mask))      # 1 feature
    else:
        # Segmentation failed on this image (e.g. bad crop/coverage) —
        # fall back to neutral defaults so the row isn't dropped entirely
        features.update({
            'notch_depth': 0.0, 'notch_angle': 180.0,
            'margin_roughness': 0.0, 'serration_count': 0,
            'principal_length': 0.0, 'principal_width': 0.0,
            'principal_aspect_ratio': 0.0, 'vein_edge_density': 0.0,
        })

    return features  # 105 total


print('✅ All feature extraction functions defined!')
print()
print('   Feature Group      Features   What it captures')
print('   ────────────────────────────────────────────────────')
print('   GLCM                  6        Vein texture patterns')
print('   Gabor                40        Vein frequency & direction')
print('   HOG                   5        Leaf shape & edges')
print('   Hu-moments            7        Overall leaf shape')
print('   HSV Color             6        Leaf colour shade')
print('   Contour Shape         7        Geometric properties')
print('   LBP                  26        Local surface texture')
print('   Handcrafted (notch,')
print('   margin, axis, vein)   8        Botanically-targeted features')
print('   ────────────────────────────────────────────────────')
print('   TOTAL                105       features per image')
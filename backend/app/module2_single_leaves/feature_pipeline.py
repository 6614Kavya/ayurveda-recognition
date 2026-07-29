import cv2
import numpy as np

from app.module2_single_leaves.feature_extraction.species_id.texture import (
    extract_glcm, extract_gabor, extract_lbp, extract_vein_edge_density,
    extract_surface_relief_features
)

from app.module2_single_leaves.feature_extraction.species_id.shape import (
    extract_hog, extract_hu_moments, extract_contour_shape,
    get_leaf_mask_and_contour, extract_notch_features,
    extract_margin_features, extract_principal_axis_features,
)

from app.module2_single_leaves.feature_extraction.species_id.colour import extract_hsv_color

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
    features.update(extract_glcm(gray_for_texture))            
    features.update(extract_gabor(gray_for_texture))             
    features.update(extract_hog(gray_for_texture))  
    features.update(extract_hu_moments(gray))       
    features.update(extract_hsv_color(image_bgr))   
    features.update(extract_contour_shape(gray))    
    features.update(extract_lbp(gray_for_texture))          

    # ── Handcrafted (botanically-informed) features ────────
    mask, cnt = get_leaf_mask_and_contour(image_bgr)
    if cnt is not None:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        leaf_length = max(w, h)

        features.update(extract_notch_features(cnt, leaf_length))   
        features.update(extract_margin_features(cnt, area, peri))   
        features.update(extract_principal_axis_features(cnt))       
        features.update(extract_vein_edge_density(gray, mask))  
        features.update(extract_surface_relief_features(image_bgr, mask))     
    else:
        features.update({
            'notch_depth': 0.0, 'notch_angle': 180.0,
            'margin_roughness': 0.0, 'serration_count': 0,
            'principal_length': 0.0, 'principal_width': 0.0,
            'principal_aspect_ratio': 0.0, 'vein_edge_density': 0.0,
        })

    return features
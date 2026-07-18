import numpy as np

from .features import (
    extract_color_features,
    extract_glcm_features,
    extract_lbp_features,
    extract_gabor_features,
    extract_shape_features,
    extract_petal_proxy,
    extract_petal_morphometrics,
    extract_filament_core_features,
    extract_petal_overlap_features,
    extract_vein_center_features,
)

FEATURE_BREAKDOWN = {
    'color'          : 105,
    'glcm'           : 10,
    'lbp'            : 26,
    'gabor'          : 24,
    'shape'          : 12,
    'petal_proxy'    : 4,
    'petal_morph'    : 10,
    'filament_core'  : 9,
    'petal_overlap'  : 5,
    'vein_center'    : 5,
}
TOTAL_FEATURES = sum(FEATURE_BREAKDOWN.values())  # 210


def extract_all_features(roi: dict) -> np.ndarray:

    color = extract_color_features(roi['roi_rgb'], roi['roi_mask'])
    glcm = extract_glcm_features(roi['roi_gray'], roi['roi_mask'])
    lbp = extract_lbp_features(roi['roi_gray'], roi['roi_mask'])
    gabor = extract_gabor_features(roi['roi_gray'], roi['roi_mask'])
    shape = extract_shape_features(roi['roi_mask'])
    petal = extract_petal_proxy(roi['roi_mask'])
    petal_morph = extract_petal_morphometrics(roi['roi_mask'])
    filament_core = extract_filament_core_features(roi['roi_rgb'], roi['roi_gray'], roi['roi_mask'])
    petal_overlap = extract_petal_overlap_features(roi['roi_mask'])
    vein_center = extract_vein_center_features(roi['roi_rgb'], roi['roi_gray'], roi['roi_mask'])

    feature_vec = np.concatenate([
        color, glcm, lbp, gabor, shape, petal,
        petal_morph, filament_core, petal_overlap, vein_center
    ])

    assert feature_vec.shape[0] == TOTAL_FEATURES, (
        f"Feature vector length {feature_vec.shape[0]} != expected {TOTAL_FEATURES}. "
        f"Check FEATURE_BREAKDOWN in feature_extractor.py against config.py / features/*.py."
    )

    feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vec

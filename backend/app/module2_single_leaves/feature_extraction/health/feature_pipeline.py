import numpy as np
from app.module2_single_leaves.feature_extraction.health.mask_utils import get_leaf_mask
from app.module2_single_leaves.feature_extraction.health.colour import extract_color_features
from app.module2_single_leaves.feature_extraction.health.spots import extract_spot_features
from app.module2_single_leaves.feature_extraction.health.shape import extract_shape_features
from app.module2_single_leaves.feature_extraction.health.texture import extract_texture_features

def extract_all_features(image_bgr):
    """Returns None if the mask is degenerate (empty/near-empty leaf area)."""
    mask = get_leaf_mask(image_bgr)
    if np.sum(mask > 0) < 200:  # sanity floor -- near-empty mask means a bad/corrupt image
        return None

    feats = {}
    for extractor in (
        lambda: extract_color_features(image_bgr, mask),
        lambda: extract_spot_features(image_bgr, mask),
        lambda: extract_shape_features(mask),
        lambda: extract_texture_features(image_bgr, mask),
    ):
        result = extractor()
        if result is None:
            return None
        feats.update(result)
    return feats

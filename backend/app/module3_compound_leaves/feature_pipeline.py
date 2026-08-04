import cv2
import numpy as np
from typing import Optional

from app.module3_compound_leaves.preprocessing.shared.resize      import letterbox_resize
from app.module3_compound_leaves.preprocessing.shared.masking     import qc_check
from app.module3_compound_leaves.preprocessing.shared.mask_guard  import select_mask_guarded
from app.module3_compound_leaves.preprocessing.species_id.enhance import enhance_for_species_id
from app.module3_compound_leaves.feature_extraction.species_id.shape      import extract_shape_features
from app.module3_compound_leaves.feature_extraction.species_id.colour     import extract_colour_features
from app.module3_compound_leaves.feature_extraction.species_id.texture    import extract_texture_features
from app.module3_compound_leaves.feature_extraction.species_id.vein       import extract_vein_features
from app.module3_compound_leaves.feature_extraction.species_id.whole_leaf import extract_whole_leaf_features
from app.module3_compound_leaves.preprocessing.config import TARGET_LONG


def _namespace_features(prefix: str, features: dict) -> dict:
    namespaced = {}
    for key, value in features.items():
        if key.startswith(f"{prefix}_"):
            namespaced[key] = value
        else:
            namespaced[f"{prefix}_{key}"] = value
    return namespaced


def extract_features(img_bgr: np.ndarray) -> tuple[Optional[dict], dict]:
   
    if img_bgr is None:
        return None, {"qc_passed": False, "qc_reason": "empty image"}

    # Step 1: letterbox resize — identical to training (TARGET_LONG, white pad)
    img_resized, _resize_meta = letterbox_resize(img_bgr, TARGET_LONG)

    # Step 2: guarded background removal (baseline vs illumination-flattened,
    # whichever has lower shadow-bleed without an 8%+ foreground area drop)
    mask_final, mask_choice, mask_diag = select_mask_guarded(img_resized)

    # QC check — same thresholds as training (2%–75% coverage)
    qc_passed, qc_reason = qc_check(mask_diag)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask_final)

    info = {
        "img_resized" : img_resized,
        "mask_final"  : mask_final,
        "mask_choice" : mask_choice,
        "coverage_pct": mask_diag.get("coverage_pct"),
        "img_masked"  : img_masked,
        "img_sharp"   : None,
        "qc_passed"   : qc_passed,
        "qc_reason"   : qc_reason,
    }

    if not qc_passed:
        return None, info

    # Step 3: enhancement (species-ID branch only — never used for health features)
    img_sharp = enhance_for_species_id(img_masked, mask_final)
    info["img_sharp"] = img_sharp

    # Step 4: feature extraction — same five groups, same source images per group
    shape_f   = extract_shape_features(mask_final)
    colour_f  = extract_colour_features(img_resized, mask_final)   # RAW resized image
    texture_f = extract_texture_features(img_sharp, mask_final)    # enhanced image
    vein_f, skel, _ = extract_vein_features(img_sharp, mask_final) # enhanced image
    whole_f   = extract_whole_leaf_features(mask_final)
    info["vein_skel"] = skel

    feats = {}
    feats.update(_namespace_features("shape", shape_f))
    feats.update(_namespace_features("colour", colour_f))
    feats.update(_namespace_features("texture", texture_f))
    feats.update(_namespace_features("vein", vein_f))
    feats.update(_namespace_features("whole", whole_f))

    return feats, info


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw uploaded bytes into a BGR uint8 array (or None if invalid)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

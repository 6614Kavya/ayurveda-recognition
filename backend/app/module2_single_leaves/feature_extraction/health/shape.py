import cv2
import numpy as np

def extract_shape_features(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    # Outer contour = the largest contour with no parent (parent == -1)
    outer_idxs = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not outer_idxs:
        return None
    outer_idx = max(outer_idxs, key=lambda i: cv2.contourArea(contours[i]))
    outer_contour = contours[outer_idx]

    leaf_area = cv2.contourArea(outer_contour)
    if leaf_area <= 0:
        return None

    # --- Internal holes: contours whose parent IS the outer contour ---
    hole_areas = [cv2.contourArea(contours[i]) for i, h in enumerate(hierarchy) if h[3] == outer_idx]
    hole_areas = [a for a in hole_areas if a > 0]
    hole_count = len(hole_areas)
    hole_area_ratio = sum(hole_areas) / leaf_area if hole_areas else 0.0

    # --- Convex hull deficit: margin chewing makes actual area << hull area ---
    hull = cv2.convexHull(outer_contour)
    hull_area = cv2.contourArea(hull)
    solidity = leaf_area / hull_area if hull_area > 0 else 1.0
    hull_deficit_ratio = 1.0 - solidity  # 0 = perfectly convex, higher = more margin loss

    # --- Perimeter irregularity ---
    perimeter = cv2.arcLength(outer_contour, True)
    perim_area_ratio = (perimeter ** 2) / leaf_area if leaf_area > 0 else 0.0

    # --- Convexity defects: count/depth of notches along the margin ---
    defect_count, defect_mean_depth, defect_max_depth = 0, 0.0, 0.0
    try:
        hull_idx = cv2.convexHull(outer_contour, returnPoints=False)
        if hull_idx is not None and len(hull_idx) > 3:
            defects = cv2.convexityDefects(outer_contour, hull_idx)
            if defects is not None:
                defects = defects.reshape(-1, 4)  # inconsistent shape across builds -- always reshape
                depths = defects[:, 3] / 256.0     # depth is stored in fixed-point, divide by 256
                sig_depths = depths[depths > 2.0]  # ignore tiny numerical-noise defects
                defect_count = len(sig_depths)
                defect_mean_depth = float(np.mean(sig_depths)) if len(sig_depths) else 0.0
                defect_max_depth = float(np.max(sig_depths)) if len(sig_depths) else 0.0
    except cv2.error:
        pass  # degenerate contour (e.g. heavily eaten leaf) -- leave defect features at 0

    # --- Edge smoothness deficit: fine wavy/rippled boundary WITHOUT missing tissue
    # (e.g. wilting-curl), distinct from hull_deficit_ratio which measures big missing
    # chunks (chewing). A morphological opening removes small-scale bumps/ripples while
    # leaving large notches mostly intact -- so the perimeter LOST by opening isolates
    # fine-scale waviness specifically, without conflating it with chewing damage.
    edge_smoothness_deficit = 0.0
    try:
        leaf_width_estimate = np.sqrt(leaf_area)
        kernel_size = max(3, int(leaf_width_estimate * 0.03) | 1)  # odd, ~3% of leaf width
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        opened_contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if opened_contours:
            opened_outer = max(opened_contours, key=cv2.contourArea)
            opened_perimeter = cv2.arcLength(opened_outer, True)
            if opened_perimeter > 0:
                edge_smoothness_deficit = float(max(0.0, (perimeter - opened_perimeter) / opened_perimeter))
    except cv2.error:
        pass  # degenerate mask -- leave at 0

    return {
        'solidity': float(solidity),
        'hull_deficit_ratio': float(hull_deficit_ratio),
        'hole_count': int(hole_count),
        'hole_area_ratio': float(hole_area_ratio),
        'perim_area_ratio': float(perim_area_ratio),
        'edge_defect_count': int(defect_count),
        'edge_defect_mean_depth': float(defect_mean_depth),
        'edge_defect_max_depth': float(defect_max_depth),
        'edge_smoothness_deficit': edge_smoothness_deficit,
    }

print('✅ extract_shape_features() ready')

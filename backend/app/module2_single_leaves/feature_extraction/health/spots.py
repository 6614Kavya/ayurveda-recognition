
import cv2
import numpy as np

def extract_spot_features(image_bgr, mask, min_spot_area=8):
    leaf_area = np.sum(mask > 0)
    if leaf_area == 0:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    leaf_v = v[mask > 0]
    median_v = np.median(leaf_v)

    # Dark spots: notably darker than the leaf's own median brightness (adapts per-leaf,
    # rather than a fixed global threshold which wouldn't generalize across species/lighting)
    dark_spot_mask = ((v.astype(np.int16) < int(median_v) - 40) & (mask > 0)).astype(np.uint8) * 255
    # Pale/white spots: notably desaturated relative to leaf tissue (scale insect patches,
    # skeletonized areas both lose color saturation even where they aren't pure white)
    pale_spot_mask = ((s < 40) & (v.astype(np.int16) > int(median_v) - 20) & (mask > 0)).astype(np.uint8) * 255

    def blob_stats(spot_mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(spot_mask, connectivity=8)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_spot_area]
        count = len(areas)
        total_area = sum(areas)
        return count, total_area, (np.mean(areas) if areas else 0.0), (max(areas) if areas else 0.0)

    dark_count, dark_area, dark_mean_area, dark_max_area = blob_stats(dark_spot_mask)
    pale_count, pale_area, pale_mean_area, pale_max_area = blob_stats(pale_spot_mask)

    return {
        'dark_spot_count': dark_count,
        'dark_spot_area_ratio': dark_area / leaf_area,
        'dark_spot_mean_area': dark_mean_area,
        'dark_spot_max_area_ratio': dark_max_area / leaf_area,
        'pale_spot_count': pale_count,
        'pale_spot_area_ratio': pale_area / leaf_area,
        'pale_spot_mean_area': pale_mean_area,
        'pale_spot_max_area_ratio': pale_max_area / leaf_area,
    }


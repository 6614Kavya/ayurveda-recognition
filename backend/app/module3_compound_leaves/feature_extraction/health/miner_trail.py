"""
feature_extraction/health/miner_trail.py

Leaf-miner trail detection -- new feature group (not previously covered by
boundary/hole/colour/scar). Leaf-miner larvae tunnel BETWEEN the leaf's
upper and lower epidermis, leaving a thin, winding, pale/tan trail that is
geometrically distinct from every other damage type currently detected:

  - not a hole (tissue is intact, just discoloured -- light passes through
    but there's no gap)
  - not boundary damage (interior to the leaf, doesn't touch the margin)
  - not scar tissue (not localized to a wound-margin band)
  - not a diffuse chlorotic patch (thin and highly tortuous, not a broad
    uniform region)

MUST run on the un-enhanced masked_raw image, same hard rule as every
other health feature group.

Detection approach: colour-gate candidate pale/tan interior pixels ->
connected components -> skeletonize each component -> reject anything too
short or not winding enough (round pale patches / light chlorotic blotches
have low tortuosity; genuine mine trails don't).

Thresholds here are a first pass, same caveat as colour_health.py:
recalibrate against a handful of manually-confirmed miner-trail crops
(you already have at least two clean examples: the small leaflet in
beli__high__image_06, and the serpentine trail closeup) before citing
exact numbers in the dissertation.
"""
import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

SENTINEL = -1.0
MARGIN_EXCLUDE_PX = 6          # erode this much off the outer edge -- trails are interior
MIN_CANDIDATE_AREA_PX = 15     # sub-pixel-noise floor for a candidate blob
MIN_SKELETON_LEN_PX = 4
MIN_TORTUOSITY = 1.3           # skeleton_length / straight_line_endpoint_distance


def _skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    """Pixels on the skeleton with exactly one 8-connected skeleton
    neighbour. Used to measure straight-line span cheaply (O(n) instead of
    all-pairs) -- good enough for the mostly-linear trails we're looking
    for; branch points just mean we pick up the longest span among several
    endpoint pairs, which is fine for a tortuosity estimate."""
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    conv = convolve(skel.astype(np.uint8), kernel, mode="constant")
    return np.argwhere(skel & (conv == 11))


def extract_miner_trail_features(img_bgr: np.ndarray, mask_final: np.ndarray) -> dict:
    """
    Parameters
    ----------
    img_bgr : masked_raw image, un-enhanced.
    mask_final : binary leaf mask, foreground = 255.

    Returns
    -------
    dict of miner_trail_* features.
    """
    fg = mask_final.astype(np.uint8)
    leaf_area = int(np.count_nonzero(fg))
    if leaf_area == 0:
        return {
            "miner_trail_length_norm": SENTINEL,
            "miner_trail_coverage_pct": SENTINEL,
            "miner_trail_mean_tortuosity": SENTINEL,
            "miner_trail_count": 0,
        }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGIN_EXCLUDE_PX + 1,) * 2)
    interior = cv2.erode(fg, kernel).astype(bool)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, S = lab[..., 0], hsv[..., 1]

    # pale/tan interior tissue: lighter and less saturated than surrounding
    # healthy green, but not blown-out (specular) or fully bleached (pale
    # patch, which is broader/rounder and handled by colour_health.py) --
    # this is a mid-band gate deliberately narrower than colour_health's
    # `pale` category, since trails are visually a lighter shade *within*
    # otherwise-green tissue rather than a fully bleached patch.
    candidate = (L > 140) & (L < 220) & (S < 90) & interior
    candidate_u8 = (candidate.astype(np.uint8)) * 255
    candidate_u8 = cv2.morphologyEx(
        candidate_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)

    total_length = 0.0
    total_area = 0
    tortuosities = []
    trail_count = 0

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_CANDIDATE_AREA_PX:
            continue

        blob = labels == i
        skel = skeletonize(blob)
        skel_len = int(np.count_nonzero(skel))
        if skel_len < MIN_SKELETON_LEN_PX:
            continue

        endpoints = _skeleton_endpoints(skel)
        if len(endpoints) < 2:
            # closed loop or single blob with no clear endpoint -- use
            # bounding-box diagonal as a fallback straight-line estimate
            ys, xs = np.nonzero(skel)
            straight_dist = float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min()))
        else:
            d = np.sqrt(((endpoints[:, None, :] - endpoints[None, :, :]) ** 2).sum(-1))
            straight_dist = float(d.max())

        if straight_dist < 1e-6:
            continue

        tortuosity = skel_len / straight_dist
        if tortuosity < MIN_TORTUOSITY:
            # too straight to be a mine trail -- likely a light patch,
            # vein-adjacent highlight, or scratch artifact
            continue

        total_length += skel_len
        total_area += area
        tortuosities.append(tortuosity)
        trail_count += 1

    return {
        "miner_trail_length_norm": float(total_length / np.sqrt(leaf_area)),
        "miner_trail_coverage_pct": float(total_area / leaf_area * 100.0),
        "miner_trail_mean_tortuosity": float(np.mean(tortuosities)) if tortuosities else 0.0,
        "miner_trail_count": trail_count,
    }
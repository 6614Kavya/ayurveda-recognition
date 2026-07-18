
import cv2
import numpy as np
import networkx as nx
from scipy.signal import find_peaks, savgol_filter, peak_widths
from skimage.morphology import skeletonize



# STANDARD descriptor set 


def extract_shape_features(leaf_mask: np.ndarray) -> dict:
   
    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return _empty_shape_features()

    cnt   = max(cnts, key=cv2.contourArea)
    area  = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, closed=True))
    x, y, w, h = cv2.boundingRect(cnt)

    # ── Basic ratios ──────────────────────────────────────────────────────
    aspect_ratio = float(w) / h    if h > 0       else 0.0
    compactness  = area / (w * h)  if (w * h) > 0 else 0.0
    circularity  = (4.0 * np.pi * area / perim ** 2) if perim > 0 else 0.0

    # ── Convex hull ───────────────────────────────────────────────────────
    hull       = cv2.convexHull(cnt)
    hull_area  = float(cv2.contourArea(hull))
    hull_perim = float(cv2.arcLength(hull, closed=True))

    # solidity  — how well the contour FILLS its convex hull (area ratio)
    solidity   = area / hull_area        if hull_area  > 0 else 0.0

    # convexity — how SMOOTH the boundary is relative to its convex hull
    #             (perimeter ratio).  FIXED: was identical to solidity before.
    #             hull_perim <= perim always, so convexity is in (0, 1].
    convexity  = hull_perim / perim      if perim     > 0 else 0.0

    # ── Ellipse fit ───────────────────────────────────────────────────────
    if len(cnt) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
        elongation = float(mi / ma) if ma > 0 else 0.0
    else:
        elongation = 0.0

    # ── Hu moments from binary MASK (not contour) ─────────────────────────
    # cv2.moments on the full mask image integrates over all foreground
    # pixels — shadow pixels at the edge produce negligible influence.
    M      = cv2.moments(leaf_mask.astype(np.uint8))
    hu     = cv2.HuMoments(M).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    feats = {
        "aspect_ratio": aspect_ratio,
        "circularity" : circularity,
        "solidity"    : solidity,
        "convexity"   : convexity,      # now hull_perim / perim, not area / hull_area
        "compactness" : compactness,
        "elongation"  : elongation,
    }
    for i, val in enumerate(hu_log):
        feats[f"hu_{i+1}"] = float(val)

    # ── NEW: botanical sinus/apex features (see module docstring) ─────────
    # Wrapped in try/except so a degenerate mask (e.g. QC-borderline image)
    # falls back to sentinel values instead of crashing the whole pipeline
    # — consistent with whole_leaf.py's existing -1.0 sentinel convention.
    try:
        feats.update(_extract_botanical_shape_features(leaf_mask))
    except Exception:
        feats.update(_empty_botanical_shape_features())

    return feats


def _empty_shape_features() -> dict:
    feats = {k: 0.0 for k in [
        "aspect_ratio", "circularity", "solidity", "convexity",
        "compactness", "elongation",
    ]}
    for i in range(1, 8):
        feats[f"hu_{i}"] = 0.0
    feats.update(_empty_botanical_shape_features())
    return feats


# BOTANICAL / HANDCRAFTED additions


_BOTANICAL_SENTINEL = -1.0   # same convention as whole_leaf.py's NaN sentinel


def _empty_botanical_shape_features() -> dict:
    return {
        "botanical_apex_curvature_median":  _BOTANICAL_SENTINEL,
        "botanical_apex_retuse_fraction":   _BOTANICAL_SENTINEL,
        "botanical_margin_serration_freq":  _BOTANICAL_SENTINEL,
        "botanical_margin_serration_amp":   _BOTANICAL_SENTINEL,
        "botanical_margin_tooth_sharpness": _BOTANICAL_SENTINEL,
        "botanical_pair_offset_median":     _BOTANICAL_SENTINEL,
        "botanical_arc_elongation_median":  _BOTANICAL_SENTINEL,
        "botanical_arc_elongation_iqr":     _BOTANICAL_SENTINEL,
        "botanical_sinus_prominence_median": _BOTANICAL_SENTINEL,
        "botanical_sinus_prominence_iqr":   _BOTANICAL_SENTINEL,
    }


def _get_rachis_axis(mask: np.ndarray, n_samples: int = 300):
    """
    Rachis axis as the LONGEST PATH in the skeleton graph.

    Naive alternatives that were tried and rejected:
    - straight PCA line: fails on the several species with visibly curved
      rachises (Kattakumanjal, Kalawal, Ranawara, Siymbala).
    - ordering ALL skeleton pixels by PCA projection: lets leaflet-tip
      skeleton spurs jump into the ordering, zigzagging the "axis" through
      leaflet tips. Confirmed on a real test image (25 spurious apices
      detected on a ~19-leaflet leaf) before this fix.

    Longest-path-in-graph is the standard fix: the rachis is the longest
    branch in the skeleton tree by construction (every leaflet spur is
    shorter than the stem it hangs off), so a graph diameter search
    reliably isolates it.
    """
    skel = skeletonize(mask > 0)
    ys, xs = np.nonzero(skel)
    if len(xs) < 10:
        return None

    coords = set(zip(xs.tolist(), ys.tolist()))
    G = nx.Graph()
    G.add_nodes_from(coords)
    for (x, y) in coords:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb in coords:
                    G.add_edge((x, y), nb, weight=np.hypot(dx, dy))

    if G.number_of_nodes() == 0:
        return None

    # Double-BFS/Dijkstra longest-path-in-tree trick: farthest node from an
    # arbitrary start, then farthest node from THAT node, gives the two
    # endpoints of the longest path in the graph.
    start = next(iter(G.nodes))
    lengths = nx.single_source_dijkstra_path_length(G, start, weight="weight")
    node_a = max(lengths, key=lengths.get)
    lengths_a = nx.single_source_dijkstra_path_length(G, node_a, weight="weight")
    node_b = max(lengths_a, key=lengths_a.get)
    path = nx.shortest_path(G, node_a, node_b, weight="weight")

    pts = np.array(path, dtype=np.float32)
    cum_dist = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    if cum_dist[-1] < 1e-6:
        return None

    t_new = np.linspace(0, cum_dist[-1], n_samples)
    x_new = np.interp(t_new, cum_dist, pts[:, 0])
    y_new = np.interp(t_new, cum_dist, pts[:, 1])
    return np.stack([x_new, y_new], axis=1)


def _compute_side_profiles(mask: np.ndarray, axis_pts: np.ndarray, half_win=None):
   
    h, w = mask.shape
    if half_win is None:
        ys, xs = np.nonzero(mask)
        bbox_diag = np.hypot(xs.max() - xs.min(), ys.max() - ys.min()) if len(xs) else max(h, w)
        half_win = max(15, int(0.35 * bbox_diag))

    n = len(axis_pts)
    right_w = np.zeros(n)
    left_w = np.zeros(n)
    right_pts = np.zeros((n, 2))
    left_pts = np.zeros((n, 2))

    for i in range(n):
        i0, i1 = max(0, i - 2), min(n - 1, i + 2)
        tangent = axis_pts[i1] - axis_pts[i0]
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            right_pts[i], left_pts[i] = axis_pts[i], axis_pts[i]
            continue
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]])
        cx, cy = axis_pts[i]

        d_pos = 0
        for d in range(1, half_win):
            px, py = int(cx + normal[0] * d), int(cy + normal[1] * d)
            if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
                d_pos = d
            else:
                break
        d_neg = 0
        for d in range(1, half_win):
            px, py = int(cx - normal[0] * d), int(cy - normal[1] * d)
            if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
                d_neg = d
            else:
                break

        right_w[i], left_w[i] = d_pos, d_neg
        right_pts[i] = [cx + normal[0] * d_pos, cy + normal[1] * d_pos]
        left_pts[i]  = [cx - normal[0] * d_neg, cy - normal[1] * d_neg]

    return right_w, left_w, right_pts, left_pts


def _smooth(arr: np.ndarray, win_cap: int = 15) -> np.ndarray:
    win = min(win_cap, (len(arr) // 2) * 2 - 1)
    win = max(win, 5)
    if win >= len(arr):
        win = len(arr) - 1 if len(arr) % 2 == 0 else len(arr)
        win = max(win, 3)
    if win < 3:
        return arr.copy()
    return savgol_filter(arr, window_length=win, polyorder=2)


def _polyline_curvature(pts: np.ndarray, window: int) -> np.ndarray:
   
    n = len(pts)
    curv = np.zeros(n)
    for i in range(window, n - window):
        v1 = pts[i] - pts[i - window]
        v2 = pts[i + window] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angle = np.arccos(cos_ang)
        sign = np.sign(np.cross(v1, v2))
        curv[i] = sign * angle
    return curv


def _peak_sharpness_stats(curv_1d: np.ndarray):
    
    absc = np.abs(curv_1d)
    if absc.max() < 1e-9:
        return np.array([]), np.array([])
    peaks, props = find_peaks(absc, prominence=0.05 * absc.max())
    if len(peaks) < 3:
        return np.array([]), np.array([])
    widths = peak_widths(absc, peaks, rel_height=0.5)[0]
    heights = props["prominences"]
    spacing = float(np.median(np.diff(peaks)))
    if spacing < 1e-6:
        return np.array([]), np.array([])
    rel_width = widths / spacing
    valid = rel_width > 1e-6
    return heights[valid], rel_width[valid]


def _extract_botanical_shape_features(mask: np.ndarray) -> dict:
    mask_bin = (mask > 0).astype(np.uint8)
    n_samples = 300

    axis_pts = _get_rachis_axis(mask_bin, n_samples)
    if axis_pts is None:
        return _empty_botanical_shape_features()

    right_w, left_w, right_pts, left_pts = _compute_side_profiles(mask_bin, axis_pts)
    combined = right_w + left_w
    combined_smooth = _smooth(combined)

    # ── Apex / sinus detection on the COMBINED (leaflet-pair) profile ─────
    # LOW prominence floor deliberately, so shallow waists from touching
    # leaflets are still found (with low prominence) rather than missed
    # entirely — see module docstring for the validated behaviour.
    rng = combined_smooth.max() - combined_smooth.min() + 1e-6
    min_prom = 0.03 * rng
    min_dist = max(3, int(0.02 * n_samples))
    apex_idx, apex_props = find_peaks(combined_smooth, prominence=min_prom, distance=min_dist)
    sinus_idx, sinus_props = find_peaks(-combined_smooth, prominence=min_prom, distance=min_dist)

    feats = {}

    # ── Sinus prominence: packing/spacing proxy (Ranawara vs Siymbala) ────
    if len(sinus_idx) > 0:
        prom = sinus_props["prominences"]
        feats["botanical_sinus_prominence_median"] = float(np.median(prom))
        feats["botanical_sinus_prominence_iqr"] = float(
            np.percentile(prom, 75) - np.percentile(prom, 25))
    else:
        feats["botanical_sinus_prominence_median"] = _BOTANICAL_SENTINEL
        feats["botanical_sinus_prominence_iqr"] = _BOTANICAL_SENTINEL

    # ── Apex curvature / retuse-notch fraction ─────────────────────────────
    # Window scaled to leaf size (not a fixed pixel count) so it works
    # across both large (Kattakumanjal) and small (Kasthuri_Dehi) species.
    apex_window = max(3, int(0.03 * n_samples))
    right_curv = _polyline_curvature(right_pts, apex_window)
    left_curv = _polyline_curvature(left_pts, apex_window)

    if len(apex_idx) > 0:
        apex_curvs = np.concatenate([right_curv[apex_idx], left_curv[apex_idx]])
        feats["botanical_apex_curvature_median"] = float(np.median(np.abs(apex_curvs)))
        # A retuse (notched) apex shows local CONCAVITY right at the tip
        # instead of the expected convex point -- curvature sign flips.
        feats["botanical_apex_retuse_fraction"] = float(np.mean(apex_curvs < 0))
    else:
        feats["botanical_apex_curvature_median"] = _BOTANICAL_SENTINEL
        feats["botanical_apex_retuse_fraction"] = _BOTANICAL_SENTINEL

    # ── Margin serration: fine-scale curvature oscillation ────────────────

    serr_window = max(2, int(0.008 * n_samples))
    right_fine = _polyline_curvature(right_pts, serr_window)
    left_fine = _polyline_curvature(left_pts, serr_window)
    fine_curv = np.concatenate([right_fine, left_fine])
    fine_curv = fine_curv[np.abs(fine_curv) > 1e-6]
    if len(fine_curv) > 3:
        signs = np.sign(fine_curv)
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        feats["botanical_margin_serration_freq"] = float(sign_changes / len(fine_curv))
        feats["botanical_margin_serration_amp"] = float(np.mean(np.abs(fine_curv)))
    else:
        feats["botanical_margin_serration_freq"] = _BOTANICAL_SENTINEL
        feats["botanical_margin_serration_amp"] = _BOTANICAL_SENTINEL

    # ── Margin tooth sharpness: rounded/crenate vs pointed/serrate ────────

    h_r, w_r = _peak_sharpness_stats(right_fine)
    h_l, w_l = _peak_sharpness_stats(left_fine)
    heights_all = np.concatenate([h_r, h_l])
    relwidths_all = np.concatenate([w_r, w_l])
    if len(heights_all) >= 2:
        sharpness_ratios = heights_all / relwidths_all
        feats["botanical_margin_tooth_sharpness"] = float(np.median(sharpness_ratios))
    else:
        feats["botanical_margin_tooth_sharpness"] = _BOTANICAL_SENTINEL

    # ── Leaflet-pair offset: base-obliqueness / opposite-vs-alternate proxy

    boundaries = np.sort(np.concatenate([[0], sinus_idx, [n_samples - 1]]))
    offsets, elongations = [], []
    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        if e - s < 4:
            continue
        r_seg, l_seg = right_w[s:e + 1], left_w[s:e + 1]
        r_peak = s + int(np.argmax(r_seg))
        l_peak = s + int(np.argmax(l_seg))
        span = e - s
        offsets.append(abs(r_peak - l_peak) / span)

        height = combined_smooth[s:e + 1].max()
        if height > 1e-3:
            elongations.append(span / height)

    if offsets:
        feats["botanical_pair_offset_median"] = float(np.median(offsets))
    else:
        feats["botanical_pair_offset_median"] = _BOTANICAL_SENTINEL

    if elongations:
        elong_arr = np.array(elongations)
        q1, q3 = np.percentile(elong_arr, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clean = elong_arr[(elong_arr >= lo) & (elong_arr <= hi)]
        if len(clean) == 0:
            clean = elong_arr
        feats["botanical_arc_elongation_median"] = float(np.median(clean))
        feats["botanical_arc_elongation_iqr"] = float(
            np.percentile(clean, 75) - np.percentile(clean, 25))
    else:
        feats["botanical_arc_elongation_median"] = _BOTANICAL_SENTINEL
        feats["botanical_arc_elongation_iqr"] = _BOTANICAL_SENTINEL

    return feats
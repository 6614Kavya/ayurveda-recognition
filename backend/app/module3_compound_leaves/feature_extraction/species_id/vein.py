import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from app.module3_compound_leaves.preprocessing.config import GLCM_DIST, GLCM_ANGLES   # kept for config parity

_BOTANICAL_SENTINEL = -1.0

# Keys added in this revision (targeting the Kasthuri_Dehi/Thunpath_Kurundu
# and Beli/Wal_Kollu pairs, which the original 2-feature botanical vein set
# did not cover -- see module docstring on each helper below). Kept as a
# single list so every early-return path in extract_vein_features() can
# stay schema-consistent without repeating five sentinel lines each time.
_NEW_BOTANICAL_VEIN_KEYS = [
    "botanical_vein_base_branch_angle",
    "botanical_vein_base_branch_count",
    "botanical_vein_angle_median",
    "botanical_vein_loop_fraction",
    "botanical_tertiary_reticulation_density",
]


def _empty_new_botanical_vein_features() -> dict:
    return {k: _BOTANICAL_SENTINEL for k in _NEW_BOTANICAL_VEIN_KEYS}


# ===========================================================================
# Skeleton-graph helpers (same longest-path-in-graph pattern as shape.py's
# rachis-axis extraction, kept independent/duplicated here per vein.py's
# existing convention of not importing across feature_extraction modules)
# ===========================================================================

def _skeleton_to_graph(skel: np.ndarray):
    """8-connected pixel graph of a binary skeleton. None if too sparse."""
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
    return G


def _largest_component_and_longest_path(G):
    """
    Restrict to the largest connected skeleton fragment (vein skeletons are
    often split into several disconnected pieces, unlike shape.py's mask
    skeleton which is one connected blob), then find the longest path in
    it via the standard double-Dijkstra tree-diameter trick. Returns
    (G_largest, path) or (None, None).
    """
    if G is None or G.number_of_nodes() < 10:
        return None, None
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) < 10:
            return None, None
        G = G.subgraph(largest_cc).copy()
    start = next(iter(G.nodes))
    lengths = nx.single_source_dijkstra_path_length(G, start, weight="weight")
    node_a = max(lengths, key=lengths.get)
    lengths_a = nx.single_source_dijkstra_path_length(G, node_a, weight="weight")
    node_b = max(lengths_a, key=lengths_a.get)
    path = nx.shortest_path(G, node_a, node_b, weight="weight")
    return G, path


def _base_branch_features(G, path) -> dict:
   
    if G is None or path is None or len(path) < 10:
        return {"botanical_vein_base_branch_angle": _BOTANICAL_SENTINEL,
                "botanical_vein_base_branch_count": 0.0}

    path_pts = np.array(path, dtype=np.float64)
    path_set = set(path)
    n = len(path)
    base_cutoff = max(3, int(0.2 * n))   # first/last 20% of the main path
    branch_points = [node for node in G.nodes
                     if G.degree(node) >= 3 and node not in path_set]

    all_angles = []
    for end in ("start", "end"):
        if end == "start":
            base_arr = path_pts[:base_cutoff]
            base_dir = path_pts[min(base_cutoff, n - 1)] - path_pts[0]
        else:
            base_arr = path_pts[-base_cutoff:]
            base_dir = path_pts[-1] - path_pts[max(0, n - 1 - base_cutoff)]
        bd_norm = np.linalg.norm(base_dir)
        if bd_norm < 1e-6:
            continue
        base_dir = base_dir / bd_norm

        for bp in branch_points:
            bp_arr = np.array(bp, dtype=np.float64)
            if np.linalg.norm(base_arr - bp_arr, axis=1).min() > 20:
                continue  # not near this end of the main path
            try:
                sub_lengths = nx.single_source_dijkstra_path_length(
                    G, bp, weight="weight", cutoff=250)
            except Exception:
                continue
            if not sub_lengths:
                continue
            far_node = max(sub_lengths, key=sub_lengths.get)
            far_len = sub_lengths[far_node]
            if far_len < 40:
                continue  # too short to be a rival main vein
            far_arr = np.array(far_node, dtype=np.float64)
            branch_dir = far_arr - bp_arr
            bn = np.linalg.norm(branch_dir)
            if bn < 1e-6:
                continue
            branch_dir = branch_dir / bn
            cos_ang = np.clip(np.dot(branch_dir, base_dir), -1, 1)
            angle_deg = float(np.degrees(np.arccos(abs(cos_ang))))  # undirected, 0-90
            if angle_deg < 35:
                all_angles.append(angle_deg)

    if all_angles:
        return {"botanical_vein_base_branch_angle": float(np.median(all_angles)),
                "botanical_vein_base_branch_count": float(len(all_angles))}
    return {"botanical_vein_base_branch_angle": _BOTANICAL_SENTINEL,
            "botanical_vein_base_branch_count": 0.0}


def _vein_angle_feature(vein_skel_work: np.ndarray, mask_work: np.ndarray) -> dict:
    
    try:
        ys, xs = np.nonzero(mask_work)
        if len(xs) < 20:
            return {"botanical_vein_angle_median": _BOTANICAL_SENTINEL}
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        principal = vt[0]
        principal_angle = np.degrees(np.arctan2(principal[1], principal[0]))

        lines = cv2.HoughLinesP(vein_skel_work, 1, np.pi / 180, threshold=15,
                                 minLineLength=12, maxLineGap=3)
        if lines is None or len(lines) == 0:
            return {"botanical_vein_angle_median": _BOTANICAL_SENTINEL}

        angles = []
        for (x1, y1, x2, y2) in lines[:, 0, :]:
            seg_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            rel = abs(seg_angle - principal_angle) % 180
            if rel > 90:
                rel = 180 - rel
            angles.append(rel)
        return {"botanical_vein_angle_median": float(np.median(angles))}
    except Exception:
        return {"botanical_vein_angle_median": _BOTANICAL_SENTINEL}


def _vein_loop_fraction(vein_binary_work: np.ndarray, mask_work: np.ndarray) -> dict:
    
    try:
        mask_px = mask_work > 0
        if mask_px.sum() < 200:
            return {"botanical_vein_loop_fraction": _BOTANICAL_SENTINEL}
        lamina = (mask_px & (vein_binary_work == 0)).astype(np.uint8)
        n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(lamina, connectivity=8)
        if n_cc <= 1:
            return {"botanical_vein_loop_fraction": _BOTANICAL_SENTINEL}

        bg = (~mask_px).astype(np.uint8)
        bg_dilated = cv2.dilate(bg, np.ones((3, 3), np.uint8)) > 0

        total_area = float(lamina.sum())
        enclosed_area = 0.0
        for i in range(1, n_cc):
            comp = (lbl == i)
            if not np.any(comp & bg_dilated):
                enclosed_area += float(stats[i, cv2.CC_STAT_AREA])

        return {"botanical_vein_loop_fraction": float(enclosed_area / total_area)}
    except Exception:
        return {"botanical_vein_loop_fraction": _BOTANICAL_SENTINEL}


def _tertiary_reticulation_density(gray_eq: np.ndarray, vein_skel_work: np.ndarray,
                                    mask_work: np.ndarray) -> dict:
    
    try:
        mask_px = mask_work > 0
        if mask_px.sum() < 200:
            return {"botanical_tertiary_reticulation_density": _BOTANICAL_SENTINEL}

        k_fine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fine_tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, k_fine)
        fine_tophat = cv2.bitwise_and(fine_tophat, fine_tophat, mask=mask_work)

        blk = max(7, (gray_eq.shape[0] // 30) | 1)
        fine_binary = cv2.adaptiveThreshold(
            fine_tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blk, -2)
        fine_binary = cv2.bitwise_and(fine_binary, fine_binary, mask=mask_work)
        fine_skel = skeletonize(fine_binary > 0)

        primary_dilated = cv2.dilate(vein_skel_work, np.ones((3, 3), np.uint8)) > 0
        tertiary_px = fine_skel & ~primary_dilated

        leaf_area = float(mask_px.sum())
        density = float(tertiary_px.sum()) / leaf_area * 10000
        return {"botanical_tertiary_reticulation_density": density}
    except Exception:
        return {"botanical_tertiary_reticulation_density": _BOTANICAL_SENTINEL}


def _extract_botanical_vein_features(gray_work: np.ndarray, gray_eq: np.ndarray,
                                      vein_skel_work: np.ndarray,
                                      vein_binary_work: np.ndarray,
                                      mask_work: np.ndarray) -> dict:
    """
   

    botanical_vein_spacing_period — distance (px, at WORK_SIZE) between
        adjacent parallel secondary veins. Targets Kathurupila (closely
        spaced, "comb-like ribbed" veins) vs Nil_Awariya (widely spaced,
        faint venation). Computed by projecting vein-skeleton pixels onto
        the axis PERPENDICULAR to the leaf's principal axis (found via a
        lightweight local PCA on mask_work -- not the same graph-based
        rachis axis used in shape.py, kept independent to avoid an
        import dependency between feature_extraction modules) and taking
        the first non-zero-lag peak of the autocorrelation of that
        1-D projection profile.

    botanical_vein_prominence_contrast — how strongly veins stand out
        from the surrounding blade tone. Targets Kathurupila (strongly
        raised, closely ribbed) vs Nil_Awariya (faint, barely raised),
        and Kalawal (fine visible venation) vs Kattakumanjal (glossy
        surface muting venation visibility). Computed as the mean
        absolute difference between the equalised grayscale at vein
        pixels and a heavily blurred ("local blade tone") version of the
        same image at those pixels, normalised by the blade's own tonal
        spread so the feature stays comparable across images of
        different overall contrast.

    botanical_vein_base_branch_angle / _count, botanical_vein_angle_median,
    botanical_vein_loop_fraction, botanical_tertiary_reticulation_density
    — added this revision, see their own helper functions above for full
    rationale. Added specifically because the original two features above
    target Kathurupila/Nil_Awariya and Kalawal/Kattakumanjal, leaving
    Kasthuri_Dehi/Thunpath_Kurundu (venation architecture) and
    Beli/Wal_Kollu (vein angle, looping, tertiary reticulation) with no
    dedicated handcrafted vein feature at all.

    STATUS: not yet visually validated against real Kathurupila /
    Nil_Awariya, Kalawal / Kattakumanjal, Kasthuri_Dehi / Thunpath_Kurundu,
    or Beli / Wal_Kollu photos. Same "provisional until pairwise
    validation" convention as shape.py's botanical additions.
    """
    feats = {}
    vein_px = vein_skel_work > 0
    mask_px = mask_work > 0

    if vein_px.sum() < 20 or mask_px.sum() < 100:
        feats["botanical_vein_spacing_period"] = _BOTANICAL_SENTINEL
        feats["botanical_vein_prominence_contrast"] = _BOTANICAL_SENTINEL
        feats.update(_empty_new_botanical_vein_features())
        return feats

    # ── Vein spacing periodicity ────────────────────────────────────────
    try:
        ys, xs = np.nonzero(mask_px)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        principal = vt[0]                       # leaf's long axis
        perpendicular = np.array([-principal[1], principal[0]])  # cross-vein axis

        vys, vxs = np.nonzero(vein_px)
        vein_pts = np.stack([vxs, vys], axis=1).astype(np.float64)
        proj = (vein_pts - mean) @ perpendicular

        # 1-D histogram of vein-pixel projections = "how many vein pixels
        # cross this line perpendicular to the leaf" -- parallel veins
        # produce a periodic signal in this histogram.
        n_bins = max(20, int(proj.max() - proj.min()))
        hist, _ = np.histogram(proj, bins=n_bins)
        hist = hist.astype(np.float64) - hist.mean()

        autocorr = np.correlate(hist, hist, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]  # keep zero-lag onward
        # first local max after lag 0 = dominant spacing period
        if len(autocorr) > 5:
            d = np.diff(autocorr)
            rising = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
            # first local minimum (trough) marks where autocorr starts
            # rising back up toward the next periodic peak
            if len(rising) > 0:
                search_start = rising[0] + 1
                peak_lag = search_start + int(np.argmax(autocorr[search_start:search_start + n_bins // 2]))
                feats["botanical_vein_spacing_period"] = float(peak_lag)
            else:
                feats["botanical_vein_spacing_period"] = _BOTANICAL_SENTINEL
        else:
            feats["botanical_vein_spacing_period"] = _BOTANICAL_SENTINEL
    except Exception:
        feats["botanical_vein_spacing_period"] = _BOTANICAL_SENTINEL

    # ── Vein prominence / contrast ──────────────────────────────────────
    try:
        blade_tone = cv2.GaussianBlur(gray_eq, (0, 0), sigmaX=25)
        diff = np.abs(gray_eq.astype(np.float64) - blade_tone.astype(np.float64))
        vein_contrast = diff[vein_px].mean()
        blade_spread = gray_eq[mask_px].astype(np.float64).std() + 1e-6
        feats["botanical_vein_prominence_contrast"] = float(vein_contrast / blade_spread)
    except Exception:
        feats["botanical_vein_prominence_contrast"] = _BOTANICAL_SENTINEL

    # ── Basal branch angle/count (triplinerved vs simple pinnate) ─────────
    try:
        G = _skeleton_to_graph(vein_skel_work)
        G_lcc, path = _largest_component_and_longest_path(G)
        feats.update(_base_branch_features(G_lcc, path))
    except Exception:
        feats["botanical_vein_base_branch_angle"] = _BOTANICAL_SENTINEL
        feats["botanical_vein_base_branch_count"] = 0.0

    # ── Dominant vein angle relative to leaf axis (Beli ~45-55 deg) ───────
    feats.update(_vein_angle_feature(vein_skel_work, mask_work))

    # ── Loop fraction (brochidodromous Beli vs open-pinnate Wal_Kollu) ────
    feats.update(_vein_loop_fraction(vein_binary_work, mask_work))

    # ── Tertiary reticulation density (Wal_Kollu's dense net pattern) ─────
    feats.update(_tertiary_reticulation_density(gray_eq, vein_skel_work, mask_work))

    return feats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All vein processing is done at this resolution (longest side).
# Keeps kernel sizes, CLAHE tiles and adaptive block sizes consistent
# across images regardless of how small the leaf is in the frame.
WORK_SIZE   = 512

# Padding around the bounding box before upscale.
# Prevents edge-halo artefacts from top-hat at the crop boundary.
ROI_PAD_PX  = 12

# Minimum linear dimension (px) of the crop before upscale.
# Crops smaller than this have no usable vein detail — return zeros.
MIN_CROP_PX = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_padded_bbox(leaf_mask: np.ndarray,
                     pad: int = ROI_PAD_PX
                     ) -> tuple[int, int, int, int] | None:
    
    coords = cv2.findNonZero(leaf_mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    H, W = leaf_mask.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return x1, y1, x2, y2


def _upscale_to_work_size(img: np.ndarray) -> tuple[np.ndarray, float]:
    
    h, w = img.shape[:2]
    scale = WORK_SIZE / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp), scale


def _build_vein_map(gray_work: np.ndarray,
                    mask_work: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    
    # Step A: CLAHE on full grayscale (no mask) — shadow normalisation
    # Applied BEFORE masking so each tile has a realistic local histogram
    # that includes both vein and lamina pixels.
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq  = clahe.apply(gray_work)

    # Step B: Black top-hat on FULL equalised image (no mask yet)
    # Applied to unmasked image so boundary sees real image content,
    # not the artificial zero-edge that masking would create.
    k_bthat      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    black_tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, k_bthat)

    # Step C: Clamp to foreground AFTER top-hat
    black_tophat = cv2.bitwise_and(black_tophat, black_tophat, mask=mask_work)

    # Step D: Adaptive threshold
    blk = max(11, (gray_work.shape[0] // 20) | 1)
    vein_binary = cv2.adaptiveThreshold(
        black_tophat, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blk, -2,
    )
    vein_binary = cv2.bitwise_and(vein_binary, vein_binary, mask=mask_work)

    # Step E: Skeletonise → 1-px vein centrelines
    vein_skel = skeletonize(vein_binary > 0).astype(np.uint8) * 255

    return vein_skel, vein_binary, gray_eq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_vein_features(img_sharp_bgr: np.ndarray,
                          leaf_mask: np.ndarray
                          ) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Extract shadow-robust, small-leaf-stable vein features.

    Parameters
    ----------
    img_sharp_bgr : enhanced BGR uint8 image (output of enhance.py), 512×512
    leaf_mask     : uint8 binary mask (255 = foreground), 512×512

    Returns
    -------
    feats       : dict — 4 vein features + 2 diagnostic columns
    vein_skel   : uint8 skeleton image in ORIGINAL 512×512 frame
    vein_binary : uint8 thresholded vein map in ORIGINAL 512×512 frame

    Pipeline summary
    ----------------
    1. Crop to padded leaf bounding box.
    2. Upscale crop to WORK_SIZE (512px longest side) with INTER_CUBIC.
       → All subsequent ops see the leaf at a consistent scale regardless
         of how small it was in the original frame.
    3. CLAHE (8×8 tiles = 64px each at WORK_SIZE) on full grayscale.
    4. Black top-hat (15px ellipse) on full equalised image, then mask.
    5. Adaptive threshold + skeletonise.
    6. Downscale vein maps back to original crop size, place in full frame.
    7. Compute density ratios using ORIGINAL mask pixel count (not upscaled)
       so that features are consistent with colour/texture/shape features
       which all use the original 512×512 coordinate space.
    """
    gray    = cv2.cvtColor(img_sharp_bgr, cv2.COLOR_BGR2GRAY)
    H, W    = gray.shape[:2]
    px_mask = leaf_mask > 0

    # Guard: completely empty mask
    if px_mask.sum() < 100:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {
            "botanical_vein_density": 0.0, "botanical_vein_length_ratio": 0.0,
            "botanical_vein_branch_density": 0.0, "botanical_vein_end_point_density": 0.0,
            "vein_coverage_pct": 0.0, "vein_roi_scale": 1.0,
            "botanical_vein_spacing_period": _BOTANICAL_SENTINEL,
            "botanical_vein_prominence_contrast": _BOTANICAL_SENTINEL,
            **_empty_new_botanical_vein_features(),
        }, empty, empty

    # ── Diagnostic: coverage in original frame ────────────────────────────
    leaf_area_px    = float(px_mask.sum())
    coverage_pct    = leaf_area_px / float(H * W)

    # ── Step 1: Crop to padded bounding box ───────────────────────────────
    bbox = _get_padded_bbox(leaf_mask, pad=ROI_PAD_PX)
    if bbox is None:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {
            "botanical_vein_density": 0.0, "botanical_vein_length_ratio": 0.0,
            "botanical_vein_branch_density": 0.0, "botanical_vein_end_point_density": 0.0,
            "vein_coverage_pct": round(coverage_pct, 4), "vein_roi_scale": 1.0,
            "botanical_vein_spacing_period": _BOTANICAL_SENTINEL,
            "botanical_vein_prominence_contrast": _BOTANICAL_SENTINEL,
            **_empty_new_botanical_vein_features(),
        }, empty, empty

    x1, y1, x2, y2 = bbox
    gray_crop = gray[y1:y2, x1:x2]
    mask_crop = leaf_mask[y1:y2, x1:x2]

    crop_h, crop_w = gray_crop.shape[:2]

    # Guard: crop too small to extract any vein detail
    if min(crop_h, crop_w) < MIN_CROP_PX:
        empty = np.zeros((H, W), dtype=np.uint8)
        return {
            "botanical_vein_density": 0.0, "botanical_vein_length_ratio": 0.0,
            "botanical_vein_branch_density": 0.0, "botanical_vein_end_point_density": 0.0,
            "vein_coverage_pct": round(coverage_pct, 4), "vein_roi_scale": 0.0,
            "botanical_vein_spacing_period": _BOTANICAL_SENTINEL,
            "botanical_vein_prominence_contrast": _BOTANICAL_SENTINEL,
            **_empty_new_botanical_vein_features(),
        }, empty, empty

    # ── Step 2: Upscale crop to WORK_SIZE ─────────────────────────────────
    # This is the key fix: every leaf is processed at the same effective
    # resolution, so CLAHE tiles, top-hat kernel and adaptive block size
    # always see veins at a consistent pixel scale.
    gray_work, roi_scale = _upscale_to_work_size(gray_crop)
    mask_work, _         = _upscale_to_work_size(mask_crop)
    # Re-binarise mask after interpolation artefacts from resize
    mask_work = (mask_work > 127).astype(np.uint8) * 255

    # ── Steps 3-5: CLAHE → top-hat → threshold → skeleton ────────────────
    vein_skel_work, vein_binary_work, gray_eq_work = _build_vein_map(gray_work, mask_work)

    # ── Step 6: Downscale results back to original crop size ──────────────
    # INTER_NEAREST preserves binary structure (no new grey values created).
    vein_skel_crop   = cv2.resize(vein_skel_work,   (crop_w, crop_h),
                                  interpolation=cv2.INTER_NEAREST)
    vein_binary_crop = cv2.resize(vein_binary_work, (crop_w, crop_h),
                                  interpolation=cv2.INTER_NEAREST)

    # Place crop results back into full 512×512 output frames
    vein_skel   = np.zeros((H, W), dtype=np.uint8)
    vein_binary = np.zeros((H, W), dtype=np.uint8)
    vein_skel[y1:y2, x1:x2]   = vein_skel_crop
    vein_binary[y1:y2, x1:x2] = vein_binary_crop

    # ── Step 7: Density features — original coordinate denominators ───────
    # leaf_area_px is from the original mask (512×512 space).
    # skel_px is counted in the full-frame vein_skel (same space).
    # This keeps vein features in the same units as colour/texture/shape.
    #
    # RECLASSIFIED to botanical_* (this revision): vein density and
    # branching pattern are an established character in plant leaf
    # morphology / venation-architecture classification -- not a generic
    # CV descriptor computed for its own sake. These four were originally
    # left unprefixed when the botanical_ naming convention was introduced
    # later in the project for the newer oil-gland/gloss/reticulation
    # features; that was an oversight in the naming convention, not a
    # judgement that vein density/branching lacks botanical grounding.
    # Reclassified so the SVM branch's guaranteed_botanical / pairwise_aware
    # selection (which keys off the "botanical_" substring) treats them
    # consistently with the rest of the botanical feature set.
    skel_px = float((vein_skel > 0).sum())

    feats: dict = {}
    feats["botanical_vein_density"] = skel_px / leaf_area_px if leaf_area_px > 0 else 0.0

    # ── Length ratio (normalised by contour perimeter) ─────────────────────
    # Perimeter from original mask — consistent with shape features.
    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = (float(cv2.arcLength(max(cnts, key=cv2.contourArea), True))
                 if cnts else 1.0)
    feats["botanical_vein_length_ratio"] = skel_px / perimeter if perimeter > 0 else 0.0

    # ── Branch & end-point densities ──────────────────────────────────────
    k_n    = np.ones((3, 3), np.uint8);  k_n[1, 1] = 0
    skel_b = (vein_skel > 0).astype(np.uint8)
    nbr    = cv2.filter2D(skel_b.astype(np.float32), -1, k_n.astype(np.float32))
    nbr    = (nbr * skel_b).astype(np.uint8)

    branch_pts = int((nbr >= 3).sum())
    end_pts    = int((nbr == 1).sum())

    feats["botanical_vein_branch_density"]    = branch_pts / leaf_area_px if leaf_area_px > 0 else 0.0
    feats["botanical_vein_end_point_density"] = end_pts    / leaf_area_px if leaf_area_px > 0 else 0.0

    # ── Diagnostic columns (not used by classifier, for audit CSV only) ───
    feats["vein_coverage_pct"] = round(coverage_pct, 4)
    feats["vein_roi_scale"]    = round(roi_scale, 4)

    # ── NEW: botanical vein-spacing / prominence features ──────────────────
    # Computed at WORK_SIZE resolution (before downscale) so they inherit
    # the same small-leaf-stable scale normalisation as the density
    # features above -- see module docstring.
    try:
        botanical_f = _extract_botanical_vein_features(
            gray_work, gray_eq_work, vein_skel_work, vein_binary_work, mask_work)
        feats.update(botanical_f)
    except Exception:
        feats["botanical_vein_spacing_period"] = _BOTANICAL_SENTINEL
        feats["botanical_vein_prominence_contrast"] = _BOTANICAL_SENTINEL
        feats.update(_empty_new_botanical_vein_features())

    return feats, vein_skel, vein_binary
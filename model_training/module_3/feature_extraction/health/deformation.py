"""
feature_extraction/health/deformation.py

Leaf-curling / non-planarity proxy features for the health branch.

WHY THIS EXISTS: none of the existing health feature groups (boundary,
holes, colour, scar, miner_trail, texture_health) measure curling,
wilting, or blade deformation at all -- confirmed gap, identified after
spot-checking leaves labeled "damaged_high" (visibly curled, dulled,
texture-changed) that scored near-healthy on the existing LDSI/health
index. This module targets that gap specifically.

All features here work from a single flat 2D photo -- there's no true
3D/depth information, so "curling" is inferred from proxies that a
non-planar, folded, or creased leaf surface produces differently than a
flat one under the same diffuse studio lighting used throughout this
project's photography:

  1. Specular highlight fragmentation (deform_specular_pct,
     deform_specular_blob_density): a FLAT leaf presents one continuous
     surface angle to the camera/light, so at most one smooth highlight
     region appears. A CURLED/WAVY leaf presents several different local
     surface angles, each of which can catch a highlight independently
     -- multiple small, scattered specular blobs instead of one. Reuses
     colour_health.py's exact specular definition (L>225 & S<40), which
     that module only used to EXCLUDE false positives from the pale-
     patch gate; here the raw coverage/fragmentation itself is the
     signal, not noise to discard.

  2. Width-profile roughness (deform_width_profile_roughness): projects
     the mask onto its principal/rachis axis and measures leaf width at
     each point along that axis. A healthy leaf's width profile tapers
     smoothly; folding/curling creates local width contractions that
     make the profile choppy. Same underlying idea as the width-profile
     features sketched for species-ID's planned morphology.py (project
     memory records that as future work, not yet implemented there) --
     applied here for a different purpose (health, not species-ID), and
     implemented independently since the two branches' pipelines don't
     share a features module.

  3. Luminance spread (deform_luminance_std): simple std of LAB L* over
     the whole masked leaf. A crease/fold produces a sharper local
     shadow-highlight pair than the smooth graded shading of a flat
     surface, so a wavier leaf tends toward higher L* spread. This is
     the cheapest and noisiest of the three -- keep it, but expect
     Ridge/correlation to downweight it if it doesn't hold up on real
     data (same treatment every other candidate column in this project
     gets, per health_index.py's documented pattern).

None of these three are validated against real severity labels yet --
same status as texture_health.py's features when they were first added
("round 3" in health_index.py's docstring): add as new candidates to
SUBSCORE_RAW_COLUMNS, let the binary-target Ridge fit + per-subscore
Spearman rho (per_subscore_correlation()) tell you honestly whether
each one earns its place. Do not assume any of the three works before
that check -- these are principled proxies, not verified signal.
"""
import cv2
import numpy as np

SENTINEL = -1.0

# Specular definition -- IDENTICAL to colour_health.py's `specular` gate,
# so the two modules stay in agreement about what counts as a highlight
# rather than drifting into two different definitions of the same thing.
SPECULAR_L_THRESH = 225
SPECULAR_S_THRESH = 40
MIN_SPECULAR_BLOB_PX = 5  # ignore single-pixel noise, not a real facet

# Width-profile roughness
N_AXIS_BINS = 40
MIN_PTS_PER_BIN = 5
MIN_VALID_BINS = 10
MIN_FOREGROUND_PX = 200


def _principal_axis(coords_xy: np.ndarray):
    """PCA on a set of (x, y) points -> (centroid, principal unit vector,
    perpendicular unit vector). Returns (None, None, None) if degenerate."""
    if coords_xy.shape[0] < 2:
        return None, None, None
    mean = coords_xy.mean(axis=0)
    centered = coords_xy - mean
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return None, None, None
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    norm = np.linalg.norm(principal)
    if norm < 1e-9:
        return None, None, None
    principal = principal / norm
    perp = np.array([-principal[1], principal[0]])
    return mean, principal, perp


def _width_profile_roughness(mask_final: np.ndarray, rachis_mask=None) -> float:
    """
    Scale-invariant roughness of the leaf's width profile along its
    principal axis. Uses rachis_mask to determine axis orientation when
    available (more anatomically meaningful for compound leaves, same
    convention as boundary.py/scar.py); falls back to the whole mask's
    own PCA axis otherwise.

    Returns a value >= 0, higher = choppier/less smooth width profile.
    SENTINEL if the mask is too small/degenerate to trust a profile from.
    """
    fg = mask_final.astype(bool)
    ys, xs = np.where(fg)
    if len(ys) < MIN_FOREGROUND_PX:
        return SENTINEL

    if rachis_mask is not None and np.count_nonzero(rachis_mask) >= 50:
        axis_ys, axis_xs = np.where(rachis_mask.astype(bool))
        axis_coords = np.stack([axis_xs, axis_ys], axis=1).astype(np.float64)
    else:
        axis_coords = np.stack([xs, ys], axis=1).astype(np.float64)

    mean, principal, perp = _principal_axis(axis_coords)
    if principal is None:
        return SENTINEL

    pts = np.stack([xs, ys], axis=1).astype(np.float64) - mean
    t = pts @ principal
    s = pts @ perp

    t_min, t_max = float(t.min()), float(t.max())
    if t_max - t_min < 1e-6:
        return SENTINEL

    bin_edges = np.linspace(t_min, t_max, N_AXIS_BINS + 1)
    bin_idx = np.clip(np.digitize(t, bin_edges) - 1, 0, N_AXIS_BINS - 1)

    widths = []
    for b in range(N_AXIS_BINS):
        sel = s[bin_idx == b]
        if sel.size >= MIN_PTS_PER_BIN:
            widths.append(float(sel.max() - sel.min()))

    if len(widths) < MIN_VALID_BINS:
        return SENTINEL

    widths = np.asarray(widths)
    median_w = float(np.median(widths))
    if median_w < 1e-6:
        return SENTINEL

    norm_widths = widths / median_w
    # second discrete difference: ~0 for a smooth monotonic/tapering
    # profile, larger for a profile with local bumps/contractions
    second_diff = np.diff(norm_widths, n=2)
    if second_diff.size == 0:
        return SENTINEL
    return float(np.mean(np.abs(second_diff)))


def _specular_features(masked_raw_bgr: np.ndarray, mask_final: np.ndarray):
    fg = mask_final.astype(bool)
    leaf_px = int(np.count_nonzero(fg))
    if leaf_px < MIN_FOREGROUND_PX:
        return SENTINEL, SENTINEL

    lab = cv2.cvtColor(masked_raw_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(masked_raw_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, S = lab[..., 0], hsv[..., 1]

    specular = (L > SPECULAR_L_THRESH) & (S < SPECULAR_S_THRESH) & fg
    specular_pct = float(np.count_nonzero(specular) / leaf_px * 100.0)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        specular.astype(np.uint8), connectivity=8
    )
    blob_count = sum(
        1 for lbl in range(1, n_labels)
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_SPECULAR_BLOB_PX
    )
    # normalised per 10k leaf px so leaf/leaflet size doesn't confound
    # blob count directly -- same normalisation principle as boundary.py's
    # boundary_notch_density.
    blob_density = float(blob_count / (leaf_px / 10000.0))

    return specular_pct, blob_density


def _luminance_std(masked_raw_bgr: np.ndarray, mask_final: np.ndarray) -> float:
    fg = mask_final.astype(bool)
    if np.count_nonzero(fg) < MIN_FOREGROUND_PX:
        return SENTINEL
    lab = cv2.cvtColor(masked_raw_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_fg = lab[..., 0][fg]
    return float(np.std(L_fg))


def extract_deformation_features(
    masked_raw_bgr: np.ndarray,
    mask_final: np.ndarray,
    rachis_mask: np.ndarray = None,
) -> dict:
    """
    Parameters
    ----------
    masked_raw_bgr : unenhanced BGR image, background already zeroed
                      (same input every other health module takes).
    mask_final     : binary leaf mask, foreground = 255/True.
    rachis_mask    : optional, from masking.py's diag output. Recommended
                      for compound leaves (gives width_profile_roughness a
                      more meaningful axis) -- safe to omit, falls back to
                      the whole mask's own PCA axis.

    Returns
    -------
    dict of deform_* features, all SENTINEL (-1.0) if the mask is too
    small/degenerate to compute from.
    """
    specular_pct, specular_blob_density = _specular_features(masked_raw_bgr, mask_final)
    width_roughness = _width_profile_roughness(mask_final, rachis_mask)
    luminance_std = _luminance_std(masked_raw_bgr, mask_final)

    return {
        "deform_specular_pct": specular_pct,
        "deform_specular_blob_density": specular_blob_density,
        "deform_width_profile_roughness": width_roughness,
        "deform_luminance_std": luminance_std,
    }
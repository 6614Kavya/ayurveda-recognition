"""
Wound periderm / scar-tissue feature.

Scar tissue (corky, desaturated tan/grey periderm forming around old
wounds) has a distinct colour signature from both healthy green and fresh
necrotic brown -- so it's sampled only in a spatial BAND around damage
sites (margin region + hole boundaries), not across the whole leaf. This
keeps the feature specific to periderm-around-damage rather than picking
up unrelated tan variegation elsewhere on healthy tissue.

--- FIX (this session) ---
The margin band used to wrap the ENTIRE outer contour, including the
concave gaps between leaflets on compound leaves. Those gaps often catch
soft shadow (mid-brightness, desaturated) that matches the scar colour
signature, so every compound leaf picked up a nonzero scar_tissue_ratio
from leaflet-junction shadow alone. Same fix as boundary.py: exclude the
rachis-proximity band from the margin ring before sampling. The hole-band
component is left ungated -- a hole is a hole regardless of where it sits
relative to the rachis.
"""
import cv2
import numpy as np

SENTINEL = -1.0
BAND_RADIUS_PX = 12
DEFAULT_RACHIS_PROXIMITY_PX = 15  # matches boundary.py / masking.py


def _damage_site_mask(
    mask_final: np.ndarray,
    mask_before_holefill,
    rachis_mask: np.ndarray = None,
    rachis_proximity_px: int = DEFAULT_RACHIS_PROXIMITY_PX,
) -> np.ndarray:
    """Union of (a) a band just inside the leaf's outer contour (proxy for
    margin damage sites, excluding rachis-junction gaps) and (b) a dilated
    band around detected holes."""
    fg = mask_final.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND_RADIUS_PX + 1,) * 2)

    eroded = cv2.erode(fg, kernel)
    margin_band = cv2.subtract(fg, eroded)

    if rachis_mask is not None:
        rkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rachis_proximity_px + 1,) * 2)
        rachis_dilated = cv2.dilate(rachis_mask.astype(np.uint8), rkernel)
        margin_band = cv2.bitwise_and(margin_band, cv2.bitwise_not(rachis_dilated))

    sites = margin_band
    if mask_before_holefill is not None:
        filled_in = cv2.bitwise_and(fg, cv2.bitwise_not(mask_before_holefill.astype(np.uint8)))
        hole_band = cv2.dilate(filled_in, kernel)
        sites = cv2.bitwise_or(sites, hole_band)

    return cv2.bitwise_and(sites, fg).astype(bool)


def extract_scar_features(
    img_bgr: np.ndarray,
    mask_final: np.ndarray,
    mask_before_holefill,
    rachis_mask: np.ndarray = None,
    rachis_proximity_px: int = DEFAULT_RACHIS_PROXIMITY_PX,
) -> dict:
    """
    Parameters
    ----------
    img_bgr : masked_raw image, un-enhanced.
    mask_final : binary leaf mask.
    mask_before_holefill : Stage-7 pre-holefill mask, or None (see
        holes.py docstring -- same availability caveat applies here).
    rachis_mask : binary rachis mask from masking.py's diag output, or
        None. Recommended for compound leaves (see module docstring).

    Returns
    -------
    dict with scar_tissue_ratio (fraction of the damage-site band
    classified as scar/periderm colour).
    """
    site_mask = _damage_site_mask(mask_final, mask_before_holefill, rachis_mask, rachis_proximity_px)
    band_px = int(np.count_nonzero(site_mask))
    if band_px == 0:
        return {"scar_tissue_ratio": 0.0}

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, A = lab[..., 0][site_mask], lab[..., 1][site_mask]
    S = hsv[..., 1][site_mask]

    # desaturated tan/grey, mid-brightness -- distinguishes from vivid
    # healthy green (a* lower here) and wet necrotic brown-black (L lower,
    # S typically higher for fresh necrosis vs. dried scar)
    scar = (S < 70) & (L > 90) & (L < 190) & (A > 118) & (A < 140)

    ratio = float(np.count_nonzero(scar) / band_px)
    return {"scar_tissue_ratio": ratio}
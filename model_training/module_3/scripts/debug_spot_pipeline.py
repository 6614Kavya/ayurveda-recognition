"""
debug_spot_pipeline.py

Run this against ONE real "high"-damage image to find exactly which
stage is eating every candidate pixel. Prints raw pixel counts at each
step rather than just the final spot_count, so we can see whether the
problem is:
  (a) _classify_damage_masks() itself finding ~0 necrotic/chlorotic/pale
      pixels on this dataset (would contradict the earlier colour_pct_*
      analysis, but worth confirming directly on THIS dataset copy), or
  (b) the rachis exclusion swallowing everything (e.g. rachis_mask is
      much larger than a thin stem line for this species/masking run,
      and dilating it by 15px balloons it to cover the whole leaflet), or
  (c) the edge-artifact strip (should be tiny -- 2px -- but confirming), or
  (d) the hole exclusion swallowing everything.

Usage (from module_3/ root):
    python debug_spot_pipeline.py path/to/one/high/damage/image.jpg
"""
import sys

import cv2
import numpy as np

from preprocessing.shared.resize import letterbox_resize
from preprocessing.shared.masking import select_mask, qc_check
from feature_extraction.health.colour_health import _classify_damage_masks
from feature_extraction.health.spots import _build_exclusion_mask, extract_spot_features


def main(img_path: str):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[error] could not read {img_path}")
        return

    resized, _ = letterbox_resize(img_bgr)
    mask_final, mask_choice, diag = select_mask(resized)
    qc_ok, qc_reason = qc_check(diag)
    print(f"qc_pass={qc_ok} qc_reason={qc_reason!r} mask_choice={mask_choice}")

    mask_before_holefill = diag.get("mask_before_holefill")
    rachis_mask = diag.get("rachis_mask")

    leaf_px = int(np.count_nonzero(mask_final))
    print(f"leaf_area (mask_final) = {leaf_px} px")
    print(f"mask_before_holefill available: {mask_before_holefill is not None}"
          + (f" | pixels={int(np.count_nonzero(mask_before_holefill))}" if mask_before_holefill is not None else ""))
    print(f"rachis_mask available: {rachis_mask is not None}"
          + (f" | rachis raw pixels={int(np.count_nonzero(rachis_mask))}"
             f" ({int(np.count_nonzero(rachis_mask))/leaf_px*100:.1f}% of leaf area)" if rachis_mask is not None else ""))

    masked_raw = cv2.bitwise_and(resized, resized, mask=mask_final.astype(np.uint8))

    damage_masks = _classify_damage_masks(masked_raw, mask_final)
    raw_necrotic = int(damage_masks["necrotic"].sum())
    raw_chlorotic = int(damage_masks["chlorotic"].sum())
    raw_pale = int(damage_masks["pale"].sum())
    raw_all_damage = int((damage_masks["necrotic"] | damage_masks["chlorotic"] | damage_masks["pale"]).sum())
    print(f"\n[stage 1: raw colour classification, BEFORE any exclusion]")
    print(f"  necrotic px = {raw_necrotic} ({raw_necrotic/leaf_px*100:.2f}% of leaf)")
    print(f"  chlorotic px = {raw_chlorotic} ({raw_chlorotic/leaf_px*100:.2f}% of leaf)")
    print(f"  pale px = {raw_pale} ({raw_pale/leaf_px*100:.2f}% of leaf)")
    print(f"  total damage-coloured px (any category) = {raw_all_damage} ({raw_all_damage/leaf_px*100:.2f}% of leaf)")

    all_damage = damage_masks["necrotic"] | damage_masks["chlorotic"] | damage_masks["pale"]

    exclude_rachis_only = _build_exclusion_mask(mask_final, None, rachis_mask, edge_artifact_px=0)
    exclude_edge_only = _build_exclusion_mask(mask_final, None, None, edge_artifact_px=2)
    exclude_holes_only = _build_exclusion_mask(mask_final, mask_before_holefill, None, edge_artifact_px=0)
    exclude_full = _build_exclusion_mask(mask_final, mask_before_holefill, rachis_mask, edge_artifact_px=2)

    def surviving(exclude_mask, label):
        surv = int((all_damage & ~exclude_mask).sum())
        print(f"  after excluding [{label}] only: {surv} damage px survive "
              f"({surv/max(raw_all_damage,1)*100:.1f}% of raw damage px kept)")

    print(f"\n[stage 2: how much of the raw damage survives EACH exclusion individually]")
    surviving(exclude_rachis_only, "rachis (dilated)")
    surviving(exclude_edge_only, "edge-artifact strip (2px)")
    surviving(exclude_holes_only, "hole regions")

    final_surviving = int((all_damage & ~exclude_full).sum())
    print(f"\n[stage 3: final, ALL exclusions combined]")
    print(f"  surviving candidate px = {final_surviving} "
          f"({final_surviving/max(raw_all_damage,1)*100:.1f}% of raw damage px kept)")

    out = extract_spot_features(masked_raw, mask_final, mask_before_holefill, rachis_mask)
    print(f"\n[final extract_spot_features() output]")
    print(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_spot_pipeline.py path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
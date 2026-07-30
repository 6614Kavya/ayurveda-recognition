"""
VedaVision -- Health Feature Extraction QC
===========================================
Answers "are the health features actually correct?" at three levels,
combining what already exists in the codebase (per_subscore_correlation,
validate_monotonicity, check_augmentation_safety) with the two pieces
that are currently missing: (1) a full-column statistical audit instead
of just the 7 SUBSCORE_RAW_COLUMNS, and (2) a VISUAL overlay so you can
eyeball whether hole_count/scar_tissue_ratio/miner_trail_* are actually
firing on real damage, not just correlating with `level` by coincidence.

Run in this order
------------------
1. `--mode audit`  -- run on the full CSVs (or the dataset root, fresh),
   flag columns that are constant, all-sentinel, mostly-NaN, or don't
   move with severity at all. Fast, no images shown.

2. `--mode visual` -- pick N images per level (default: a few per
   species) and save a side-by-side panel per image showing:
     original -> mask_final boundary -> holes (red) -> scar band (orange)
     -> miner-trail candidates (cyan), with the extracted feature values
   printed underneath. This is the "is the code finding what a human
   would call damage" check -- no statistics can substitute for it.

3. `--mode edge-case` -- the cheapest and most important check: pull the
   `healthy` leaves and confirm their damage features are actually near
   zero. If a "healthy" leaf shows hole_count=4 or scar_tissue_ratio=0.3,
   something upstream (masking, damage-site gating, or the health/species
   folder split itself) is wrong, independent of whatever correlation
   with `level` looks like in aggregate.

Usage
-----
    python -m preprocessing.health.feature_qc --mode audit \\
        --top-csv processed/features/health_features_top.csv \\
        --bottom-csv processed/features/health_features_bottom.csv

    python -m preprocessing.health.feature_qc --mode visual \\
        --dataset-root dataset/health --species beli --n-per-level 3 \\
        --out-dir qc_panels

    python -m preprocessing.health.feature_qc --mode edge-case \\
        --top-csv processed/features/health_features_top.csv \\
        --bottom-csv processed/features/health_features_bottom.csv
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from preprocessing.shared.resize import letterbox_resize
from preprocessing.shared.masking import select_mask
from preprocessing.health.pipeline import run_health_pipeline_from_resized
from feature_extraction.health.scar import _damage_site_mask

LEVEL_ORDER = {"healthy": 0, "low": 1, "mid": 2, "high": 3}
SENTINELS = {-1.0, -1}

# Columns that are IDs/metadata, not features -- excluded from the audit.
NON_FEATURE_COLS = {
    "leaf_id", "variant_id", "species", "level", "view", "is_test",
    "is_augmented", "source_path", "image_path", "qc_pass", "qc_reason",
    "mask_choice",
}

# Features that should sit at (near) zero on a genuinely healthy leaf.
# Used only by --mode edge-case.
DAMAGE_FEATURES = [
    "hole_count", "hole_area_ratio", "scar_tissue_ratio",
    "miner_trail_count", "miner_trail_coverage_pct",
    "colour_pct_necrotic", "colour_pct_chlorotic", "colour_pct_pale_patch",
    "boundary_notch_count",
]


# --------------------------------------------------------------------------
# Mode 1: full-column statistical audit
# --------------------------------------------------------------------------
def _load(top_csv, bottom_csv):
    top = pd.read_csv(top_csv)
    bottom = pd.read_csv(bottom_csv)
    top["view"], bottom["view"] = "top", "bottom"
    return pd.concat([top, bottom], ignore_index=True)


def run_audit(top_csv, bottom_csv, originals_only=True):
    df = _load(top_csv, bottom_csv)
    if originals_only and "variant_id" in df.columns:
        df = df[df["variant_id"] == 0]
    if "qc_pass" in df.columns:
        n_total = len(df)
        df = df[df["qc_pass"] == True]  # noqa: E712
        print(f"qc_pass: {len(df)}/{n_total} rows kept "
              f"({100 * len(df) / max(n_total, 1):.1f}%)")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    ordinal = df["level"].map(LEVEL_ORDER)

    print(f"\n{'column':32s} {'nan%':>6s} {'sentinel%':>10s} "
          f"{'nunique':>8s} {'rho_vs_level':>12s} {'p':>10s}  flag")
    rows = []
    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        n = len(vals)
        nan_pct = 100 * vals.isna().sum() / n if n else float("nan")
        sentinel_pct = 100 * vals.isin(SENTINELS).sum() / n if n else float("nan")
        nunique = vals.nunique(dropna=True)

        usable = vals.notna() & ~vals.isin(SENTINELS)
        if usable.sum() > 5 and nunique > 1:
            rho, p = spearmanr(ordinal[usable], vals[usable])
        else:
            rho, p = float("nan"), float("nan")

        flags = []
        if nunique <= 1:
            flags.append("CONSTANT")
        if sentinel_pct > 50:
            flags.append("MOSTLY-SENTINEL")
        if nan_pct > 20:
            flags.append("HIGH-NAN")
        if not pd.isna(rho) and abs(rho) < 0.05:
            flags.append("NO-TREND-WITH-LEVEL")
        flag_str = ",".join(flags) if flags else "ok"

        rows.append({"column": col, "nan_pct": nan_pct, "sentinel_pct": sentinel_pct,
                      "nunique": nunique, "rho": rho, "p": p, "flags": flag_str})
        print(f"{col:32s} {nan_pct:6.1f} {sentinel_pct:10.1f} {nunique:8d} "
              f"{rho if not pd.isna(rho) else float('nan'):12.3f} "
              f"{p if not pd.isna(p) else float('nan'):10.3g}  {flag_str}")

    print("\nNOTE: a low/absent rho vs `level` is not automatically a bug -- some "
          "features (e.g. hole_mean_size) are legitimately noisy or only meaningful "
          "conditional on hole_count>0. Read HIGH-NAN / MOSTLY-SENTINEL / CONSTANT as "
          "'go look at this column', not as an automatic fail.")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Mode 2: visual overlay panels
# --------------------------------------------------------------------------
def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(out, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _overlay_mask(base_img, mask_bool, color, alpha=0.45, outline_only=False):
    """Alpha-blend a colored overlay so the underlying photo stays visible,
    instead of the old solid-fill approach that hid everything under it."""
    out = base_img.copy()
    if outline_only:
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)
        return out
    colored = np.zeros_like(base_img)
    colored[:] = color
    blended = cv2.addWeighted(base_img, 1 - alpha, colored, alpha, 0)
    out = base_img.copy()
    out[mask_bool] = blended[mask_bool]
    return out


def _make_panel(img_resized, mask_final, mask_before_holefill, rachis_mask, feats):
    """Returns a horizontal strip of SEPARATE labeled sub-panels rather than
    one composite image -- solid-fill overlays stacked on top of each other
    made it impossible to tell the real photo from the annotations."""
    fg_bool = mask_final.astype(bool)
    panels = []

    # 1. original photo, unmodified
    panels.append(_label(img_resized.copy(), "original"))

    # 2. mask boundary only (thin outline, doesn't hide anything)
    p_mask = _overlay_mask(img_resized, fg_bool, (255, 255, 255), outline_only=True)
    panels.append(_label(p_mask, "mask boundary"))

    # 3. holes -- semi-transparent red, only where holes actually are
    if mask_before_holefill is not None:
        filled_in = cv2.bitwise_and(
            mask_final.astype(np.uint8),
            cv2.bitwise_not(mask_before_holefill.astype(np.uint8)),
        ).astype(bool)
        p_holes = _overlay_mask(img_resized, filled_in, (0, 0, 255), alpha=0.6)
    else:
        p_holes = img_resized.copy()
    panels.append(_label(p_holes, f"holes (n={feats.get('hole_count')})"))

    # 4. scar/damage-site band -- semi-transparent orange
    site_mask = _damage_site_mask(mask_final, mask_before_holefill, rachis_mask)
    p_scar = _overlay_mask(img_resized, site_mask, (0, 165, 255), alpha=0.35)
    panels.append(_label(p_scar, f"scar band (ratio={feats.get('scar_tissue_ratio', 0):.2f})"))

    # 5. rachis mask -- thin outline only (it should be a THIN line; if this
    #    panel looks like it covers most of the leaf, that's a bug to chase,
    #    not a rendering issue)
    if rachis_mask is not None:
        rachis_bool = rachis_mask.astype(bool)
        rachis_pct = 100 * rachis_bool.sum() / max(fg_bool.sum(), 1)
        p_rachis = _overlay_mask(img_resized, rachis_bool, (0, 255, 255), alpha=0.7)
        panels.append(_label(p_rachis, f"rachis mask ({rachis_pct:.1f}% of leaf)"))
    else:
        panels.append(_label(img_resized.copy(), "rachis mask: unavailable"))

    strip = np.hstack(panels)
    footer = (f"necrotic%={feats.get('colour_pct_necrotic', 0):.1f}  "
              f"chlorotic%={feats.get('colour_pct_chlorotic', 0):.1f}  "
              f"miner_cnt={feats.get('miner_trail_count')}  "
              f"boundary_notch_count={feats.get('boundary_notch_count')}")
    footer_bar = np.zeros((20, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer_bar, footer, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, footer_bar])


def run_visual(dataset_root: Path, out_dir: Path, species: str | None, n_per_level: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = ["healthy", "damaged_low", "damaged_mid", "damaged_high"]
    pattern_species = species if species else "*"
    all_rows = []

    for level in levels:
        paths = sorted(dataset_root.glob(f"{pattern_species}/{level}/*/*.jpg"))
        paths = [p for p in paths if not p.stem.startswith("test_")][:n_per_level]
        if not paths:
            print(f"[warn] no images found for level={level}")
            continue

        for img_path in paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            resized, _ = letterbox_resize(img_bgr)
            mask_final, _, diag = select_mask(resized)
            mask_bhf = diag.get("mask_before_holefill")
            rachis_mask = diag.get("rachis_mask")

            row = run_health_pipeline_from_resized(
                resized, mask_final, mask_bhf, rachis_mask, image_path=str(img_path)
            )
            if not row.get("qc_pass"):
                print(f"[skip] QC failed for {img_path}: {row.get('qc_reason')}")
                continue

            panel = _make_panel(resized, mask_final, mask_bhf, rachis_mask, row)
            out_stem = f"{img_path.parts[-4]}__{level}__{img_path.stem}"
            cv2.imwrite(str(out_dir / f"{out_stem}.png"), panel)

            # every extracted feature, not just the handful shown on the panel
            with open(out_dir / f"{out_stem}.json", "w") as f:
                json.dump(row, f, indent=2, default=str)

            print(f"\n[saved] {out_dir / f'{out_stem}.png'}  (+ .json alongside it)")
            print(f"  --- all {len(row)} extracted columns for this image ---")
            for k in sorted(row.keys()):
                print(f"  {k:32s} {row[k]}")

            row_with_meta = {"image_path": str(img_path), "level": level, **row}
            all_rows.append(row_with_meta)

    if all_rows:
        summary_path = out_dir / "sampled_features.csv"
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        print(f"\n[saved] all sampled images' full feature rows -> {summary_path} "
              f"(one row per image, every column -- open in Excel/pandas to scan "
              f"the whole feature bank side by side rather than one image at a time)")

    print(f"\nReview the panels in {out_dir} -- each PNG is a strip of 5 "
          f"labeled sub-images: original | mask boundary | holes (red) | "
          f"scar band (orange) | rachis mask (yellow), each semi-transparent "
          f"over the real photo so nothing gets hidden. Confirm red/orange "
          f"actually sit on visible damage, not on shadow or leaflet-junction "
          f"gaps -- and check the rachis panel's '% of leaf' figure: it "
          f"should be a small number (a thin line), not a large chunk of "
          f"the leaf area.")


# --------------------------------------------------------------------------
# Mode 3: edge-case check on healthy leaves
# --------------------------------------------------------------------------
def run_edge_case(top_csv, bottom_csv):
    df = _load(top_csv, bottom_csv)
    if "variant_id" in df.columns:
        df = df[df["variant_id"] == 0]
    if "qc_pass" in df.columns:
        df = df[df["qc_pass"] == True]  # noqa: E712

    healthy = df[df["level"] == "healthy"]
    print(f"n healthy leaves (both views, originals only): {len(healthy)}\n")

    cols = [c for c in DAMAGE_FEATURES if c in df.columns]
    print(healthy[cols].describe().round(3).to_string())

    print("\n=== worst offenders (top 5 by each damage feature, healthy leaves only) ===")
    for c in cols:
        vals = pd.to_numeric(healthy[c], errors="coerce")
        worst = healthy.assign(_v=vals).nlargest(5, "_v")[["species", "source_path", c]]
        if worst[c].max() and worst[c].max() > 0:
            print(f"\n-- {c} --")
            print(worst.to_string(index=False))

    print("\nAny 'healthy' leaf with a large damage-feature value here is either "
          "a mislabelled image (wrong folder) or a real upstream bug -- open the "
          "source_path and check by eye before trusting the aggregate correlations.")


def main():
    parser = argparse.ArgumentParser(description="VedaVision health feature QC")
    parser.add_argument("--mode", choices=["audit", "visual", "edge-case"], required=True)
    parser.add_argument("--top-csv", default="processed/features/health_features_top.csv")
    parser.add_argument("--bottom-csv", default="processed/features/health_features_bottom.csv")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/health"))
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--n-per-level", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("qc_panels"))
    args = parser.parse_args()

    if args.mode == "audit":
        run_audit(args.top_csv, args.bottom_csv)
    elif args.mode == "visual":
        run_visual(args.dataset_root, args.out_dir, args.species, args.n_per_level)
    elif args.mode == "edge-case":
        run_edge_case(args.top_csv, args.bottom_csv)


if __name__ == "__main__":
    main()
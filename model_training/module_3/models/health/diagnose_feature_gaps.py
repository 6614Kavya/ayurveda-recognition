"""
models/health/diagnose_feature_gaps.py

Per-species feature-gap diagnostic. Answers "which feature groups are
blind for which species" directly from the already-extracted CSVs --
no images needed, since everything here operates on numbers already on
disk.

For every candidate feature (raw features + sub-scores), and separately
per species, computes:
    healthy / low / mid / high group medians
    a robust effect size: (high_median - healthy_median) / healthy_IQR
        (healthy_IQR floored to avoid divide-by-tiny-number blowups)

A feature with a LOW |effect size| for a given species means: even the
"high" damage leaves of that species don't move this feature away from
its healthy baseline -- i.e. this feature is blind to whatever the real
damage looks like for that species. A feature with a HIGH |effect size|
means it IS separating healthy from high for that species -- so if
EVERY feature comes back low for a species, that's strong evidence the
real damage mode for that species' "high" leaves isn't represented by
ANY current feature (curling, diffuse dulling, etc. -- see deformation.py
and colour_health.py's threshold caveats).

Also reports, per leaf, how many of the candidate features are
"elevated" (>1.5 robust-z from that species' healthy baseline) --
this directly tests the "single-dominant-symptom dilution" hypothesis:
if most 'high' leaves have only 1 feature elevated (not many), that
explains why an unweighted/lightly-weighted average score stays low
even for genuinely damaged leaves.

Run from module_3/ root:

    python -m models.health.diagnose_feature_gaps --out gap_report.csv

Reads from the SAME CSVs models/health/train_health_index.py uses
(TOP_CSV, BOTTOM_CSV), and reuses model_training.py's normalize_levels()
and fuse_top_bottom(), so folder-label quirks are already handled the
same way.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.health.model_training import (
    load_and_merge_from, TOP_CSV, BOTTOM_CSV, NON_FEATURE_COLS,
)
from models.health.classifier import fuse_top_bottom

LEVEL_ORDER = ["healthy", "low", "mid", "high"]
EPS = 1e-6

# Candidate features to check -- raw damage-relevant columns plus the
# already-computed sub-scores. Extend this list if you add more feature
# modules later (e.g. new deform_* variants).
CANDIDATE_FEATURES = [
    "boundary_margin_deficit_ratio", "boundary_contour_roughness", "boundary_notch_density",
    "hole_area_ratio", "hole_count",
    "colour_pct_necrotic", "colour_pct_chlorotic", "colour_pct_pale_patch",
    "scar_tissue_ratio",
    "miner_trail_coverage_pct", "miner_trail_count",
    "texture_h_glcm_contrast_mean", "texture_h_lbp_entropy",
    "deform_specular_pct", "deform_specular_blob_density",
    "deform_width_profile_roughness", "deform_luminance_std",
    "ldsi_boundary_sub", "ldsi_hole_sub", "ldsi_colour_sub", "ldsi_scar_sub", "ldsi_miner_sub",
    # ADDED (this session): spots.py's discrete lesion features, after
    # fixing two masking bugs that had zeroed them on every leaf (margin-
    # band over-erosion, then rachis-mask contamination -- see spots.py's
    # module docstring FIX notes). validate_spot_features.py confirmed
    # real separation once fixed: thunpath_kurundu effect_size=3.67 on
    # spot_area_ratio, wal_bilin=2.05 -- the strongest signal found
    # anywhere in this feature bank so far for those two species.
    # chlorotic_spot_count deliberately excluded -- confirmed dead
    # (exactly 0.0 for every species at every level, same treatment as
    # DEAD_FEATURES below: chlorosis is diffuse, not discrete, so a spot
    # COUNT of something non-blobby is structurally always going to be
    # zero; colour_pct_chlorotic above is the correct representation for
    # it). spot_rachis_guard_triggered also deliberately excluded here --
    # it's a per-leaf QC flag (did this image's rachis mask look
    # contaminated), not a health signal; see model_training.py's
    # NON_FEATURE_COLS.
    "spot_count", "spot_area_ratio", "spot_density_per_1000px",
    "spot_mean_size", "necrotic_spot_count",
]


def build_fused_frame(top: pd.DataFrame, bottom: pd.DataFrame) -> pd.DataFrame:
    """Worst-of-both-views fusion, same convention as train_health_index.py."""
    feature_cols = [c for c in CANDIDATE_FEATURES if c in top.columns]
    rows = []
    for sample_id in top.index:
        fused = fuse_top_bottom(top.loc[sample_id].to_dict(), bottom.loc[sample_id].to_dict(), feature_cols)
        fused["species"] = top.loc[sample_id, "species"]
        fused["level"] = top.loc[sample_id, "level"]
        fused["leaf_id"] = top.loc[sample_id, "leaf_id"]
        rows.append(fused)
    return pd.DataFrame(rows, index=top.index)


def compute_gap_report(fused: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """One row per (species, feature): group medians + robust effect size.
    feature_cols are BARE feature names; fused carries worst_<feature>
    (from fuse_top_bottom), so we look up the prefixed column but report
    the bare name for readability."""
    rows = []
    for species, grp in fused.groupby("species"):
        healthy = grp[grp["level"] == "healthy"]
        for col in feature_cols:
            fused_col = f"worst_{col}"
            if fused_col not in grp.columns:
                continue
            vals = grp[fused_col].astype(float)
            if vals.isna().all():
                continue
            medians = grp.assign(_v=vals).groupby("level")["_v"].median().reindex(LEVEL_ORDER)

            h_vals = healthy[fused_col].astype(float).dropna()
            if len(h_vals) < 3:
                healthy_iqr = float(np.nanpercentile(vals.dropna(), 75) - np.nanpercentile(vals.dropna(), 25))
            else:
                healthy_iqr = float(np.nanpercentile(h_vals, 75) - np.nanpercentile(h_vals, 25))

            healthy_med = medians.get("healthy", np.nan)
            high_med = medians.get("high", np.nan)
            # FIX (this session): a species with ZERO real variance in its
            # healthy baseline (healthy_iqr == 0, e.g. kathurupila's
            # miner-trail columns) used to have that floored straight to
            # EPS=1e-6, producing millions-magnitude "effect sizes" that
            # are a divide-by-near-zero artifact, not a real measure of
            # separation (kathurupila's ldsi_hole_sub showed 2,590,166.12
            # before this fix). Report effect size as N/A (None) for these
            # instead of fabricating an unbounded number -- the raw
            # healthy_median/high_median columns are still printed as-is,
            # so the actual healthy->high shift (e.g. 30.0 -> 32.59) is
            # still visible, just not divided into a meaningless ratio.
            if healthy_iqr <= EPS:
                effect = np.nan
            else:
                effect = (high_med - healthy_med) / healthy_iqr if pd.notna(healthy_med) and pd.notna(high_med) else np.nan

            rows.append({
                "species": species,
                "feature": col,
                "n_healthy": int(len(h_vals)),
                "n_high": int(grp[grp["level"] == "high"][fused_col].notna().sum()),
                "healthy_median": None if pd.isna(medians.get("healthy")) else round(float(medians["healthy"]), 3),
                "low_median": None if pd.isna(medians.get("low")) else round(float(medians["low"]), 3),
                "mid_median": None if pd.isna(medians.get("mid")) else round(float(medians["mid"]), 3),
                "high_median": None if pd.isna(medians.get("high")) else round(float(medians["high"]), 3),
                "effect_size_high_vs_healthy": None if pd.isna(effect) else round(float(effect), 2),
            })
    return pd.DataFrame(rows)


def compute_elevated_feature_counts(fused: pd.DataFrame, feature_cols: list, z_thresh: float = 1.5) -> pd.DataFrame:
    """
    Per-leaf: how many candidate features sit >= z_thresh robust-z away
    from THAT species' own healthy baseline. Tests the single-dominant-
    symptom-dilution hypothesis directly: if 'high' leaves mostly show
    n_elevated == 1, an averaging-style score will systematically
    underscore them regardless of which specific feature fired.
    feature_cols are bare names; looked up as worst_<feature> in fused.
    """
    present_cols = [c for c in feature_cols if f"worst_{c}" in fused.columns]
    stats = {}
    for species, grp in fused.groupby("species"):
        healthy = grp[grp["level"] == "healthy"]
        stats[species] = {}
        for col in present_cols:
            fused_col = f"worst_{col}"
            vals = healthy[fused_col].astype(float).dropna()
            if len(vals) < 3:
                med, iqr = np.nan, np.nan
            else:
                med = float(np.median(vals))
                raw_iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                # FIX (this session): same zero-variance guard as
                # train_stage1_binary.py's compute_species_baselines --
                # a species with ZERO real spread in its healthy leaves
                # on this feature should never produce an "elevated"
                # verdict from a fabricated near-infinite z-score. Treat
                # as unbaselineable (NaN) instead, same as <3 healthy
                # leaves.
                if raw_iqr <= EPS:
                    med, iqr = np.nan, np.nan
                else:
                    iqr = raw_iqr
            stats[species][col] = (med, iqr)

    counts = []
    for _, row in fused.iterrows():
        species = row["species"]
        n_elevated = 0
        for col in present_cols:
            fused_col = f"worst_{col}"
            med, iqr = stats[species].get(col, (np.nan, np.nan))
            if pd.isna(med) or pd.isna(row.get(fused_col)):
                continue
            z = abs((float(row[fused_col]) - med) / iqr)
            if z >= z_thresh:
                n_elevated += 1
        counts.append(n_elevated)
    fused = fused.copy()
    fused["n_features_elevated"] = counts
    return fused[["leaf_id", "species", "level", "n_features_elevated"]]


def main():
    parser = argparse.ArgumentParser(description="Per-species feature-gap diagnostic (no images needed)")
    parser.add_argument("--top-csv", default=TOP_CSV)
    parser.add_argument("--bottom-csv", default=BOTTOM_CSV)
    parser.add_argument("--out", default="gap_report.csv")
    parser.add_argument("--elevated-out", default="elevated_feature_counts.csv")
    parser.add_argument("--z-thresh", type=float, default=1.5)
    args = parser.parse_args()

    top, bottom = load_and_merge_from(args.top_csv, args.bottom_csv)
    top = top[top["variant_id"] == 0]
    bottom = bottom.loc[bottom.index.isin(top.index)]

    fused = build_fused_frame(top, bottom)
    feature_cols = [c for c in CANDIDATE_FEATURES if f"worst_{c}" in fused.columns]

    gap_report = compute_gap_report(fused, feature_cols)
    gap_report = gap_report.sort_values(
        "effect_size_high_vs_healthy", key=lambda s: s.abs(), na_position="first"
    )
    gap_report.to_csv(args.out, index=False)
    print(f"[done] wrote {len(gap_report)} rows -> {Path(args.out).resolve()}")
    print("\n[worst 20 species/feature pairs -- smallest |effect size|, i.e. most 'blind']:")
    print(gap_report.head(20).to_string(index=False))

    print("\n[per-species: how many features show |effect size| < 0.5 (essentially no separation)?]")
    weak = gap_report[gap_report["effect_size_high_vs_healthy"].abs() < 0.5]
    print(weak.groupby("species").size().sort_values(ascending=False).to_string())

    elevated = compute_elevated_feature_counts(fused, feature_cols, z_thresh=args.z_thresh)
    elevated.to_csv(args.elevated_out, index=False)
    print(f"\n[done] wrote per-leaf elevated-feature counts -> {Path(args.elevated_out).resolve()}")
    print("\n[distribution of n_features_elevated by level -- tests the "
          "single-dominant-symptom dilution hypothesis]:")
    print(elevated.groupby("level")["n_features_elevated"].describe().reindex(LEVEL_ORDER).to_string())
    print("\n-> if 'high' leaves mostly show n_features_elevated around 1 (not several), "
          "that confirms damage is usually concentrated in a single feature group per leaf, "
          "and any averaging-style aggregation will systematically underscore them regardless "
          "of which specific feature fired.")


if __name__ == "__main__":
    main()
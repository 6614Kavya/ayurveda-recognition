"""
feature_extraction/health/health_index.py

VedaVision Health Index (VHI) -- replacement for the Stage-2 low/mid/high
classifier.

Computes a continuous, explainable severity score from the EXISTING
ldsi_*_sub columns already produced by
feature_extraction/health/severity_index.py (ldsi_boundary_sub,
ldsi_hole_sub, ldsi_colour_sub, ldsi_scar_sub), plus two coarse
"distribution" proxies (hole_count, boundary_notch_count) that were
already being extracted but never used as severity signal directly.

Nothing here requires new feature extraction to run -- it operates on
the CSVs you already have.

Why NOT hand-picked weights (e.g. 0.4*colour + 0.2*boundary + ...):
a viva panel will ask "why 0.4 and not 0.35?" and there is no biological
answer. Instead, weights are fit with a non-negative Ridge regression
against a label-derived proxy score, so every weight can be traced back
to "how well does this sub-score track the existing low/mid/high
labels", not to a guess.

Why NOT judged by classification accuracy:
your own confusion matrix shows "mid" confused with BOTH "low" and
"high" -- the signature of a forced discretization of a continuum, not
three separable clusters. The right metric for a continuum validated
against noisy discrete labels is monotonicity (Spearman rho, Kruskal-
Wallis across groups), not F1/accuracy. This module's
validate_monotonicity() implements that.

--- FIX (this session): severity-ordinal target retired as primary ---
fit_health_index() (fitting weights against LEVEL_PROXY_SCORE /
SEVERITY_PROXY_SCORE, a healthy->low->mid->high continuum) was run
against real labeled data and failed: per-level group medians overlap
substantially between low/mid/high (confirmed via validate_monotonicity's
group_medians output and bootstrap_ci.py's healthy-vs-low gap check).
This is NOT treated here as "the weights need re-tuning" -- overlapping
medians after a proper non-negative Ridge fit are evidence that the
low/mid/high boundary, as currently labeled, is not a signal the
extracted handcrafted features can separate. Continuing to fit an
ordinal target against a boundary that isn't really there just fits
noise into the weights.

fit_health_index_binary() below is the new primary method: labels are
collapsed to healthy (0) vs. unhealthy (1) -- low/mid/high all become
"unhealthy" -- and weights are fit against that strictly easier, more
likely genuine binary split. The species-baseline normalization
(fit_species_norm_stats / apply_species_norm) is UNCHANGED and reused
as-is, since that part of the pipeline was never implicated in the
overlap failure -- it only touches healthy-class statistics and was
never fit against low/mid/high at all.

The resulting HealthIndexModel still produces a CONTINUOUS 0-100 score
via .score() for every leaf, healthy or unhealthy -- this is what
supplies the actual "how far from this leaf's species' healthy
baseline" value for unhealthy leaves. What changes is only what target
the weights are fit against; unhealthy leaves are never bucketed back
into low/mid/high by this model, and no claim is made that the index
can rank low vs. mid vs. high reliably. validate_binary_separation()
replaces validate_monotonicity() as the primary validation metric;
per-original-label group medians are still reported afterward, but as
descriptive information only, not as a pass/fail separation claim.

fit_health_index() and validate_monotonicity() are kept, unmodified, as
documented negative-result evidence for the dissertation -- do not
delete them, and do not present their output as the reported method.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, kruskal, mannwhitneyu

# These are the worst_<feature> columns emitted by
# models.health.classifier.fuse_top_bottom() (worst-side-wins, matching
# the dataset's own worst-side ground-truth labeling convention). If you
# fuse differently, adjust the prefix here, not the logic below.
#
# FIX (this session): worst_ldsi_colour_sub REMOVED in favour of the raw
# worst_colour_pct_* components. Empirically, ldsi_colour_sub is an
# unweighted average of necrotic/chlorotic/pale_patch (severity_index.py
# _sub_index), and colour_pct_necrotic carries ~zero severity signal
# (Spearman rho=0.019, p=0.61, on unhealthy-only leaves -- still
# uncalibrated per colour_health.py's own docstring), while
# colour_pct_chlorotic carries real signal alone (rho=0.168, p<0.0001).
# Averaging the two together diluted the real signal down to rho=0.044
# (not significant). Exposing the three raw components separately lets
# Ridge assign them independent weights -- and correctly push necrotic's
# weight toward zero -- instead of baking a bad average in upstream.
#
# FIX (this session, round 2): worst_ldsi_hole_sub, worst_ldsi_scar_sub,
# and worst_boundary_notch_count all received EXACTLY 0.0 Ridge weight in
# the fit against real data -- confirmed not a fluke of one run (weights
# were reproduced identically across a non-augmented and an augmented
# feature-extraction pass). Dropping them frees up Ridge's degrees of
# freedom on the same ~582 unhealthy training leaves without discarding
# any signal (worst_hole_count -- the raw count, not the sub-index
# average -- is kept, since IT carries real signal: rho=0.225,
# p<0.0001, and got real weight). If reinstating any of these three for
# a future ablation comparison, do it explicitly, not by default.
# FIX (this session, round 3): worst_texture_h_glcm_contrast_mean and
# worst_texture_h_lbp_entropy added as candidates -- from the new
# texture_health.py group. Not yet Cliff's-delta/Spearman-validated on
# real data (no dataset run with these columns exists yet at the time of
# this edit) -- treat their learned weights in the NEXT run as the first
# real diagnostic, same as every other feature group before it. If either
# lands near-zero weight like worst_ldsi_hole_sub/scar_sub did, drop it
# in the next trim pass rather than assuming it's automatically useful
# just because it's new.
# FIX (this session, round 4): worst_colour_pct_necrotic REMOVED.
# colour_diagnostic.py showed its group median is flat ~2.1-2.3% across
# EVERY severity level, including healthy leaves (2.29% healthy vs 2.13%
# high) -- not a miscalibrated threshold, but very likely a fixed
# structural artifact (vein/rachis shadow) unrelated to actual necrosis.
# Consistent with the rest of the winning feature set (hole_count,
# pale_patch, boundary/margin roughness, miner_trail) pointing to
# insect-feeding damage + desiccation as the dominant visible damage mode
# in this dataset, not fungal/bacterial necrotic lesions -- recalibrating
# this gate's threshold is unlikely to create signal that isn't present.
#
# worst_colour_lab_a_iqr ADDED: previously extracted but never fed into
# the index. Real signal (rho=0.133, p<0.0001 vs full healthy->high
# ordinal) that the flat percentage features miss -- within-leaf colour
# SPREAD increases with damage even though the whole-leaf median barely
# moves (damage is patchy/localised, so a median dilutes it; IQR catches
# the resulting heterogeneity instead).
# REVERTED (this session, round 5): worst_colour_lab_a_iqr tested and
# removed. Real univariate signal alone (rho=0.132, p<0.0001) but Ridge
# gave it exactly 0.0 weight once fit alongside the other columns --
# its information is redundant with what's already captured elsewhere.
# Worse: its mere PRESENCE as a candidate measurably redistributed the
# OTHER weights (pale_patch dropped 0.207->0.145) and shrank the
# healthy-vs-low group median gap from 5.07 to 0.67 -- a real
# qualitative regression in exactly the separation that matters most,
# even though the aggregate test rho (0.249->0.229) looked like noise-
# range movement on its own. Reverting to the prior 7-column set.
# FIX (this session): worst_deform_specular_pct, worst_deform_specular_blob_density,
# and worst_deform_width_profile_roughness/worst_deform_luminance_std added as
# candidates from the new deformation.py group -- targets leaf curling/
# non-planarity, a symptom none of the other six feature groups measure.
# NOT yet Spearman/Ridge-validated on real data at the time of this edit
# (added after spot-checking visibly-curled leaves that scored near-
# healthy on the existing feature set). Treat their learned weights in
# the NEXT run as the first real diagnostic, same as every other feature
# group before it -- if any lands near-zero weight like
# worst_ldsi_hole_sub/scar_sub or worst_texture_h_lbp_entropy did, drop it
# in the next trim pass rather than assuming it's automatically useful
# just because it targets a real, confirmed gap. On the 3 real photos
# checked during development, deform_specular_pct measured exactly 0.0
# for all three -- possibly not informative under this dataset's
# lighting; deform_width_profile_roughness and deform_luminance_std both
# showed real spread and are the more promising of the four.
SUBSCORE_RAW_COLUMNS: List[str] = [
    "worst_ldsi_boundary_sub",
    "worst_colour_pct_chlorotic",
    "worst_colour_pct_pale_patch",
    "worst_ldsi_miner_sub",
    "worst_hole_count",           # distribution proxy: # of damaged regions
    "worst_texture_h_glcm_contrast_mean",
    "worst_texture_h_lbp_entropy",
    # REMOVED (this session): worst_deform_specular_pct and
    # worst_deform_width_profile_roughness. Checked per_subscore_correlation()
    # against real fitted data (not assumed from the docstring note above,
    # which pre-dated this check): specular_pct rho=0.008 p=0.849 (not
    # significant, matches gap_report.csv showing it's exactly 0.0 for
    # almost every species/level); width_profile_roughness rho=0.000
    # p=0.999 -- exactly the "essentially uncorrelated with severity"
    # result train_stage1_binary.py's docstring already flagged
    # independently. Despite that, the binary-target Ridge fit was still
    # assigning width_profile_roughness a weight of 0.12 -- the THIRD
    # largest of 13 -- which is Ridge fitting noise, not signal (a
    # non-negative-constrained Ridge can still assign real weight to a
    # column with p=0.999 if it happens to correlate with training-set
    # residuals by chance). Removed rather than trusting the Ridge weight
    # over the univariate check.
    #
    # worst_deform_luminance_std KEPT for now despite p=0.232 (n.s.) --
    # unlike the two removed above, Ridge already assigns it exactly 0.0
    # weight on its own, so its presence isn't distorting anything; no
    # need to force a removal that changes nothing. Revisit only if a
    # future refit gives it nonzero weight without a matching significant
    # rho.
    #
    # worst_ldsi_miner_sub KEPT -- do NOT copy train_stage1_binary.py's
    # DEAD_FEATURES exclusion here without checking. That exclusion is
    # from a per-SPECIES z-scored context where miner_trail's false-
    # positive rate on several species swamped its genuine signal on
    # kalawal/kattakumanjal. Checked separately here, in this model's own
    # whole-dataset context: rho=0.163, p=6.2e-05 (real, significant) and
    # Ridge weight=0.056. The two models use this feature differently --
    # exclude it from the classifier's species-relative z-features
    # (already done) without also exclude it here just because the name
    # matches.
    "worst_deform_luminance_std",
    "worst_spot_count",
    "worst_spot_area_ratio",
    "worst_spot_density_per_1000px",
]

# Proxy numeric targets used ONLY to orient the index's direction/scale
# when fitting weights. NOT treated as ground truth -- the real evidence
# for whether the index works is validate_monotonicity(), not how
# closely it reproduces these specific numbers. Document this choice
# explicitly in the dissertation.
LEVEL_PROXY_SCORE: Dict[str, float] = {"healthy": 5.0, "low": 35.0, "mid": 60.0, "high": 85.0}

# FIX (this session): a severity-ONLY proxy, used when fitting on
# unhealthy leaves alone (fit_on_unhealthy_only=True, now the default).
# Without this, Ridge was fit against the full healthy->high span, which
# is dominated by the easy healthy-vs-unhealthy gap (already solved by
# Stage 1) -- so learned weights reflected THAT split, not true
# low/mid/high separation. Evidence: worst_ldsi_colour_sub got weight
# 0.193 despite ~0 raw correlation with severity order among unhealthy
# leaves alone (rho=0.044, p=0.24) -- it was still doing useful work
# separating healthy from unhealthy, just not the job this index is for.
SEVERITY_PROXY_SCORE: Dict[str, float] = {"low": 20.0, "mid": 55.0, "high": 90.0}

# FIX (this session): binary proxy target for fit_health_index_binary().
# low/mid/high all collapse to the same "unhealthy" target -- this
# module makes no claim about separating them, only about how far a
# leaf sits from its species' own healthy baseline.
BINARY_PROXY_SCORE: Dict[str, float] = {"healthy": 0.0, "low": 1.0, "mid": 1.0, "high": 1.0}

DEFAULT_LEVEL_ORDER: Dict[str, int] = {"healthy": 0, "low": 1, "mid": 2, "high": 3}
EPS = 1e-6


@dataclass
class SpeciesNormStats:
    """Per-species healthy-class median/IQR for each sub-score column.
    Strips the species baseline confound identified empirically
    (specnorm.py finding: variance ratio for e.g. scar_tissue_ratio
    dropped from 61x to 8.5x after this normalization). Falls back to a
    global median/IQR for species with too few healthy leaves to fit a
    stable per-species baseline.

    sign_corrections (added this session): +1.0/-1.0 per (species, col).
    A single global Ridge weight assumes every column's "higher z-score
    -> more damage" direction is the same across all species -- checked
    against real per-species Spearman rho and it isn't. Example:
    worst_hole_count correlates POSITIVELY with severity in most species
    (more holes = more damage, matching the global weight's sign) but
    NEGATIVELY for siyambala specifically (rho=-0.624) -- siyambala's
    healthy leaves are naturally holier than its damaged ones. The
    global weight, forced to pick one sign, actively works backwards for
    that species. -1.0 flips that species' z-score for that column
    before weighting; default 1.0 (trust the global sign) everywhere
    else. Fit from TRAIN data only, see fit_sign_corrections()."""
    median: Dict[str, Dict[str, float]] = field(default_factory=dict)
    iqr: Dict[str, Dict[str, float]] = field(default_factory=dict)
    global_median: Dict[str, float] = field(default_factory=dict)
    global_iqr: Dict[str, float] = field(default_factory=dict)
    sign_corrections: Dict[str, Dict[str, float]] = field(default_factory=dict)
    min_healthy_n: int = 5


def fit_sign_corrections(
    df: pd.DataFrame,
    columns: List[str],
    species_col: str = "species",
    level_col: str = "level",
    min_n: int = 15,
    rho_threshold: float = 0.3,
    p_threshold: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """For each (species, column), checks whether that species' OWN train
    leaves show a raw-feature-vs-severity relationship whose SIGN
    disagrees with the column's global (all-species-pooled) sign. Only
    flips (-1.0) when a species has enough leaves (>=min_n) for a stable
    estimate AND its own correlation clears rho_threshold/p_threshold AND
    is actually stronger than the global correlation -- a species that's
    merely noisier than average shouldn't get a flip just from sampling
    variation; this requires it to show a REAL, opposite, and dominant
    relationship. Everyone else defaults to +1.0 (trust the pooled sign).

    Tested via sign_fix_prototype.py against the held-out test split
    before being wired in here: overall test ROC-AUC 0.732 -> 0.804,
    with the previously worst-performing species (siyambala, which was
    AUC=0.32 -- worse than random, because worst_hole_count's global
    sign is backwards for it) improving to 0.64. Not perfect for every
    species (kattakumanjal, nil_awariya, siyambala still sit in the
    0.60-0.68 range) -- report per-species AUC in the dissertation
    rather than only the pooled number, same as before this fix.
    """
    ordinal = df[level_col].map(LEVEL_PROXY_SCORE).values
    corrections: Dict[str, Dict[str, float]] = {}
    for col in columns:
        global_rho, _ = spearmanr(ordinal, df[col].astype(float))
        for species, grp in df.groupby(species_col):
            corrections.setdefault(species, {})
            if len(grp) < min_n:
                corrections[species][col] = 1.0
                continue
            grp_ordinal = grp[level_col].map(LEVEL_PROXY_SCORE).values
            rho, p = spearmanr(grp_ordinal, grp[col].astype(float))
            if (
                p < p_threshold
                and abs(rho) >= rho_threshold
                and np.sign(rho) != np.sign(global_rho)
                and abs(rho) > abs(global_rho)
            ):
                corrections[species][col] = -1.0
            else:
                corrections[species][col] = 1.0
    return corrections


def fit_species_norm_stats(
    df: pd.DataFrame,
    columns: List[str],
    species_col: str = "species",
    level_col: str = "level",
    min_healthy_n: int = 5,
    sign_correct: bool = False,
) -> SpeciesNormStats:
    stats = SpeciesNormStats(min_healthy_n=min_healthy_n)
    healthy = df[df[level_col] == "healthy"]

    for col in columns:
        vals = df[col].astype(float)
        stats.global_median[col] = float(np.nanmedian(vals))
        q75, q25 = np.nanpercentile(vals, 75), np.nanpercentile(vals, 25)
        stats.global_iqr[col] = float(max(q75 - q25, EPS))

    for species, grp in healthy.groupby(species_col):
        stats.median[species] = {}
        stats.iqr[species] = {}
        for col in columns:
            vals = grp[col].astype(float).dropna()
            if len(vals) < min_healthy_n:
                stats.median[species][col] = stats.global_median[col]
                stats.iqr[species][col] = stats.global_iqr[col]
                continue
            med = float(vals.median())
            q75, q25 = np.nanpercentile(vals, 75), np.nanpercentile(vals, 25)
            # Floor relative to the column's GLOBAL iqr, not just EPS.
            # Some species have iqr==0 for discrete/count columns (e.g.
            # boundary_notch_count) among healthy leaves -- without this
            # floor, dividing by ~0 makes that one column's z-score
            # explode and dominate the Ridge fit entirely (observed:
            # weight 0.999 on a single feature, all others ~0).
            floor = max(0.25 * stats.global_iqr[col], EPS)
            stats.median[species][col] = med
            stats.iqr[species][col] = float(max(q75 - q25, floor))

    if sign_correct:
        # Needs the FULL df (all levels, not just the healthy subset
        # above) to compute per-species severity correlation direction.
        stats.sign_corrections = fit_sign_corrections(df, columns, species_col, level_col)
    return stats


def apply_species_norm(
    df: pd.DataFrame,
    columns: List[str],
    stats: SpeciesNormStats,
    species_col: str = "species",
) -> pd.DataFrame:
    """Robust z-score of each column against that row's species' own
    healthy-class median/IQR. Missing values fall back to 0 (i.e.
    "assumed at that species' healthy baseline") rather than silently
    corrupting the weighted sum.

    If stats.sign_corrections is non-empty (fit_species_norm_stats was
    called with sign_correct=True), each column is also multiplied by
    that row's species' +1.0/-1.0 correction before the clip -- see
    SpeciesNormStats' docstring for why a single global sign can be
    backwards for a specific species."""
    out = pd.DataFrame(index=df.index)
    for col in columns:
        med = df[species_col].map(lambda s: stats.median.get(s, {}).get(col, stats.global_median[col]))
        iqr = df[species_col].map(lambda s: stats.iqr.get(s, {}).get(col, stats.global_iqr[col]))
        z = (df[col].astype(float) - med) / iqr
        if stats.sign_corrections:
            sign = df[species_col].map(lambda s: stats.sign_corrections.get(s, {}).get(col, 1.0))
            z = z * sign
        out[col] = z
    # Second safety net: even with the iqr floor above, cap extreme
    # z-scores so no single leaf/column can dominate the weighted sum
    # or the Ridge fit.
    return out.fillna(0.0).clip(lower=-5.0, upper=5.0)


@dataclass
class HealthIndexModel:
    subscore_columns: List[str]
    weights: np.ndarray  # non-negative, sums to 1
    index_min: float
    index_max: float
    species_stats: SpeciesNormStats
    # ADDED (this session): optional PER-SPECIES 0-100 scaling anchors.
    # index_min/index_max above are pooled across every species -- fine
    # for rank-ordering (ROC-AUC is invariant to this either way, since
    # it's a monotonic transform), but it means the ABSOLUTE number isn't
    # comparable in the way you'd want: species differ a lot in how far
    # their worst leaves actually deviate (train raw-score range spanned
    # from ~1.0 for ranawara's worst leaf to ~3.3 for
    # maha_undupiyaliya's), so with one shared scale, ranawara's worst
    # leaf -- maximally bad FOR RANAWARA -- lands nowhere near severity
    # 100, while an equally-maximally-bad maha_undupiyaliya leaf does.
    # When these two dicts are non-empty, score()/score_breakdown()
    # anchor each species to ITS OWN observed train range instead: that
    # species' own healthy leaves' median raw score -> severity ~0
    # (health_value ~100), that species' own worst observed train leaf ->
    # severity 100. Species with too few train leaves to fit a stable
    # anchor (see fit_health_index_binary's per_species_scale_min_n)
    # aren't in either dict and fall back to the pooled index_min/
    # index_max above -- same species-level fallback pattern already
    # used in SpeciesNormStats for species with too few healthy leaves.
    species_index_min: Dict[str, float] = field(default_factory=dict)
    species_index_max: Dict[str, float] = field(default_factory=dict)

    def _bounds_for(self, species: str):
        lo = self.species_index_min.get(species, self.index_min)
        hi = self.species_index_max.get(species, self.index_max)
        return lo, hi

    def score(self, df: pd.DataFrame, species_col: str = "species") -> np.ndarray:
        z = apply_species_norm(df, self.subscore_columns, self.species_stats, species_col)
        raw = z.values @ self.weights
        if self.species_index_min:
            lo = df[species_col].map(lambda s: self.species_index_min.get(s, self.index_min)).values
            hi = df[species_col].map(lambda s: self.species_index_max.get(s, self.index_max)).values
            scaled = 100.0 * (raw - lo) / np.maximum(hi - lo, EPS)
        else:
            scaled = 100.0 * (raw - self.index_min) / max(self.index_max - self.index_min, EPS)
        return np.clip(scaled, 0.0, 100.0)

    def score_breakdown(self, row: pd.Series, species_col: str = "species") -> Dict[str, float]:
        """Per-subscore contribution for a single leaf -- what powers the
        explainable report (e.g. 'Colour Damage: 71%').

        FIX (this session): contributions are now clipped to >=0 before
        computing percentage shares. The previous version divided by the
        raw signed sum, which could include NEGATIVE contributions (a
        subscore below that species' typical baseline -- i.e. healthier
        than usual for that species). When contributions had mixed signs
        and the raw total happened to be small, this produced nonsensical
        output on real leaves (observed: 156.6%, -109.9%, -4.3% on a
        single test row) -- not a data artifact, a formula flaw. A
        subscore that's better than typical for its species should
        contribute 0% to "what's driving THIS leaf's damage," not a
        negative or inflated share.

        NOTE: unaffected by species_index_min/max above -- this is a
        share WITHIN one leaf's own contributions (they sum to 100%
        regardless of the final severity-score anchor), not a score on
        the 0-100 scale itself.
        """
        one = pd.DataFrame([row])
        z = apply_species_norm(one, self.subscore_columns, self.species_stats, species_col).iloc[0]
        contributions = np.clip(z.values * self.weights, 0.0, None)
        total = contributions.sum()
        breakdown = {}
        for col, contrib in zip(self.subscore_columns, contributions):
            pct = 0.0 if total < EPS else float(100.0 * contrib / total)
            breakdown[col] = round(pct, 1)
        return breakdown


def fit_health_index(
    df: pd.DataFrame,
    subscore_columns: List[str] = SUBSCORE_RAW_COLUMNS,
    species_col: str = "species",
    level_col: str = "level",
    fit_on_unhealthy_only: bool = True,
) -> HealthIndexModel:
    """
    fit_on_unhealthy_only (default True, FIX this session): restrict the
    Ridge fit to unhealthy (low/mid/high) leaves only, against
    SEVERITY_PROXY_SCORE, instead of all leaves against the full
    LEVEL_PROXY_SCORE (healthy->high) span. Species-normalisation stats
    (fit_species_norm_stats) still use the FULL dataframe, since those
    need the healthy-class distribution as the baseline to normalise
    against -- only the WEIGHT-FITTING step changes scope.

    Set False to reproduce the original (pre-fix) behaviour for an
    ablation comparison in the dissertation.
    """
    stats = fit_species_norm_stats(df, subscore_columns, species_col, level_col)
    z_full = apply_species_norm(df, subscore_columns, stats, species_col)

    if fit_on_unhealthy_only:
        fit_mask = (df[level_col] != "healthy").values
        proxy_map = SEVERITY_PROXY_SCORE
    else:
        fit_mask = np.ones(len(df), dtype=bool)
        proxy_map = LEVEL_PROXY_SCORE

    z_fit = z_full.loc[fit_mask]
    y_proxy = df.loc[fit_mask, level_col].map(proxy_map).values.astype(float)

    ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False)
    ridge.fit(z_fit.values, y_proxy)

    coef = np.clip(ridge.coef_, 0.0, None)
    if coef.sum() <= EPS:
        coef = np.ones(len(subscore_columns))  # degenerate fallback: equal weights
    weights = coef / coef.sum()

    raw_scores_fit = z_fit.values @ weights
    return HealthIndexModel(
        subscore_columns=subscore_columns,
        weights=weights,
        index_min=float(raw_scores_fit.min()),
        index_max=float(raw_scores_fit.max()),
        species_stats=stats,
    )


def fit_health_index_binary(
    df: pd.DataFrame,
    subscore_columns: List[str] = SUBSCORE_RAW_COLUMNS,
    species_col: str = "species",
    level_col: str = "level",
    sign_correct: bool = False,
    per_species_scale: bool = False,
    per_species_scale_min_n: int = 15,
) -> HealthIndexModel:
    """
    PRIMARY method (replaces fit_health_index() -- see module docstring
    for why). Collapses low/mid/high into a single "unhealthy" class and
    fits non-negative Ridge weights against that binary split instead of
    the healthy->low->mid->high ordinal, which real data showed is not
    cleanly separable by these features.

    Species-baseline normalization is unchanged from fit_health_index():
    fit_species_norm_stats() still uses the full dataframe's healthy
    rows only, since that step was never implicated in the low/mid/high
    overlap failure.

    sign_correct (added this session, default False for backward
    compatibility -- train_health_index.py passes True): also fits
    per-species sign corrections (fit_sign_corrections) so a column
    whose real relationship to severity flips sign for one species
    (e.g. worst_hole_count is negatively correlated with severity for
    siyambala, positively for most others) doesn't get forced into the
    global sign for every species. Tested on held-out data: overall
    test ROC-AUC 0.732 -> 0.804. See SpeciesNormStats' docstring for
    the full reasoning and per-species before/after numbers.

    Returns the same HealthIndexModel type as fit_health_index(), so
    .score() and .score_breakdown() work identically downstream. The
    only difference is what the weights were fit against. Unhealthy
    leaves get a continuous 0-100 deviation-from-baseline score, not a
    low/mid/high bucket -- do not reinterpret this score's absolute
    position as a severity-tier claim.

    per_species_scale (added this session, default False): anchor the
    final 0-100 scale per species instead of pooling every species into
    one shared range -- see HealthIndexModel.species_index_min/max's
    docstring for why the pooled range under/over-states severity for
    species whose worst leaves simply don't deviate as far in raw terms.
    Does NOT change ROC-AUC/ranking (a monotonic per-group rescale can't
    change within-species rank order) -- this is purely about making the
    absolute number mean the same thing ("how bad for THIS species")
    across species, not about separating healthy from unhealthy any
    better. Species with fewer than per_species_scale_min_n train leaves
    don't get their own anchor and fall back to the pooled range.
    """
    stats = fit_species_norm_stats(df, subscore_columns, species_col, level_col, sign_correct=sign_correct)
    z_full = apply_species_norm(df, subscore_columns, stats, species_col)

    y_binary = df[level_col].map(BINARY_PROXY_SCORE).values.astype(float)
    if np.isnan(y_binary).any():
        bad = df.loc[np.isnan(y_binary), level_col].unique().tolist()
        raise ValueError(
            f"Unrecognised level value(s) {bad} not in BINARY_PROXY_SCORE "
            f"{list(BINARY_PROXY_SCORE.keys())}. Fix the label or extend "
            f"BINARY_PROXY_SCORE before fitting."
        )

    ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False)
    ridge.fit(z_full.values, y_binary)

    coef = np.clip(ridge.coef_, 0.0, None)
    if coef.sum() <= EPS:
        coef = np.ones(len(subscore_columns))  # degenerate fallback: equal weights
    weights = coef / coef.sum()

    raw_scores_full = z_full.values @ weights

    species_index_min = {}
    species_index_max = {}
    if per_species_scale:
        raw_by_row = pd.Series(raw_scores_full, index=df.index)
        for species, grp in df.groupby(species_col):
            if len(grp) < per_species_scale_min_n:
                continue  # falls back to the pooled index_min/index_max at score() time
            grp_raw = raw_by_row.loc[grp.index]
            healthy_mask = grp[level_col] == "healthy"
            if healthy_mask.sum() == 0:
                continue  # can't anchor "healthy" for a species with no healthy train leaves
            species_index_min[species] = float(grp_raw[healthy_mask].median())
            species_index_max[species] = float(grp_raw.max())

    return HealthIndexModel(
        subscore_columns=subscore_columns,
        weights=weights,
        index_min=float(raw_scores_full.min()),
        index_max=float(raw_scores_full.max()),
        species_stats=stats,
        species_index_min=species_index_min,
        species_index_max=species_index_max,
    )


def validate_binary_separation(
    index_scores: np.ndarray,
    levels: pd.Series,
) -> dict:
    """
    PRIMARY validation metric for fit_health_index_binary() (replaces
    validate_monotonicity() as the headline number -- that function is
    kept for the negative-result writeup, not deleted).

    Reports:
      - roc_auc: healthy vs. unhealthy (low/mid/high combined) separation
        by the raw index score. This is the real claim -- report this
        number in the dissertation, not a low/mid/high accuracy figure.
      - cliffs_delta: healthy vs. unhealthy effect size (rank-biserial,
        derived from Mann-Whitney U), -1..1, magnitude is what matters
        (>=0.474 is conventionally "large").
      - mannwhitney_p: significance of the healthy-vs-unhealthy gap.
      - group_medians: descriptive median score per ORIGINAL label
        (healthy/low/mid/high), reported for transparency -- this is
        informational only. A trend across low/mid/high here is a nice
        bonus if it appears, but its ABSENCE does not invalidate the
        healthy-vs-unhealthy result above, and its presence should not be
        oversold as proven separation without also reporting group
        overlap (e.g. IQR overlap) alongside the medians.
    """
    is_unhealthy = (levels != "healthy").values
    healthy_scores = index_scores[~is_unhealthy]
    unhealthy_scores = index_scores[is_unhealthy]

    auc = float("nan")
    if len(healthy_scores) > 0 and len(unhealthy_scores) > 0:
        y_true = np.concatenate([np.zeros(len(healthy_scores)), np.ones(len(unhealthy_scores))])
        y_score = np.concatenate([healthy_scores, unhealthy_scores])
        auc = float(roc_auc_score(y_true, y_score))

    delta = float("nan")
    mw_p = float("nan")
    if len(healthy_scores) > 0 and len(unhealthy_scores) > 0:
        u_stat, mw_p = mannwhitneyu(unhealthy_scores, healthy_scores, alternative="two-sided")
        # rank-biserial / Cliff's delta from Mann-Whitney U:
        # delta = 2U/(n1*n2) - 1, oriented so positive = unhealthy scores
        # higher than healthy (matches this module's severity-direction
        # convention: higher score = more deviation from healthy baseline).
        n1, n2 = len(unhealthy_scores), len(healthy_scores)
        delta = float(2.0 * u_stat / (n1 * n2) - 1.0)
        mw_p = float(mw_p)

    medians = (
        pd.DataFrame({"level": levels.values, "index": index_scores})
        .groupby("level")["index"].median()
        .reindex(sorted(DEFAULT_LEVEL_ORDER, key=DEFAULT_LEVEL_ORDER.get))
    )
    iqrs = (
        pd.DataFrame({"level": levels.values, "index": index_scores})
        .groupby("level")["index"]
        .apply(lambda s: float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25)) if len(s) > 0 else None)
        .reindex(sorted(DEFAULT_LEVEL_ORDER, key=DEFAULT_LEVEL_ORDER.get))
    )

    return {
        "n_healthy": int(len(healthy_scores)),
        "n_unhealthy": int(len(unhealthy_scores)),
        "roc_auc_healthy_vs_unhealthy": auc,
        "cliffs_delta_healthy_vs_unhealthy": delta,
        "mannwhitney_p": mw_p,
        "group_medians_descriptive_only": {
            k: (None if pd.isna(v) else round(float(v), 2)) for k, v in medians.to_dict().items()
        },
        "group_iqrs_descriptive_only": {
            k: (None if v is None or pd.isna(v) else round(float(v), 2)) for k, v in iqrs.to_dict().items()
        },
    }


def validate_monotonicity(
    index_scores: np.ndarray,
    levels: pd.Series,
    level_order: Optional[Dict[str, int]] = None,
) -> dict:
    """The evaluation metric for this approach. Replaces classification
    F1/accuracy on low/mid/high with: does the index rise, ON AVERAGE,
    across healthy -> low -> mid -> high?"""
    if level_order is None:
        level_order = DEFAULT_LEVEL_ORDER
    ordinal = levels.map(level_order).values

    rho, rho_p = spearmanr(ordinal, index_scores)

    groups = [index_scores[ordinal == v] for v in sorted(set(ordinal))]
    groups = [g for g in groups if len(g) > 0]
    kw_stat, kw_p = kruskal(*groups) if len(groups) > 1 else (float("nan"), float("nan"))

    medians = (
        pd.DataFrame({"level": levels.values, "index": index_scores})
        .groupby("level")["index"].median()
        .reindex(sorted(level_order, key=level_order.get))
    )

    return {
        "n": int(len(index_scores)),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kruskal_stat": float(kw_stat),
        "kruskal_p": float(kw_p),
        "group_medians": {k: (None if pd.isna(v) else round(float(v), 2)) for k, v in medians.to_dict().items()},
    }


def per_subscore_correlation(
    df: pd.DataFrame,
    subscore_columns: List[str],
    level_col: str = "level",
    level_order: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Diagnostic: Spearman rho of each RAW sub-score (before weighting)
    against severity order -- shows which of the sub-scores is actually
    carrying signal vs. dead weight, mirroring the earlier Cliff's-delta
    finding that only colour_pct_chlorotic carried real signal among the
    whole-leaf colour statistics."""
    if level_order is None:
        level_order = DEFAULT_LEVEL_ORDER
    ordinal = df[level_col].map(level_order).values
    rows = []
    for col in subscore_columns:
        rho, p = spearmanr(ordinal, df[col].astype(float).values)
        rows.append({"subscore": col, "spearman_rho": round(float(rho), 3), "p_value": p})
    return pd.DataFrame(rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
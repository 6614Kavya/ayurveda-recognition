# Vedavision Project Memory

*Last verified against actual codebase (`module_3.zip`) — supersedes the interim report and all earlier memory notes wherever they conflict.*

---

## 1. Project Overview
- Project Name: VedaVision
- Goal: Machine vision system to identify morphologically similar Ayurvedic medicinal species and assess leaf health.
- Assigned Module: **Module 3 — Compound Leaf Analysis** (Species Identification + Multi-level Health Assessment)
- Student: Wijesinghe S.A. (Siluni), final-year, University of Moratuwa
- Supervisor: Dr. Ranathunga
- Collaboration: Gampaha Wickramarachchi Ayurveda Hospital
- Scope: 12 compound leaf species (expanded from an earlier 15-species scope)
- Architecture Direction (system-wide, not yet built for Module 3): React + Vite frontend, Axios, FastAPI + Uvicorn backend, Pydantic validation, MongoDB Atlas, Docker + Render deployment

---

## 2. Confirmed Codebase Structure

```
preprocessing/
  config.py                      ← single source of truth for all parameters
  batch_processor.py             ← train/test batch entry point
  shared/
    resize.py                    ← letterbox resize
    masking.py                   ← v5.1.1 background removal (see §4)
    augmentation.py              ← offline Albumentations pipeline
  species_id/
    enhance.py                   ← bilateral + CLAHE + unsharp (species-ID branch only)
    pipeline.py                  ← single-image orchestrator
  health/                        ← folder scaffolded, EMPTY — not yet implemented
feature_extraction/species_id/
  shape.py, colour.py, texture.py, vein.py, whole_leaf.py
models/species_id/
  model_training.py              ← ensemble training + evaluation (IMPLEMENTED)
notebooks/
  vedavision_single_leaf_pipeline.ipynb   ← stage-by-stage visual walkthrough
```

**Status change from earlier notes:** classifier training is **no longer future work** — `model_training.py` is implemented and runs end-to-end (CV + held-out test evaluation + model save). Health assessment (`preprocessing/health/`) is still an empty scaffold — genuinely not started.

---

## 3. Dataset & Train/Test Convention

- 12 species, `dataset/raw/<species>/top|bottom/*.jpg`
- **Test images**: filename must start with `test_` (e.g. `test_001.jpg`–`test_005.jpg`), 5 per species per view. Never augmented.
- **Train images**: any filename that does NOT start with `test_` — original camera-generated names (e.g. `PXL_*.jpg`) are used as-is, no renaming required (this replaces the earlier plan to rename 120 files to `train_001..030`).
- Same leaf ID must exist in both `top/` and `bottom/` folders per species (enforces leaf-level, not just image-level, split).
- `image_path` is the StratifiedGroupKFold group key — keeps all augmented rows of one physical leaf in the same CV fold.

---

## 4. Masking Pipeline — v5.1.1 (confirmed from source, `masking.py`)

9-stage seed + region-growing algorithm, NOT a simple dual-threshold mask:

1. **Seed** — ExG > 20 & S > 25 & L < 130 (Tier 1); relaxed fallback ExG > 8 & S > 15 & L < 150 (Tier 2, dark-pigmented species e.g. Kattakumanjal) if Tier-1 coverage < 1%.
2. **Leaf colour model** — per-image LAB mean ± std from seed pixels, std floor clamped to ≥ 8.
3. **Candidate map** — sigma-distance gate (2.5σ) + saturation gate (S>20, excludes achromatic shadow) + paper-gap gate (excludes S<30 & L>160 pale/achromatic pixels).
4. **Region growing** — iterative dilation (40 iters max, 5×5 kernel) constrained to candidate map, early-stops on convergence.
5. **Tight/loose structure selection** — tight (k3, iter=1) used if ≥3 components AND tight_area ≥ 0.25 × loose_area, else loose (k5, iter=1). iter=1 is deliberate (iter=2 was found to fuse pinnate leaflets in v5.0).
6. **Rachis mask** — separate detection: Tier A brown/tan (LAB b>133, S>35, 50<L<150), Tier B green stem (ExG 3–18, S>20, L<140); paper-gap exclusion (S<30 & L>160); proximity-gated to within 15px of leaflet mask; NO morph_open, NO `_remove_noise` (preserves thin 2–4px rachis lines).
7. **Union** — leaflet_mask OR rachis_mask.
8. **Hole fill** — border-seeded flood fill on inverted mask, run BEFORE `_remove_noise` (order matters: fills enclosed holes without closing inter-leaflet gaps, since gaps touch the image border).
9. **Final clean** — light close (k3, iter=1) + noise removal + paper-leak veto (L>175 & S<25 forced to background) + padding exclusion.

Public API unchanged: `select_mask()` → `(mask_final, mask_choice, diag)`, `qc_check()`.

**QC thresholds (confirmed from `config.py`, corrected from earlier notes):**
- `QC_MIN_COVERAGE = 0.02` (2%)
- `QC_MAX_COVERAGE = 0.75` (75% — **not 95%** as earlier notes stated)

### ⚠️ Known limitation (confirmed by user, important — do not re-propose without this caveat)
`leaflet_mask` / the final combined mask does **not** reliably separate individual leaflet instances. Touching/overlapping leaflets can merge into a single connected component. **Any feature design that assumes per-leaflet instance segmentation (leaflet count, terminal-vs-lateral leaflet identification via connected components, etc.) is currently unreliable and should not be relied on.** True leaflet-instance segmentation (e.g. watershed/distance-transform) is documented as future work, not attempted under the current timeline.

---

## 5. Enhancement (species-ID branch only)

Bilateral filter (d=9, σColor=σSpace=75) → CLAHE on L channel (clip=2.5, tile=8×8) → Unsharp mask (σ=3, strength=1.5×), then re-masked. Applied only to the branch feeding texture/vein features and the CNN input. **Never applied before health feature extraction** — this remains a hard architectural rule.

---

## 6. Augmentation (`augmentation.py`, confirmed implemented)

- Applied to RAW BGR images, before resize/mask/enhance.
- `N_AUGMENTATIONS = 6` variants per original (7 rows per leaf per view including original).
- Applied to train images only; test images are never augmented.
- Augmented images are NOT saved to disk — regenerated on demand; only extracted feature rows are saved to CSV.
- Included: HorizontalFlip, VerticalFlip, Rotate ±30° (white border fill), BrightnessContrast ±15%, HueSaturationValue (mild: hue±8, sat±15, val±10), GaussianBlur (3–5px), GaussNoise (std_range 0.01–0.05), RandomShadow.
- Excluded (with reasons in code): RandomCrop/ElasticTransform (destroys leaflet/shape features), CoarseDropout (fabricates fake lesions, corrupts health branch), strong colour shifts (fabricates false yellowing/browning).

---

## 7. Feature Extraction (species identification — PROTOTYPE, NOT FINALIZED)

### ⚠️ Status correction (this session)
The five-group feature set below and the ensemble classifier in §8 were built as an **exploratory prototype to get early feedback from the supervisor** — they are **not the finalized feature extraction or classification approach**. Both are explicitly open to redesign. The confirmed project objective is:

> Design **handcrafted, botanically-grounded features** as the primary/core feature set for species identification (not generic textbook CV descriptors fed into an off-the-shelf classifier), then build classification around those handcrafted features.

Treat everything below in §7–§8 as a working baseline / reference implementation, not the target architecture. Do not describe it in the dissertation as the final feature extraction methodology.

Five groups, computed per `pipeline.py` (prototype baseline):
- **Shape** (`shape.py`) — from `mask_final`: aspect_ratio, circularity, solidity, convexity, compactness, elongation (+ Hu moments per module design).
- **Colour** (`colour.py`) — from the RAW resized image (pre-enhancement) within the mask: robust (median/IQR-based) per-channel statistics across BGR/HSV/LAB + hue histogram.
- **Texture** (`texture.py`) — from the enhanced (`img_sharp`) image within the mask: GLCM + LBP descriptors.
- **Vein** (`vein.py`) — from the enhanced image, ROI-crop-then-upscale pipeline: `vein_density`, `vein_length_ratio`, `vein_branch_density`, `vein_end_point_density` (normalised ratios) + 2 diagnostic-only columns `vein_coverage_pct`, `vein_roi_scale` (excluded from classifier input).
- **Whole-leaf** (`whole_leaf.py`) — from `mask_final`: leaflet arrangement descriptors (aspect, area CV, symmetry, spacing CV, etc.).

All feature dicts are namespaced by group prefix (`shape_*`, `colour_*`, `texture_*`, `vein_*`, `whole_*`) and flattened into one row per image/variant in `pipeline.py`.

**Note on total feature count:** an earlier session's manual audit on generated data reported ~129 raw features with 23 recommended for dropping (sentinel/near-constant columns), improving CV F1-macro from ~0.9095 to ~0.9238. This audit was performed on a prior CSV export and has **not been re-verified against the current codebase** in this zip — re-run the audit against the latest `vedavision_features_train_clf.csv` before citing exact numbers in the dissertation.

### 🔴 Supervisor feedback — standard features are not sufficient
Dr. Ranathunga has flagged that the five feature groups above are **generic/standard CV descriptors** (Hu moments, GLCM, LBP, etc.) and require a genuinely **handcrafted, botanically-grounded** feature set as the project's core novelty contribution — not just more of the same descriptor types, and not standard descriptors merely fed into a classifier.

**Agreed direction (updated — supervisor has since clarified the handcrafted feature vector can be large, up to ~100 dimensions, not limited to a handful):** build a broad, botanically-organized handcrafted feature bank in `feature_extraction/species_id/morphology.py`, covering multiple diagnostic categories (not just the axis-profile features originally scoped under the 2-week time pressure):
- **Arrangement/profile features** (no leaflet-instance segmentation needed) — width-along-rachis profile: `profile_peak_count`, `profile_symmetry`, `taper_ratio`, multi-bin width histogram (can contribute 10–20 dims alone at finer bin resolution)
- **Tip/base region shape** — aspect/area ratios of the mask regions near the far and near ends of the rachis axis
- **Margin character** — local contour curvature histogram (serration frequency/depth), sampled at multiple scales for a multi-dimensional descriptor
- **Blade-vs-rachis proportion** — rachis length, blade length, rachis-to-blade ratio, rachis-pixel-density along its length
- **Botanically-reframed shape descriptors** — leaflet blade elongation, ovate/lanceolate/elliptic proxy ratios, apex angle, base angle — computed specifically as named botanical characters (not generic Hu moments), even where the underlying math is similar to a standard descriptor
- **Vein-pattern structural descriptors** (building on existing `vein.py` skeleton output, reframed with botanical naming) — primary vein angle relative to rachis axis, secondary vein density gradient from base to tip

**Status: expanded scope decided this session, feature bank not yet implemented in the codebase.** Original 4–6 feature MVP (from the 2-week-deadline discussion) is now the *minimum viable subset*, not the target — supervisor expects closer to ~100 handcrafted dimensions total.

**Novelty framing for the dissertation:** structure-aware handcrafted features derived via axis-projection that explicitly avoid the leaflet-instance-segmentation problem documented above — positioned as the practical alternative to (fragile) full leaflet segmentation, which prior compound-leaf work either requires or ignores entirely.

---

## 8. Classification (species identification — PROTOTYPE, NOT FINALIZED, confirmed from `model_training.py`)

This ensemble was built as a working baseline to validate the pipeline end-to-end and get supervisor feedback — the classifier choice is open for reconsideration alongside the new handcrafted feature set (e.g. an interpretable model may fit a smaller, botanically-meaningful feature vector better than a 3-way black-box ensemble). Re-evaluate classifier choice once `morphology.py` features exist.

- **Model (prototype)**: soft-voting `VotingClassifier` — RandomForest (n=200, class_weight=balanced) + SVM-RBF (C=10, gamma=scale, class_weight=balanced, probability=True) + HistGradientBoosting (max_iter=150, class_weight=balanced), wrapped in a `Pipeline` with `StandardScaler`.
- **CV**: `StratifiedGroupKFold(n_splits=5)`, grouped by `image_path`, scored on `f1_macro`.
- **Held-out test evaluation**: classification report + confusion matrix (saved as PNG) on the untouched `test_` images.
- **Look-alike pair accuracy check** (hardcoded in `model_training.py` — confirmed current pairs, **different from the interim report's original 6 pairs**):
  - `thunpath_kurundu` vs `kasthuri_dehi`
  - `beli` vs `wal_kollu`
  (Only 2 pairs currently checked in code; the interim report's other pairs — Kattakumanjal/Kalawal, Bilin/Kamaranka, Nil Awariya/Kathurupila, Maha Undupiyaliya/Ambulwanna — are not yet wired into this evaluation block and should be added or confirmed as still relevant to the current 12-species set.)
- Model persisted via `joblib` to `vedavision_species_model.pkl`.
- MobileNetV2 CNN branch: still conditional on dataset expansion, not implemented in this zip.

---

## 9. On the Horizon (next 2 weeks, priority order)

1. **Design and implement `morphology.py`** as the primary handcrafted feature set (width-profile + tip-region features, §7) — target 3–4 days including retraining/validation.
2. **Decide/finalize the classifier** to pair with the handcrafted features (keep, simplify, or replace the §8 ensemble prototype) — this decision is explicitly open, not settled.
3. **Ablation**: evaluate handcrafted-features-primary (± retained standard features as support) vs. the original standard-only prototype; compare CV F1-macro and confusion matrix (especially on the look-alike pairs); write up the novelty paragraph framing handcrafted features as the core contribution.
4. **Health assessment module** (`preprocessing/health/` — currently empty):
   - Feature pipeline routes through `masked_raw/` images (already saved by `batch_processor.py`), with **no enhancement** applied (hard rule, see §5).
   - Two-stage hierarchical classifier: Stage 1 Healthy vs Unhealthy (binary), Stage 2 Unhealthy → Low/Mid/Full Degraded.
   - Health dataset target: ~15 images/view/degradation level + 5 test images/level/species; healthy images reused from the species-ID dataset (no new healthy collection needed).
5. MobileNetV2 CNN branch — remains conditional on dataset expansion; not a near-term priority given the timeline.

---

## 10. Key Learnings & Principles (confirmed still valid)

- Enhancement must never be applied before health feature extraction — corrupts colour degradation signals (bilateral filter/CLAHE/unsharp all distort exactly the signals health assessment needs).
- Species identity is a prerequisite for health assessment (health baselines are species-specific).
- Augmentation happens on raw images before preprocessing, not on processed/enhanced images — ensures the masking pipeline trains on realistic variation.
- Augmented images are never saved (cheap to regenerate); only feature rows are saved.
- Test images are never augmented.
- `StratifiedGroupKFold` on `image_path` is mandatory to prevent augmentation leakage across CV folds.
- Black background (via `cv2.bitwise_and`) is the correct masking default — zero-value pixels can't corrupt colour statistics the way white padding could.
- Scale-dependent raw pixel-count features are excluded from the classifier by design.
- **New (this session)**: leaflet-instance segmentation is currently unreliable — do not design features that depend on it; use axis-projection/profile-based structural features instead as the practical, achievable alternative.
- Every hardcoded threshold in the masking pipeline has an explicit physical/statistical/literature-based justification recorded in the code docstrings (for viva defensibility) — maintain this standard for any new `morphology.py` thresholds too.

---

## 11. Tools & Resources

- Python, OpenCV, Albumentations 2.0.8, scikit-learn, scikit-image, pandas, numpy, matplotlib, joblib
- VS Code primary IDE; Jupyter notebooks for prototyping/visualisation (`notebooks/vedavision_single_leaf_pipeline.ipynb`)
- Dataset: `dataset/raw/<species>/top|bottom/`
- Collaboration institution: Gampaha Wickramarachchi Ayurveda Hospital

---

## 12. Approach & Communication Preferences

- Prefers very simple, plain language for slide text and verbal explanations.
- Expects complete, working deliverables (full files, not snippets).
- Wants Claude to flag when a proposed fix generalises poorly to other species/edge cases.
- Values principled, academically defensible design decisions over computationally convenient ones — every design choice should be citable/justifiable in a viva.
- Currently under a hard 2-week deadline covering both the handcrafted feature novelty work and the entire health assessment module — prioritise ruthlessly, document honestly (including negative/marginal results) rather than over-engineering.

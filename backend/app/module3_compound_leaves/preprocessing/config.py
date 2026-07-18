

# Resolution 
TARGET_LONG   = 512       # Longest side px — must match training.

#  Background removal
MIN_COMP_FRAC  = 0.001    # Minimum connected component as fraction of image area.
SIGMA_THRESH   = 2.5      # LAB colour-model gate for candidate foreground pixels.

# Texture features 
LBP_RADIUS    = 3
LBP_POINTS    = 24
GLCM_DIST     = [1, 3]
GLCM_ANGLES   = [0, 1.5707963, 0.7853981, 2.3561944]  # 0°, 90°, 45°, 135°

#  Enhancement (species-ID branch only) ─
BILATERAL_D          = 9
BILATERAL_SIGMA      = 75
CLAHE_CLIP           = 2.5
CLAHE_TILE           = (8, 8)
UNSHARP_SIGMA        = 3
UNSHARP_STRENGTH     = 1.5

#  QC thresholds (kept for reference / explicit calls; qc_check()'s own
#    defaults already match these) 
QC_MIN_COVERAGE  = 0.02   # < 2%  → leaf not detected
QC_MAX_COVERAGE  = 0.75   # > 75% → background leaking

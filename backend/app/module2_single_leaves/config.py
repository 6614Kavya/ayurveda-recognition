"""
module2_single_leaves/config.py

"""

# ============================================================
# IMAGE SIZES
# ============================================================
WORK_SIZE = (512, 512)     # working resolution during preprocessing
TARGET_SIZE = (224, 224)   # final output resolution
PAD_VALUE = 255            # white padding when preserving aspect ratio

VIEWS = ['top', 'bottom']
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


# ============================================================
# SEGMENTATION (segmentation.py -> segment_leaf)
# ============================================================
SEGMENTATION = {
    'dark_v_thresh': 90,          # HSV V median below this = shadow/background
    'min_shadow_area': 200,       # enclosed regions smaller than this stay "undecided"
    'max_artifact_area': 150,     # small undecided blobs adjacent to shadow get promoted
    'bridge_break_px': 7,         # erosion kernel size to break hairline bg/leaf bridges
    'morph_kernel_size': 3,       # close/open kernel size for mask cleanup
    'min_blob_area': 100,         # minimum leaf blob area to survive small-blob removal
    'otsu_relax_factor': 0.8,     # relax Otsu saturation threshold for pale leaves
}

CONTOUR_FILTER = {
    'keep_ratio': 0.05,           # keep contours >= this fraction of the largest
    'min_absolute_area': 1500,    # absolute floor so small legit lobes aren't dropped
}

CROP = {
    'padding_frac': 0.10,         # padding added around leaf bbox when cropping
}

# Coverage sanity bounds -- reject a mask if the leaf takes up too
# little or too much of the frame (likely a bad segmentation)
COVERAGE_MIN = 0.02
COVERAGE_MAX = 0.95


# ============================================================
# SHADOW CORRECTION (shadow_removal.py -> correct_leaf_shadow)
# ============================================================
SHADOW_CORRECTION = {
    'blur_ksize': 61,             # Gaussian blur kernel for illumination map
    'max_gain': 2.2,              # cap on brightness rescaling to avoid blowing out pixels
    'shadow_percent_limit': 15,   # skip correction if >15% of leaf flagged as shadow
    'low_sat_ratio': 0.40,        # shadow pixel saturation must be below median * this
}


# ============================================================
# ENHANCEMENT (enhancement.py -> CLAHE + denoising)
# ============================================================
CLAHE = {
    'clip_limit': 2.0,
    'tile_grid_size': (8, 8),
}

DENOISE = {
    'diameter': 9,                # bilateral filter d
    'sigma_color': 50,
    'sigma_space': 50,
    'edge_trim_px': 2,            # erosion applied to mask after denoising
}


# ============================================================
# FEATURE EXTRACTION -- TEXTURE (texture.py: GLCM, Gabor, LBP)
# ============================================================
GLCM = {
    'distances': [1, 2, 3, 4],
    'angles': [0, 0.785398, 1.570796, 2.356194],  # 0, pi/4, pi/2, 3pi/4
    'properties': ['contrast', 'dissimilarity', 'homogeneity',
                   'energy', 'correlation', 'ASM'],
    'levels': 256,
}

GABOR = {
    'frequencies': [0.1, 0.2, 0.3, 0.4, 0.5],
    'orientations': [0, 0.785398, 1.570796, 2.356194],  # 0, pi/4, pi/2, 3pi/4
}

LBP = {
    'radius': 3,
    'n_points_multiplier': 8,     # n_points = 8 * radius
    'method': 'uniform',
}


# ============================================================
# FEATURE EXTRACTION -- SHAPE (shape.py: HOG, Hu, contour, handcrafted)
# ============================================================
HOG = {
    'orientations': 9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
}

NOTCH = {
    'min_defect_depth_px': 3,
}

MARGIN = {
    'defect_depth_thresh_px': 4,
}

VEIN_EDGE = {
    'canny_low': 40,
    'canny_high': 120,
}


# ============================================================
# FEATURE EXTRACTION -- COLOUR (colour.py: HSV)
# ============================================================
HSV_COLOR = {
    'background_threshold': 250,  # grayscale value above which a pixel is "white bg"
}


# ============================================================
# EXPECTED FEATURE COUNTS (for validation / sanity checks)
# ============================================================
FEATURE_COUNTS = {
    'glcm': 6,
    'gabor': 40,
    'hog': 5,
    'hu_moments': 7,
    'hsv_color': 6,
    'contour_shape': 7,
    'lbp': 26,
    'handcrafted': 8,   # notch(2) + margin(2) + principal_axis(3) + vein_edge(1)
    'total': 105,
}
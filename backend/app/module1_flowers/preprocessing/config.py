import numpy as np

CFG = {
    'roi_size'            : (224, 224),
    'min_flower_coverage' : 0.01,
    'max_flower_coverage' : 0.92,
    'glcm_distances'      : [1, 3],
    'glcm_angles'         : [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    'gabor_frequencies'   : [0.1, 0.3, 0.5],
    'gabor_orientations'  : [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    'lbp_radius'          : 3,
    'lbp_n_points'        : 24,
    'hist_bins'           : 32,
}

"""
features package
=================
Each module here extracts ONE family of features from an ROI dict
(the dict returned by `roi_extraction.extract_roi`). Every function has
the signature `extract_*(...) -> np.ndarray` and never mutates its inputs.

Module              -> feature count
--------------------------------------
color                  105
texture_glcm            10
texture_lbp             26
texture_gabor           24
shape                   12
petal_proxy               4
petal_morphometrics     10
filament_core             9
petal_overlap             5
vein_center               5
                       -----
TOTAL                   210
"""

from .color import extract_color_features
from .texture_glcm import extract_glcm_features
from .texture_lbp import extract_lbp_features
from .texture_gabor import extract_gabor_features
from .shape import extract_shape_features
from .petal_proxy import extract_petal_proxy
from .petal_morphometrics import extract_petal_morphometrics
from .filament_core import extract_filament_core_features
from .petal_overlap import extract_petal_overlap_features
from .vein_center import extract_vein_center_features

__all__ = [
    "extract_color_features",
    "extract_glcm_features",
    "extract_lbp_features",
    "extract_gabor_features",
    "extract_shape_features",
    "extract_petal_proxy",
    "extract_petal_morphometrics",
    "extract_filament_core_features",
    "extract_petal_overlap_features",
    "extract_vein_center_features",
]

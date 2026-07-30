import cv2
import numpy as np

def extract_color_features(image_bgr, mask):
    leaf_pixels = mask > 0
    if leaf_pixels.sum() == 0:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    h, s, v = hsv[..., 0][leaf_pixels], hsv[..., 1][leaf_pixels], hsv[..., 2][leaf_pixels]
    L, a_raw, b_raw = lab[..., 0][leaf_pixels], lab[..., 1][leaf_pixels], lab[..., 2][leaf_pixels]

    a = a_raw - 128.0
    b = b_raw - 128.0
    chroma = np.sqrt(a**2 + b**2)

    # OpenCV hue range is 0-179; green ~ 35-85.
    green_ratio = np.mean((h >= 35) & (h <= 85))

    # Yellowing / chlorosis: hue shifting toward yellow (~20-35) while still fairly saturated
    yellow_ratio = np.mean((h >= 20) & (h < 35) & (s > 40))

    # Browning: low value, low-to-mid saturation, hue in brown/orange range
    brown_ratio = np.mean((h >= 5) & (h < 25) & (v < 150))

    # Necrosis/near-black: very low value regardless of hue (fungal lesion centers,
    # skeletonized tissue edges often trend dark)
    dark_ratio = np.mean(v < 60)

    specular_highlight_ratio = float(np.mean((v > 220) & (s < 60)))

    return {
        'hue_mean': float(np.mean(h)), 'hue_std': float(np.std(h)),
        'sat_mean': float(np.mean(s)), 'sat_std': float(np.std(s)),
        'val_mean': float(np.mean(v)), 'val_std': float(np.std(v)),
        'L_mean': float(np.mean(L)), 'L_std': float(np.std(L)),
        'a_mean': float(np.mean(a)), 'b_mean': float(np.mean(b)),
        'chroma_mean': float(np.mean(chroma)), 'chroma_std': float(np.std(chroma)),
        'green_ratio': float(green_ratio),
        'yellow_ratio': float(yellow_ratio),
        'brown_ratio': float(brown_ratio),
        'dark_ratio': float(dark_ratio),
        'specular_highlight_ratio': specular_highlight_ratio,
    }

 

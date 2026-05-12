"""
features/species_id/build_feature_matrix.py
Runs steps 7a + 8a on all preprocessed images → saves features.pkl

Usage:
    python features/species_id/build_feature_matrix.py

Or from notebook:
    from features.species_id.build_feature_matrix import build_matrix
    data = build_matrix()
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from preprocessing.config import PROCESSED_DIR, FEATURES_PKL, FEATURES_AUG_PKL, IMG_EXTENSIONS
from features.species_id.leaflet_segmentation import segment_leaflets
from features.species_id.feature_extractor import extract_features


def build_matrix(processed_dir=PROCESSED_DIR,
                 output_pkl=FEATURES_PKL) -> dict:
    """
    Iterates over all preprocessed images, extracts features, saves matrix.

    Output format (saved as .pkl):
        {
            'X'            : np.ndarray (N_images × N_features),
            'y'            : list of species label strings,
            'filenames'    : list of image filenames,
            'feature_names': list of feature column names
        }

    This .pkl is the direct input to the classifier notebook (04).
    """
    processed_dir = Path(processed_dir)
    all_imgs = [
        f for f in processed_dir.rglob('*')
        if f.suffix.lower() in IMG_EXTENSIONS
    ]

    if not all_imgs:
        print(f'⚠️  No images in {processed_dir}. Run preprocessing first.')
        return {}

    all_features, all_labels, all_files = [], [], []
    errors = 0

    for img_path in tqdm(all_imgs, desc='Extracting features'):
        species  = img_path.parent.name
        mask_path = img_path.with_name(img_path.stem.replace('', '') + '_mask.npy').with_suffix('.npy')

        img = cv2.imread(str(img_path))
        if img is None:
            errors += 1
            continue

        # Load mask (saved alongside image during preprocessing)
        if mask_path.exists():
            mask = np.load(str(mask_path))
        else:
            # Fallback: generate simple green mask
            hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([20,30,30]), np.array([90,255,255]))

        # Step 7a: leaflet segmentation
        leaflets, _ = segment_leaflets(img, mask)

        # Step 8a: feature extraction
        feats = extract_features(img, mask, leaflets)

        all_features.append(feats)
        all_labels.append(species)
        all_files.append(img_path.name)

    if not all_features:
        print('❌ No features extracted.')
        return {}

    feature_names = list(all_features[0].keys())
    X = np.array([[f.get(k, 0.0) for k in feature_names] for f in all_features],
                 dtype=np.float32)

    data = {
        'X'            : X,
        'y'            : all_labels,
        'filenames'    : all_files,
        'feature_names': feature_names
    }

    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

    print(f'\n✅ Feature matrix saved → {output_pkl}')
    print(f'   Shape     : {X.shape}')
    print(f'   Species   : {sorted(set(all_labels))}')
    print(f'   Errors    : {errors}')
    return data


if __name__ == '__main__':
    build_matrix()
"""
preprocessing/species_id/pipeline_sid.py
Orchestrates steps 1–6a for species ID branch.

Responsibility: produce clean preprocessed images ready for feature extraction.
Does NOT extract features — that is features/species_id/feature_extractor.py

Usage:
    Single image:
        from preprocessing.species_id.pipeline_sid import preprocess_one
        img, mask, meta = preprocess_one('dataset/raw/neem/leaf_001.jpg')

    Entire dataset:
        from preprocessing.species_id.pipeline_sid import batch_preprocess
        batch_preprocess()
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from preprocessing.config import (
    IMG_EXTENSIONS, PROCESSED_DIR, REJECTED_DIR,
    DATASET_RAW
)
from preprocessing.shared.base import (
    check_quality, step1_resize,
    step2_remove_background,
    step3_histogram_matching,
    step4_bilateral_filter
)
from preprocessing.species_id.transforms_sid import step5a_clahe, step6a_sharpen


# =============================================================================
# SINGLE IMAGE
# =============================================================================

def preprocess_one(img_path: str | Path,
                   save_stages: bool = False
                   ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """
    Runs steps 1–6a on one image.

    Returns:
        preprocessed_img : uint8 BGR 512×512, or None if rejected
        binary_mask      : uint8 leaf mask, or None if rejected
        meta             : dict with status, blur_score, filename

    Example:
        img, mask, meta = preprocess_one('dataset/raw/neem/leaf_001.jpg')
        if img is not None:
            cv2.imwrite('out.png', img)
    """
    img_path = Path(img_path)
    meta     = {'filename': img_path.name, 'status': None}
    stages   = {}

    # Load
    img = cv2.imread(str(img_path))
    if img is None:
        meta.update({'status': 'ERROR', 'reason': 'Cannot load image'})
        return None, None, meta

    # Quality check
    ok, blur_score = check_quality(img)
    meta['blur_score'] = blur_score
    if not ok:
        meta.update({'status': 'REJECTED',
                     'reason': f'Too blurry (score={blur_score}). Retake.'})
        return None, None, meta

    # Step 1
    img = step1_resize(img)
    if save_stages: stages['1_resize'] = img.copy()

    # Step 2
    img, mask = step2_remove_background(img)
    if save_stages: stages['2_bg_removed'] = img.copy()

    # Step 3 (skipped)
    img = step3_histogram_matching(img)

    # Step 4
    img = step4_bilateral_filter(img)
    if save_stages: stages['4_bilateral'] = img.copy()

    # Step 5a
    img = step5a_clahe(img)
    if save_stages: stages['5a_clahe'] = img.copy()

    # Step 6a
    img = step6a_sharpen(img)
    if save_stages: stages['6a_sharpened'] = img.copy()

    meta['status'] = 'OK'
    meta['stages'] = stages
    return img, mask, meta


# =============================================================================
# BATCH — entire dataset
# =============================================================================

def batch_preprocess(raw_dir=DATASET_RAW,
                     processed_dir=PROCESSED_DIR,
                     rejected_dir=REJECTED_DIR) -> dict:
    """
    Preprocesses all images in raw_dir.
    Saves clean images to processed_dir/species_name/
    Copies rejected images to rejected_dir/species_name/

    Run this ONCE before feature extraction.
    Re-run only if you add new images or change config params.
    """
    raw_dir       = Path(raw_dir)
    processed_dir = Path(processed_dir)
    rejected_dir  = Path(rejected_dir)

    all_imgs = [f for f in raw_dir.rglob('*') if f.suffix.lower() in IMG_EXTENSIONS]
    if not all_imgs:
        print(f'⚠️  No images found in {raw_dir}')
        return {}

    report = {'ok': 0, 'rejected': 0, 'errors': 0}

    for img_path in tqdm(all_imgs, desc='Preprocessing'):
        species  = img_path.parent.name
        out_dir  = processed_dir / species
        rej_dir  = rejected_dir  / species
        out_dir.mkdir(parents=True, exist_ok=True)
        rej_dir.mkdir(parents=True, exist_ok=True)

        img, mask, meta = preprocess_one(img_path)

        if meta['status'] == 'OK':
            cv2.imwrite(str(out_dir / (img_path.stem + '.png')), img)
            np.save(str(out_dir / (img_path.stem + '_mask.npy')), mask)
            report['ok'] += 1
        elif meta['status'] == 'REJECTED':
            shutil.copy(img_path, rej_dir / img_path.name)
            report['rejected'] += 1
        else:
            report['errors'] += 1

    print(f'\n✅ Preprocessing done — {report}')
    return report


if __name__ == '__main__':
    batch_preprocess()
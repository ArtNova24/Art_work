"""
ANTIGRAVITY Phase 1 — Step 4: LBP Feature Extraction
Extracts 256-dimensional Local Binary Pattern histogram features.

Output: features/lbp_features.npy  shape (N, 256)

Computation:
  - LBP with P=8, R=1, method='default' → 256 possible values (2^8)
  - Computed on full 224×224 grayscale image
  - Histogram normalised to sum=1
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.feature import local_binary_pattern

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURES_DIR, LBP_P, LBP_R, LBP_BINS, IMG_SIZE, LBP_DIM
from preprocessing import safe_load


def extract_lbp(gray_img: np.ndarray) -> np.ndarray:
    """
    Extract 256-dim LBP histogram from a grayscale image.

    Args:
        gray_img : (H, W) uint8 grayscale image
    Returns:
        (256,) float32 normalised histogram
    """
    # Compute LBP values (0-255 for P=8, method='default')
    lbp = local_binary_pattern(gray_img, P=LBP_P, R=LBP_R, method='default')

    # Compute 256-bin histogram, normalised
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, 256), density=False)
    hist = hist.astype(np.float32)

    # Normalise to sum to 1
    total = hist.sum()
    if total > 0:
        hist /= total

    assert hist.shape == (LBP_DIM,), f"LBP dim mismatch: {hist.shape}"
    return hist


def extract_lbp_all(image_index: list) -> np.ndarray:
    """
    Extract LBP features for all images in the index.

    Args:
        image_index : list of dicts with 'path' key
    Returns:
        (N, 256) float32 array
    """
    all_features = []
    failed = []

    for item in tqdm(image_index, desc="  Extracting LBP features ", unit="img"):
        result = safe_load(item['path'], IMG_SIZE)
        if result is None:
            failed.append(item['path'])
            all_features.append(np.zeros(LBP_DIM, dtype=np.float32))
            continue

        _, _, _, gray, _ = result
        vec = extract_lbp(gray)
        all_features.append(vec)

    if failed:
        print(f"\n  WARNING: {len(failed)} images failed to load (replaced with zeros).")

    return np.array(all_features, dtype=np.float32)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ANTIGRAVITY Phase 1 — Step 4: LBP Extraction")
    print("="*60)

    index_path = os.path.join(FEATURES_DIR, "image_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("Run download_dataset.py first to generate image_index.json")

    with open(index_path) as f:
        image_index = json.load(f)

    print(f"  Processing {len(image_index)} images...")
    lbp_features = extract_lbp_all(image_index)

    out_path = os.path.join(FEATURES_DIR, "lbp_features.npy")
    np.save(out_path, lbp_features)

    print(f"\n  ✓ LBP features saved: {lbp_features.shape} → {out_path}")
    print(f"  Min: {lbp_features.min():.4f}  Max: {lbp_features.max():.4f}  Mean: {lbp_features.mean():.4f}")

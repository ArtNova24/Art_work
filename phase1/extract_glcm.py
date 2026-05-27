"""
Historic Image Restoration Phase 1 — Step 3: GLCM Feature Extraction
Extracts 20-dimensional Grey-Level Co-occurrence Matrix features.

Output: features/glcm_features.npy  shape (N, 20)

Computation:
  - 5 properties × 4 angles = 20 values
  - Averaged over 3 distances (1, 3, 5) for robustness
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURES_DIR, GLCM_DISTANCES, GLCM_ANGLES, GLCM_PROPS, IMG_SIZE, GLCM_DIM
from preprocessing import safe_load


def extract_glcm(gray_img: np.ndarray) -> np.ndarray:
    """
    Extract 20-dim GLCM feature vector from a grayscale image.

    Args:
        gray_img : (H, W) uint8 grayscale image
    Returns:
        (20,) float32 feature vector
    """
    # Reduce to 64 grey levels for efficiency
    gray_64 = (gray_img // 4).astype(np.uint8)

    # Compute GLCM: shape (levels, levels, num_dist, num_angles)
    glcm = graycomatrix(
        gray_64,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in GLCM_PROPS:                        # 5 properties
        prop_vals = graycoprops(glcm, prop)         # shape (num_dist, num_angles)
        # Average over distances → (num_angles,)
        avg_over_dist = prop_vals.mean(axis=0)      # shape (4,)
        features.extend(avg_over_dist.tolist())     # 4 values per property

    # Total: 5 × 4 = 20-dim
    vec = np.array(features, dtype=np.float32)
    assert vec.shape == (GLCM_DIM,), f"GLCM dim mismatch: {vec.shape}"
    return vec


def extract_glcm_all(image_index: list) -> np.ndarray:
    """
    Extract GLCM features for all images in the index.

    Args:
        image_index : list of dicts with 'path' key
    Returns:
        (N, 20) float32 array
    """
    all_features = []
    failed = []

    for item in tqdm(image_index, desc="  Extracting GLCM features", unit="img"):
        result = safe_load(item['path'], IMG_SIZE)
        if result is None:
            failed.append(item['path'])
            all_features.append(np.zeros(GLCM_DIM, dtype=np.float32))
            continue

        _, _, _, gray, _ = result
        vec = extract_glcm(gray)
        all_features.append(vec)

    if failed:
        print(f"\n  WARNING: {len(failed)} images failed to load (replaced with zeros).")

    return np.array(all_features, dtype=np.float32)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Historic Image Restoration Phase 1 — Step 3: GLCM Extraction")
    print("="*60)

    index_path = os.path.join(FEATURES_DIR, "image_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("Run download_dataset.py first to generate image_index.json")

    with open(index_path) as f:
        image_index = json.load(f)

    print(f"  Processing {len(image_index)} images...")
    glcm_features = extract_glcm_all(image_index)

    out_path = os.path.join(FEATURES_DIR, "glcm_features.npy")
    np.save(out_path, glcm_features)

    print(f"\n  ✓ GLCM features saved: {glcm_features.shape} → {out_path}")
    print(f"  Min: {glcm_features.min():.4f}  Max: {glcm_features.max():.4f}  Mean: {glcm_features.mean():.4f}")

"""
ANTIGRAVITY Phase 1 — Step 5: Color Feature Extraction
Extracts 201-dimensional colour feature vector.

Output: features/color_features.npy  shape (N, 201)

Breakdown:
  - RGB histogram : H(22) + G(21) + B(21)  = 64 dims
  - HSV histogram : H(32) + S(16) + V(16)  = 64 dims
  - LAB histogram : L(32) + A(16) + B(16)  = 64 dims
  - Color moments : mean/std/skew × 3 HSV channels = 9 dims
  Total = 64 + 64 + 64 + 9 = 201 dims  ✓
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURES_DIR, IMG_SIZE, COLOR_DIM
from preprocessing import safe_load


def _channel_hist(channel: np.ndarray, n_bins: int, val_range=(0, 256)) -> np.ndarray:
    """Compute a normalised histogram for one image channel (uint8)."""
    hist, _ = np.histogram(channel.ravel(), bins=n_bins, range=val_range, density=False)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_color(img_rgb: np.ndarray,
                  img_hsv: np.ndarray,
                  img_lab: np.ndarray) -> np.ndarray:
    """
    Extract 201-dim colour feature vector.

    Args:
        img_rgb : (H, W, 3) uint8  RGB
        img_hsv : (H, W, 3) uint8  HSV  (H:0-179, S/V:0-255 — OpenCV scale)
        img_lab : (H, W, 3) uint8  LAB  (L:0-255, A/B:0-255)
    Returns:
        (201,) float32
    """
    features = []

    # ── RGB histograms (64 dims) ─────────────────────────────────────────────
    features.append(_channel_hist(img_rgb[:, :, 0], 22))   # R: 22 bins
    features.append(_channel_hist(img_rgb[:, :, 1], 21))   # G: 21 bins
    features.append(_channel_hist(img_rgb[:, :, 2], 21))   # B: 21 bins
    # 22+21+21 = 64 ✓

    # ── HSV histograms (64 dims) ─────────────────────────────────────────────
    features.append(_channel_hist(img_hsv[:, :, 0], 32, (0, 180)))  # H: 0-179 → 32 bins
    features.append(_channel_hist(img_hsv[:, :, 1], 16))             # S: 16 bins
    features.append(_channel_hist(img_hsv[:, :, 2], 16))             # V: 16 bins
    # 32+16+16 = 64 ✓

    # ── LAB histograms (64 dims) ─────────────────────────────────────────────
    features.append(_channel_hist(img_lab[:, :, 0], 32))   # L: 32 bins
    features.append(_channel_hist(img_lab[:, :, 1], 16))   # A: 16 bins
    features.append(_channel_hist(img_lab[:, :, 2], 16))   # B: 16 bins
    # 32+16+16 = 64 ✓

    # ── Color moments in HSV (9 dims) ────────────────────────────────────────
    # Mean, std, skewness for H, S, V channels
    for ch_idx in range(3):
        ch = img_hsv[:, :, ch_idx].astype(np.float32).ravel()
        ch_std = float(ch.std())
        if ch_std < 1e-6:
            ch_skew = 0.0
        else:
            ch_skew = float(skew(ch))
            if np.isnan(ch_skew) or np.isinf(ch_skew):
                ch_skew = 0.0
        features.append(np.array([ch.mean(), ch_std, ch_skew], dtype=np.float32))
    # 3 × 3 = 9 ✓

    vec = np.concatenate(features).astype(np.float32)
    assert vec.shape == (COLOR_DIM,), f"Color dim mismatch: expected {COLOR_DIM}, got {vec.shape[0]}"
    return vec


def extract_color_all(image_index: list) -> np.ndarray:
    """
    Extract colour features for all images in the index.

    Args:
        image_index : list of dicts with 'path' key
    Returns:
        (N, 201) float32 array
    """
    all_features = []
    failed = []

    for item in tqdm(image_index, desc="  Extracting Color features", unit="img"):
        result = safe_load(item['path'], IMG_SIZE)
        if result is None:
            failed.append(item['path'])
            all_features.append(np.zeros(COLOR_DIM, dtype=np.float32))
            continue

        img_rgb, img_hsv, img_lab, _, _ = result
        vec = extract_color(img_rgb, img_hsv, img_lab)
        all_features.append(vec)

    if failed:
        print(f"\n  WARNING: {len(failed)} images failed to load (replaced with zeros).")

    return np.array(all_features, dtype=np.float32)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ANTIGRAVITY Phase 1 — Step 5: Color Extraction")
    print("="*60)

    index_path = os.path.join(FEATURES_DIR, "image_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("Run download_dataset.py first to generate image_index.json")

    with open(index_path) as f:
        image_index = json.load(f)

    print(f"  Processing {len(image_index)} images...")
    color_features = extract_color_all(image_index)

    out_path = os.path.join(FEATURES_DIR, "color_features.npy")
    np.save(out_path, color_features)

    print(f"\n  ✓ Color features saved: {color_features.shape} → {out_path}")
    print(f"  Min: {color_features.min():.4f}  Max: {color_features.max():.4f}  Mean: {color_features.mean():.4f}")

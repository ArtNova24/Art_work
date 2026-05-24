"""
ANTIGRAVITY Phase 1 — Step 7: Feature Assembly & Splitting
Concatenates all feature vectors into the 989-dim hybrid matrix.
Performs stratified 70/15/15 split and saves all output files.
Computes class weights for imbalance handling.

Output files in features/:
  features_train.npy, features_val.npy, features_test.npy  → (N_split, 989) float32
  labels_train.npy,   labels_val.npy,   labels_test.npy    → (N_split,) int32
  class_weights.npy                                         → (17,) float32
  feature_summary.csv                                       → mean/std per feature per class
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_DIR, GLCM_DIM, LBP_DIM, COLOR_DIM, CNN_DIM, TOTAL_DIM,
    ALL_CLASSES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
)


def load_all_features() -> tuple:
    """Load all intermediate feature arrays and the image index."""
    index_path = os.path.join(FEATURES_DIR, "image_index.json")
    with open(index_path) as f:
        image_index = json.load(f)

    glcm  = np.load(os.path.join(FEATURES_DIR, "glcm_features.npy"))
    lbp   = np.load(os.path.join(FEATURES_DIR, "lbp_features.npy"))
    color = np.load(os.path.join(FEATURES_DIR, "color_features.npy"))
    cnn   = np.load(os.path.join(FEATURES_DIR, "cnn_features.npy"))

    print(f"  Loaded feature arrays:")
    print(f"    GLCM  : {glcm.shape}   (expected N×{GLCM_DIM})")
    print(f"    LBP   : {lbp.shape}   (expected N×{LBP_DIM})")
    print(f"    Color : {color.shape}   (expected N×{COLOR_DIM})")
    print(f"    CNN   : {cnn.shape}   (expected N×{CNN_DIM})")

    labels = np.array([item['class_idx'] for item in image_index], dtype=np.int32)
    return glcm, lbp, color, cnn, labels, image_index


def verify_dimensions(glcm, lbp, color, cnn):
    """Dimension sanity check — raise if any mismatch."""
    assert glcm.shape[1]  == GLCM_DIM,  f"GLCM dim {glcm.shape[1]} ≠ {GLCM_DIM}"
    assert lbp.shape[1]   == LBP_DIM,   f"LBP dim {lbp.shape[1]} ≠ {LBP_DIM}"
    assert color.shape[1] == COLOR_DIM,  f"Color dim {color.shape[1]} ≠ {COLOR_DIM}"
    assert cnn.shape[1]   == CNN_DIM,    f"CNN dim {cnn.shape[1]} ≠ {CNN_DIM}"
    assert glcm.shape[0] == lbp.shape[0] == color.shape[0] == cnn.shape[0], \
        "Number of samples mismatch between feature arrays!"
    print(f"  ✓ All dimensions verified: texture={GLCM_DIM+LBP_DIM}, color={COLOR_DIM}, CNN={CNN_DIM}, total={TOTAL_DIM}")


def assemble_hybrid(glcm, lbp, color, cnn) -> np.ndarray:
    """Concatenate feature groups → (N, 989) hybrid matrix."""
    hybrid = np.concatenate([glcm, lbp, color, cnn], axis=1).astype(np.float32)
    assert hybrid.shape[1] == TOTAL_DIM, f"Hybrid dim {hybrid.shape[1]} ≠ {TOTAL_DIM}"
    print(f"  ✓ Hybrid matrix assembled: {hybrid.shape}")
    return hybrid


def split_data(features: np.ndarray, labels: np.ndarray):
    """Stratified 70/15/15 train/val/test split."""
    # First split off test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED
    )
    # Then split remaining into train/val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )

    print(f"\n  Split results:")
    print(f"    Train : {X_train.shape[0]} samples ({X_train.shape[0]/len(labels)*100:.1f}%)")
    print(f"    Val   : {X_val.shape[0]} samples ({X_val.shape[0]/len(labels)*100:.1f}%)")
    print(f"    Test  : {X_test.shape[0]} samples ({X_test.shape[0]/len(labels)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_weights(y_train: np.ndarray) -> np.ndarray:
    """Compute class weights for imbalance handling (used in Phase 2 training)."""
    classes = np.arange(len(ALL_CLASSES))
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    weights = weights.astype(np.float32)

    print(f"\n  Class weights (higher = more penalised for misclassification):")
    for i, (cls, w) in enumerate(zip(ALL_CLASSES, weights)):
        bar = "█" * int(w * 5)
        print(f"    [{i:2d}] {cls:30s}: {w:.3f} {bar}")

    return weights


def save_feature_summary(features: np.ndarray, labels: np.ndarray):
    """
    Save feature_summary.csv: mean and std of each dimension per style class.
    Shape: 17 classes × 989 features × 2 stats = 17 rows, 1978 columns.
    """
    rows = []
    for cls_idx, cls_name in enumerate(ALL_CLASSES):
        mask = labels == cls_idx
        cls_features = features[mask]
        if len(cls_features) == 0:
            continue
        row = {'class_idx': cls_idx, 'class_name': cls_name, 'n_samples': len(cls_features)}
        for dim in range(features.shape[1]):
            row[f'f{dim}_mean'] = float(cls_features[:, dim].mean())
            row[f'f{dim}_std']  = float(cls_features[:, dim].std())
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(FEATURES_DIR, "feature_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ feature_summary.csv saved → {out_path} ({df.shape})")
    return df


def save_all_outputs(X_train, X_val, X_test, y_train, y_val, y_test, class_weights):
    """Save all npy files to features/ directory."""
    files = {
        "features_train.npy": X_train,
        "features_val.npy":   X_val,
        "features_test.npy":  X_test,
        "labels_train.npy":   y_train,
        "labels_val.npy":     y_val,
        "labels_test.npy":    y_test,
        "class_weights.npy":  class_weights,
    }
    for fname, arr in files.items():
        path = os.path.join(FEATURES_DIR, fname)
        np.save(path, arr)
        print(f"  ✓ Saved {fname}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ANTIGRAVITY Phase 1 — Step 7: Feature Assembly")
    print("="*60)

    # Load
    glcm, lbp, color, cnn, labels, image_index = load_all_features()

    # Verify dimensions
    verify_dimensions(glcm, lbp, color, cnn)

    # Assemble
    hybrid = assemble_hybrid(glcm, lbp, color, cnn)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(hybrid, labels)

    # Class weights
    class_weights = compute_weights(y_train)

    # Feature summary
    save_feature_summary(hybrid, labels)

    # Save everything
    print("\n  Saving output files...")
    save_all_outputs(X_train, X_val, X_test, y_train, y_val, y_test, class_weights)

    print("\n" + "*"*60)
    print("  [SUCCESS] Phase 1 Assembly Complete!")
    print(f"  Hybrid feature matrix: {hybrid.shape[0]} images × {TOTAL_DIM} dims")
    print(f"  Output directory: {FEATURES_DIR}")
    print("*"*60)

"""
Historic Image Restoration Phase 2 — Step 3: SHAP Feature Attribution
Computes SHAP values on a stratified test subset to explain which features
(GLCM, LBP, Color, CNN) are most discriminative globally and per style class.

Uses TreeExplainer on Random Forest/XGBoost for exact and rapid attributions.
Saves beeswarm and summary charts to visualizations/shap/.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

import shap

# Import paths from phase1 config
PHASE1_DIR = Path(__file__).parent.parent / "phase1"
sys.path.insert(0, str(PHASE1_DIR))

from config import (
    FEATURES_DIR, ALL_CLASSES, RANDOM_SEED, TOTAL_DIM,
    GLCM_DIM, LBP_DIM, COLOR_DIM, CNN_DIM
)


def load_data():
    X_train = np.load(os.path.join(FEATURES_DIR, "features_train.npy"))
    X_test  = np.load(os.path.join(FEATURES_DIR, "features_test.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "labels_train.npy"))
    y_test  = np.load(os.path.join(FEATURES_DIR, "labels_test.npy"))
    return X_train, X_test, y_train, y_test


def get_feature_names():
    """Generate meaningful feature names instead of simple indices."""
    names = []
    # GLCM (20 dims)
    from config import GLCM_PROPS
    # Since they are averaged over distances, we have 4 angles * 5 props = 20 dims
    for angle in ["0", "45", "90", "135"]:
        for prop in GLCM_PROPS:
            names.append(f"GLCM_{prop}_angle{angle}")

    # LBP (256 dims)
    for i in range(LBP_DIM):
        names.append(f"LBP_bin{i}")

    # Color (201 dims)
    # HSV (64) + LAB (64) + RGB (64) + moments (9) = 201
    for i in range(64):
        names.append(f"Color_HSV_bin{i}")
    for i in range(64):
        names.append(f"Color_LAB_bin{i}")
    for i in range(64):
        names.append(f"Color_RGB_bin{i}")
    for i in ["mean", "std", "skew"]:
        for ch in ["H", "S", "V"]:
            names.append(f"Color_moment_{i}_{ch}")

    # CNN (512 dims)
    for i in range(CNN_DIM):
        names.append(f"CNN_PCA_comp{i}")

    assert len(names) == TOTAL_DIM, f"Feature names count mismatch: {len(names)} ≠ {TOTAL_DIM}"
    return names


def run_shap_analysis():
    print("\n" + "="*60)
    print("  Historic Image Restoration Phase 2 — Step 3: SHAP Explainability Engine")
    print("="*60)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()
    feature_names = get_feature_names()

    # 2. Load model
    model_path = os.path.join(FEATURES_DIR, "xgb_classifier.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(FEATURES_DIR, "rf_classifier.pkl")

    if os.path.exists(model_path):
        print(f"  Loading trained model for SHAP: {os.path.basename(model_path)}...")
        model = joblib.load(model_path)
    else:
        print("  WARNING: Pre-trained tree model not found. Fitting a fast Random Forest...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

    # 3. Create stratified sub-sample of test set for fast and highly precise calculation
    # We take 10 samples per class for all 18 classes = 180 samples
    sample_indices = []
    for cls_idx in range(len(ALL_CLASSES)):
        match_idx = np.where(y_test == cls_idx)[0]
        if len(match_idx) > 0:
            chosen = np.random.choice(match_idx, size=min(10, len(match_idx)), replace=False)
            sample_indices.extend(chosen)

    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]

    print(f"  Computing SHAP values on stratified sample ({len(X_sample)} paintings)...")
    # TreeExplainer is extremely fast
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Determine SHAP format (depends on XGBoost vs RandomForest outputs)
    # sklearn RF outputs list of shape [num_classes, N, Dims]
    # xgboost output can be [N, Dims, num_classes] or [N, Dims] or list
    if isinstance(shap_values, list):
        # Already class-specific list
        pass
    elif len(shap_values.shape) == 3:
        # Array of shape (N, Dims, num_classes), transpose to list of classes
        shap_values = [shap_values[:, :, c] for c in range(len(ALL_CLASSES))]
    else:
        # Binary or single-output format fallback
        shap_values = [shap_values]

    # Save output folder
    shap_dir = os.path.join(Path(FEATURES_DIR).parent, "visualizations", "shap")
    os.makedirs(shap_dir, exist_ok=True)

    # 4. Generate Global Summary Plot (across all classes)
    # Aggregate SHAP magnitudes across all 18 classes
    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)  # Shape: (N, Dims)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    # Get top 20 features
    global_importance = np.mean(mean_abs_shap, axis=0)
    top_indices = np.argsort(global_importance)[-20:]

    y_pos = np.arange(20)
    ax.barh(y_pos, global_importance[top_indices], color='#00d2c4', edgecolor='#c8b4ff', height=0.6, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_indices], color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#888888')

    ax.set_title('Global Feature Importance — Top 20 Style Attributions', color='white', fontsize=12, pad=15)
    ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)', color='#aaaaaa', fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.grid(axis='x', color='#22223c', linestyle='--', linewidth=0.5)

    global_out = os.path.join(shap_dir, "global_importance.png")
    plt.tight_layout()
    plt.savefig(global_out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved global SHAP plot -> {global_out}")

    # 5. Generate Per-Style Beeswarm Plots for a few illustrative classes
    # We will generate summary plots for two interesting styles: "Baroque" (index 2) and "Impressionism" (index 0)
    for style_name, style_idx in [("impressionism", 0), ("baroque", 2), ("cubism", 1), ("minimalism", 8)]:
        if style_idx < len(shap_values):
            fig = plt.figure(figsize=(9, 5), facecolor='#0f0f1a')
            fig.patch.set_facecolor('#0f0f1a')

            # Render standard SHAP beeswarm / summary plot
            shap.summary_plot(
                shap_values[style_idx],
                X_sample,
                feature_names=feature_names,
                max_display=12,
                show=False
            )

            # Style adjust the figures to fit our premium dark mode
            fig.axes[0].set_facecolor('#0f0f1a')
            fig.axes[0].set_title(f'Style Specific Attributions: {style_name.upper()}', color='white', fontsize=11, pad=15)
            fig.axes[0].tick_params(colors='#888888', labelsize=8)
            fig.axes[0].xaxis.label.set_color('#aaaaaa')
            for spine in fig.axes[0].spines.values():
                spine.set_edgecolor('#333355')

            # Fix y-axis tick colors
            for label in fig.axes[0].get_yticklabels():
                label.set_color('white')

            out_path = os.path.join(shap_dir, f"{style_name}_beeswarm.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close()
            print(f"    -> Saved {style_name} SHAP beeswarm -> {out_path}")


if __name__ == "__main__":
    run_shap_analysis()

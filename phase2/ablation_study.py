"""
Historic Image Restoration Phase 2 — Step 2: Feature Ablation Study
Evaluates classifier performance on 7 different feature slices:
  - Experiment A: Texture only (GLCM + LBP)         → 276 dims
  - Experiment B: Color only (Histograms + moments) → 201 dims
  - Experiment C: CNN only (Deep features after PCA)→ 512 dims
  - Experiment D: Texture + Color                   → 477 dims
  - Experiment E: Color + CNN                       → 713 dims
  - Experiment F: Texture + CNN                     → 788 dims
  - Experiment G: Full Hybrid                       → 989 dims

Saves comparative results as CSV and generates a beautiful summary bar chart.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Import paths from phase1 config
PHASE1_DIR = Path(__file__).parent.parent / "phase1"
sys.path.insert(0, str(PHASE1_DIR))

from config import (
    FEATURES_DIR, GLCM_DIM, LBP_DIM, COLOR_DIM, CNN_DIM, TOTAL_DIM,
    RANDOM_SEED
)


def load_data():
    """Load pre-split numpy arrays from features/."""
    X_train = np.load(os.path.join(FEATURES_DIR, "features_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "features_val.npy"))
    X_test  = np.load(os.path.join(FEATURES_DIR, "features_test.npy"))

    y_train = np.load(os.path.join(FEATURES_DIR, "labels_train.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "labels_val.npy"))
    y_test  = np.load(os.path.join(FEATURES_DIR, "labels_test.npy"))
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_slices():
    """Define slicing indices for the 4 feature groups."""
    # Slices are: GLCM (20), LBP (256), Color (201), CNN (512)
    # Cumulative limits:
    # GLCM  : 0 -> 20
    # LBP   : 20 -> 276
    # Color : 276 -> 477
    # CNN   : 477 -> 989

    # GLCM + LBP
    texture_idx = np.arange(0, GLCM_DIM + LBP_DIM)
    # Color
    color_idx = np.arange(GLCM_DIM + LBP_DIM, GLCM_DIM + LBP_DIM + COLOR_DIM)
    # CNN
    cnn_idx = np.arange(GLCM_DIM + LBP_DIM + COLOR_DIM, TOTAL_DIM)

    experiments = {
        "A: Texture Only":       texture_idx,
        "B: Color Only":         color_idx,
        "C: CNN Only":           cnn_idx,
        "D: Texture + Color":    np.concatenate([texture_idx, color_idx]),
        "E: Color + CNN":        np.concatenate([color_idx, cnn_idx]),
        "F: Texture + CNN":      np.concatenate([texture_idx, cnn_idx]),
        "G: Full Hybrid":        np.arange(0, TOTAL_DIM),
    }
    return experiments


def run_ablation():
    print("\n" + "="*60)
    print("  Historic Image Restoration Phase 2 — Step 2: Feature Ablation Study")
    print("="*60)

    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2. Get slices
    experiments = get_slices()

    results = []

    # Use SVM (RBF) for ablation study as it is fast, highly stable, and represents classical classifiers well
    for exp_name, indices in experiments.items():
        print(f"  Running {exp_name} (dimensions: {len(indices)})...")
        t0 = time.time()

        X_tr_slice = X_train[:, indices]
        X_va_slice = X_val[:, indices]
        X_te_slice = X_test[:, indices]

        model = SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
        model.fit(X_tr_slice, y_train)

        # Predict
        y_pred_val = model.predict(X_va_slice)
        y_pred_test = model.predict(X_te_slice)

        acc_val = accuracy_score(y_val, y_pred_val)
        f1_val = f1_score(y_val, y_pred_val, average='macro')

        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, average='macro')

        elapsed = time.time() - t0
        print(f"    Val F1: {f1_val:.4f} | Test F1: {f1_test:.4f} ({elapsed:.1f}s)")

        results.append({
            "Experiment": exp_name,
            "Dimensions": len(indices),
            "Val_Accuracy": float(acc_val),
            "Val_F1_Macro": float(f1_val),
            "Test_Accuracy": float(acc_test),
            "Test_F1_Macro": float(f1_test),
            "Train_Time_sec": float(elapsed)
        })

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(FEATURES_DIR, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  [OK] Saved ablation results -> {csv_path}")

    # Generate visual bar chart
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    x_labels = df["Experiment"].values
    test_f1 = df["Test_F1_Macro"].values
    val_f1 = df["Val_F1_Macro"].values

    x = np.arange(len(x_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, val_f1, width, label='Val F1 (Macro)', color='#7c6af7', alpha=0.85)
    rects2 = ax.bar(x + width/2, test_f1, width, label='Test F1 (Macro)', color='#00d2c4', alpha=0.85)

    ax.set_title('Feature Ablation Study — Historic Image Restoration Style Classification', color='white', fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15, color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#888888')
    ax.set_ylabel('Macro F1-Score', color='#aaaaaa', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor='#181828', edgecolor='#333355', labelcolor='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    # Gridlines
    ax.grid(axis='y', color='#22223c', linestyle='--', linewidth=0.5)

    # Annotate bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='white', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    vis_out = os.path.join(Path(FEATURES_DIR).parent, "visualizations", "ablation_study.png")
    os.makedirs(os.path.dirname(vis_out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(vis_out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved ablation bar chart -> {vis_out}")


if __name__ == "__main__":
    run_ablation()

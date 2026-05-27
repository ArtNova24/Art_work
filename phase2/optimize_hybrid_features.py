"""
Historic Image Restoration Phase 2.5 — Feature Normalization & Validation-Tuned Blending
1. Slices features into Handcrafted (GLCM + LBP + Color) and CNN.
2. Normalizes each block independently using Z-score (StandardScaler) fitted on train.
3. Computes global SHAP feature importances using a fast Random Forest.
4. Performs a grid search over beta (0.5 to 0.95) to balance Handcrafted vs CNN features.
   - Weight handcrafted by sqrt(1 - beta)
   - Weight CNN by sqrt(beta)
   - Evaluate performance on Validation set using SVM.
5. Selects the best beta, applies the final scaled weights, and overwrites active features.
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import shap

# Add parent path to path
PHASE1_DIR = Path(__file__).parent.parent / "phase1"
sys.path.insert(0, str(PHASE1_DIR))

from config import (
    FEATURES_DIR, TOTAL_DIM, GLCM_DIM, LBP_DIM, COLOR_DIM, CNN_DIM, RANDOM_SEED
)

def load_data():
    """Load raw split arrays."""
    train_backup = os.path.join(FEATURES_DIR, "features_train_raw.npy")
    val_backup = os.path.join(FEATURES_DIR, "features_val_raw.npy")
    test_backup = os.path.join(FEATURES_DIR, "features_test_raw.npy")
    
    # Fallback to standard files if raw backups don't exist yet
    if not os.path.exists(train_backup):
        print("  Creating backups of raw feature arrays...")
        np.save(train_backup, np.load(os.path.join(FEATURES_DIR, "features_train.npy")))
        np.save(val_backup, np.load(os.path.join(FEATURES_DIR, "features_val.npy")))
        np.save(test_backup, np.load(os.path.join(FEATURES_DIR, "features_test.npy")))
        
    X_train = np.load(train_backup)
    X_val   = np.load(val_backup)
    X_test  = np.load(test_backup)
    
    y_train = np.load(os.path.join(FEATURES_DIR, "labels_train.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "labels_val.npy"))
    y_test  = np.load(os.path.join(FEATURES_DIR, "labels_test.npy"))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    print("\n" + "="*60)
    print("  Historic Image Restoration Phase 2.5: Optimized Feature Blending (Beta Grid Search)")
    print("="*60)
    
    # 1. Load raw data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"  Loaded raw features: Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Define block slices
    # GLCM (20), LBP (256), Color (201), CNN (512)
    slices = [
        ("GLCM", 0, GLCM_DIM),
        ("LBP", GLCM_DIM, GLCM_DIM + LBP_DIM),
        ("Color", GLCM_DIM + LBP_DIM, GLCM_DIM + LBP_DIM + COLOR_DIM),
        ("CNN", GLCM_DIM + LBP_DIM + COLOR_DIM, TOTAL_DIM)
    ]
    
    # 2. Per-Block Normalization
    print("\n  [1/5] Applying Per-Block Z-score Normalization...")
    X_train_norm = np.zeros_like(X_train)
    X_val_norm = np.zeros_like(X_val)
    X_test_norm = np.zeros_like(X_test)
    
    scalers = {}
    
    for name, start, end in slices:
        print(f"    Normalizing {name:6s} block (dims {start:3d} to {end:3d})...")
        scaler = StandardScaler()
        X_train_norm[:, start:end] = scaler.fit_transform(X_train[:, start:end])
        X_val_norm[:, start:end] = scaler.transform(X_val[:, start:end])
        X_test_norm[:, start:end] = scaler.transform(X_test[:, start:end])
        scalers[name] = scaler
        
    # Save scalers for Phase 3/4 online preprocessing
    scalers_path = os.path.join(FEATURES_DIR, "block_scalers.pkl")
    joblib.dump(scalers, scalers_path)
    print(f"  [OK] Block scalers saved -> {scalers_path}")
    
    # 3. Fit Model to Extract SHAP values
    print("\n  [2/5] Fitting Random Forest classifier on normalized train features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train_norm, y_train)
    
    print("  [3/5] Running SHAP TreeExplainer on stratified subset...")
    sample_indices = []
    num_classes = len(np.unique(y_train))
    for cls_idx in range(num_classes):
        match_idx = np.where(y_train == cls_idx)[0]
        if len(match_idx) > 0:
            chosen = np.random.choice(match_idx, size=min(15, len(match_idx)), replace=False)
            sample_indices.extend(chosen)
            
    X_sample = X_train_norm[sample_indices]
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        pass
    elif len(shap_values.shape) == 3:
        shap_values = [shap_values[:, :, c] for c in range(num_classes)]
    else:
        shap_values = [shap_values]
        
    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0) # Shape: (N, Dims)
    global_importance = np.mean(mean_abs_shap, axis=0) # Shape: (Dims,)
    
    # Separate Handcrafted vs CNN indices
    hc_end = GLCM_DIM + LBP_DIM + COLOR_DIM  # 477
    
    hc_importance = global_importance[:hc_end]
    cnn_importance = global_importance[hc_end:]
    
    # Normalize importances within each group so they average to 1.0
    hc_weights = hc_importance / (np.mean(hc_importance) + 1e-8)
    cnn_weights = cnn_importance / (np.mean(cnn_importance) + 1e-8)
    
    # Clip weights to prevent individual features from dominating or disappearing
    hc_weights = np.clip(hc_weights, 0.1, 3.0)
    cnn_weights = np.clip(cnn_weights, 0.1, 3.0)
    
    # 4. Grid Search over Beta
    print("\n  [4/5] Running Grid Search over Beta (CNN blending ratio)...")
    beta_candidates = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    best_beta = None
    best_val_f1 = 0.0
    best_weights = None
    
    for beta in beta_candidates:
        # Scale each group by its beta factor
        w_hc = np.sqrt(1 - beta) * hc_weights
        w_cnn = np.sqrt(beta) * cnn_weights
        candidate_weights = np.concatenate([w_hc, w_cnn])
        
        # Scale features
        X_tr_scaled = X_train_norm * candidate_weights
        X_va_scaled = X_val_norm * candidate_weights
        
        # Train a fast SVM model
        clf = SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
        clf.fit(X_tr_scaled, y_train)
        
        # Predict & Evaluate
        y_pred = clf.predict(X_va_scaled)
        val_f1 = f1_score(y_val, y_pred, average='macro')
        print(f"    Beta: {beta:.2f} | SVM Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_beta = beta
            best_weights = candidate_weights
            
    print(f"\n  [BEST] Optimal Beta found: {best_beta:.2f} (Validation Macro F1: {best_val_f1:.4f})")
    
    # Save the optimal weights
    weights_path = os.path.join(FEATURES_DIR, "shap_scaling_weights.npy")
    np.save(weights_path, best_weights)
    print(f"  [OK] Optimal SHAP scaling weights saved -> {weights_path}")
    
    # Print average weight per block under optimal beta
    print("\n  Summary of optimal SHAP block weights:")
    for name, start, end in slices:
        block_w = best_weights[start:end]
        print(f"    {name:6s} | Dims: {end-start:3d} | Mean Weight: {np.mean(block_w):.4f} | Max: {np.max(block_w):.4f} | Min: {np.min(block_w):.4f}")
        
    # 5. Apply optimal scaling and overwrite active files
    print("\n  [5/5] Overwriting active feature matrices with optimal representation...")
    X_train_final = X_train_norm * best_weights
    X_val_final = X_val_norm * best_weights
    X_test_final = X_test_norm * best_weights
    
    train_path = os.path.join(FEATURES_DIR, "features_train.npy")
    val_path = os.path.join(FEATURES_DIR, "features_val.npy")
    test_path = os.path.join(FEATURES_DIR, "features_test.npy")
    
    np.save(train_path, X_train_final)
    np.save(val_path, X_val_final)
    np.save(test_path, X_test_final)
    
    print("  [OK] Overwrote features_train.npy, features_val.npy, features_test.npy")
    print("  [SUCCESS] Blending optimization completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

"""
ANTIGRAVITY Phase 1 — Master Orchestration Script
Runs all Phase 1 steps in order:
  1. Dataset preparation (expects data already downloaded)
  2. GLCM extraction        → features/glcm_features.npy   (N, 20)
  3. LBP extraction         → features/lbp_features.npy    (N, 256)
  4. Color extraction       → features/color_features.npy  (N, 201)
  5. CNN extraction + PCA   → features/cnn_features.npy    (N, 512)
  6. Assembly + splitting   → features/features_*.npy, labels_*.npy
  7. Visualisations         → visualizations/
  8. Final report           → phase1_report.txt
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Make phase1 importable
PHASE1_DIR = Path(__file__).parent
sys.path.insert(0, str(PHASE1_DIR))

from config import (
    DATA_DIR, FEATURES_DIR, VIS_DIR,
    WIKIART_CLASSES, ALL_CLASSES, TOTAL_DIM,
    GLCM_DIM, LBP_DIM, COLOR_DIM, CNN_DIM,
)


def header(title, char="="):
    print(f"\n{'*'*60}")
    print(f"  {title}")
    print(f"{'*'*60}")


def check_step(name, condition, detail=""):
    status = "✓" if condition else "✗ FAILED"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    return condition


def step1_verify_dataset():
    header("Step 1 — Dataset Verification")
    index_path = os.path.join(FEATURES_DIR, "image_index.json")
    if not os.path.exists(index_path):
        print("  ERROR: image_index.json not found. Run fast_download.py first.")
        sys.exit(1)
    with open(index_path) as f:
        image_index = json.load(f)
    print(f"  Total images in index: {len(image_index)}")
    per_class = {}
    for item in image_index:
        per_class[item['class']] = per_class.get(item['class'], 0) + 1
    for cls in ALL_CLASSES:
        n = per_class.get(cls, 0)
        print(f"    {cls:30s}: {n}")
    return image_index


def step2_glcm(image_index):
    header("Step 2 — GLCM Feature Extraction")
    out_path = os.path.join(FEATURES_DIR, "glcm_features.npy")
    if os.path.exists(out_path):
        arr = np.load(out_path)
        if arr.shape == (len(image_index), GLCM_DIM):
            print(f"  Cached: {arr.shape} ✓")
            return arr
        print("  Shape mismatch — re-extracting.")

    from extract_glcm import extract_glcm_all
    t0 = time.time()
    feats = extract_glcm_all(image_index)
    np.save(out_path, feats)
    print(f"  ✓ Saved {feats.shape} in {time.time()-t0:.1f}s  → {out_path}")
    return feats


def step3_lbp(image_index):
    header("Step 3 — LBP Feature Extraction")
    out_path = os.path.join(FEATURES_DIR, "lbp_features.npy")
    if os.path.exists(out_path):
        arr = np.load(out_path)
        if arr.shape == (len(image_index), LBP_DIM):
            print(f"  Cached: {arr.shape} ✓")
            return arr
        print("  Shape mismatch — re-extracting.")

    from extract_lbp import extract_lbp_all
    t0 = time.time()
    feats = extract_lbp_all(image_index)
    np.save(out_path, feats)
    print(f"  ✓ Saved {feats.shape} in {time.time()-t0:.1f}s  → {out_path}")
    return feats


def step4_color(image_index):
    header("Step 4 — Color Feature Extraction")
    out_path = os.path.join(FEATURES_DIR, "color_features.npy")
    if os.path.exists(out_path):
        arr = np.load(out_path)
        if arr.shape == (len(image_index), COLOR_DIM):
            print(f"  Cached: {arr.shape} ✓")
            return arr
        print("  Shape mismatch — re-extracting.")

    from extract_color import extract_color_all
    t0 = time.time()
    feats = extract_color_all(image_index)
    np.save(out_path, feats)
    print(f"  ✓ Saved {feats.shape} in {time.time()-t0:.1f}s  → {out_path}")
    return feats


def step5_cnn(image_index):
    header("Step 5 — CNN Feature Extraction (DINOv2 + ResNet-50 + PCA)")
    out_path = os.path.join(FEATURES_DIR, "cnn_features.npy")
    if os.path.exists(out_path):
        arr = np.load(out_path)
        if arr.shape == (len(image_index), CNN_DIM):
            print(f"  Cached: {arr.shape} ✓")
            return arr
        print("  Shape mismatch — re-extracting.")

    from extract_cnn import load_models, extract_raw_cnn_features, apply_pca
    t0 = time.time()
    dino, resnet = load_models()
    raw = extract_raw_cnn_features(image_index, dino, resnet)
    raw_path = os.path.join(FEATURES_DIR, "cnn_raw_features.npy")
    np.save(raw_path, raw)
    feats, _ = apply_pca(raw, fit=True)
    np.save(out_path, feats)
    print(f"  ✓ Saved {feats.shape} in {time.time()-t0:.1f}s  → {out_path}")
    return feats


def step6_assemble(image_index, glcm, lbp, color, cnn):
    header("Step 6 — Feature Assembly & Splitting")
    from assemble_features import (
        verify_dimensions, assemble_hybrid, split_data,
        compute_weights, save_feature_summary, save_all_outputs
    )
    labels = np.array([item['class_idx'] for item in image_index], dtype=np.int32)

    verify_dimensions(glcm, lbp, color, cnn)
    hybrid = assemble_hybrid(glcm, lbp, color, cnn)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(hybrid, labels)
    weights = compute_weights(y_train)
    save_feature_summary(hybrid, labels)
    save_all_outputs(X_train, X_val, X_test, y_train, y_val, y_test, weights)
    return hybrid, labels


def step7_visualise(image_index):
    header("Step 7 — Visualisations")
    os.makedirs(VIS_DIR, exist_ok=True)
    from visualize import generate_all_visualisations
    generate_all_visualisations(image_index)
    print(f"  ✓ Visualisations saved to {VIS_DIR}")


def final_report(image_index, glcm, lbp, color, cnn):
    header("Phase 1 — Final Report", char="*")

    lines = []
    lines.append("ANTIGRAVITY Phase 1 — Feature Extraction Report")
    lines.append("=" * 60)
    lines.append(f"Total images: {len(image_index)}")
    lines.append(f"Feature dims: GLCM={GLCM_DIM}, LBP={LBP_DIM}, Color={COLOR_DIM}, CNN={CNN_DIM}")
    lines.append(f"Total hybrid dim: {TOTAL_DIM}")
    lines.append("")

    # Output files
    files_to_check = [
        "features_train.npy", "features_val.npy", "features_test.npy",
        "labels_train.npy", "labels_val.npy", "labels_test.npy",
        "class_weights.npy", "pca_model.pkl", "cnn_scaler.pkl",
        "style_mapping.json", "image_index.json", "feature_summary.csv"
    ]
    lines.append("Output files:")
    for fname in files_to_check:
        p = os.path.join(FEATURES_DIR, fname)
        ok = os.path.exists(p)
        lines.append(f"  {'✓' if ok else '✗'} {fname}")

    # Shape checks
    lines.append("")
    lines.append("Feature shape checks:")
    shapes = {
        "glcm":  (glcm.shape,  (len(image_index), GLCM_DIM)),
        "lbp":   (lbp.shape,   (len(image_index), LBP_DIM)),
        "color": (color.shape, (len(image_index), COLOR_DIM)),
        "cnn":   (cnn.shape,   (len(image_index), CNN_DIM)),
    }
    all_ok = True
    for name, (actual, expected) in shapes.items():
        ok = actual == expected
        all_ok = all_ok and ok
        lines.append(f"  {'✓' if ok else '✗'} {name}: {actual} (expected {expected})")

    lines.append("")
    lines.append("=" * 60)
    lines.append("PHASE 1 STATUS: " + ("SUCCESS ✓" if all_ok else "PARTIAL / CHECK ABOVE"))

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(Path(FEATURES_DIR).parent, "phase1_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved → {report_path}")


if __name__ == "__main__":
    t_start = time.time()
    print("\n" + "*"*60)
    print("  ANTIGRAVITY Phase 1 — Full Pipeline")
    print("*"*60)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    image_index = step1_verify_dataset()
    glcm  = step2_glcm(image_index)
    lbp   = step3_lbp(image_index)
    color = step4_color(image_index)
    cnn   = step5_cnn(image_index)
    hybrid, labels = step6_assemble(image_index, glcm, lbp, color, cnn)
    step7_visualise(image_index)
    final_report(image_index, glcm, lbp, color, cnn)

    elapsed = time.time() - t_start
    print(f"\n  Total Phase 1 time: {elapsed/60:.1f} minutes")
    print("  ✓ Phase 1 COMPLETE — Ready for Phase 2!\n")

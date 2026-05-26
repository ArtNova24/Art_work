"""
ANTIGRAVITY Phase 2 — Master Orchestration Script
Runs all Phase 2 steps in order:
  1. Classifiers training   → features/classifier_metrics.json, features/style_predictor.pkl
  2. Ablation study         → features/ablation_results.csv, visualizations/ablation_study.png
  3. SHAP feature analysis  → visualizations/shap/
  4. t-SNE representation   → visualizations/tsne_clustering.png
  5. Final Phase 2 report   → phase2_report.txt
"""
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Make phase2 and phase1 importable
PHASE2_DIR = Path(__file__).parent
sys.path.insert(0, str(PHASE2_DIR))

from train_classifiers import main as run_training
from ablation_study import run_ablation
from shap_analysis import run_shap_analysis
from tsne_analysis import run_tsne_analysis

from config import FEATURES_DIR, TOTAL_DIM


def header(title, char="="):
    print(f"\n{char*60}")
    print(f"  {title}")
    print(f"{char*60}")


def compile_final_report():
    header("Phase 2 — Generating Summary Report", char="*")

    # Load metrics
    metrics_path = os.path.join(FEATURES_DIR, "classifier_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"  ERROR: {metrics_path} not found. Cannot compile report.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Load ablation results
    ablation_path = os.path.join(FEATURES_DIR, "ablation_results.csv")
    ablation_df = None
    if os.path.exists(ablation_path):
        ablation_df = pd.read_csv(ablation_path)

    lines = []
    lines.append("ANTIGRAVITY Phase 2 — Classifier & Explainability Report")
    lines.append("=" * 60)
    lines.append(f"Hybrid vector input dimensions: {TOTAL_DIM}")
    lines.append("")

    lines.append("1. Classifier Performance Comparison (Test Partition):")
    lines.append("-" * 60)
    lines.append(f"  {'Model':12s} | {'Accuracy':10s} | {'Macro F1':10s} | {'Weighted F1':11s}")
    lines.append(f"  {'-'*12} | {'-'*10} | {'-'*10} | {'-'*11}")

    for name, res in metrics.items():
        t_res = res["test"]
        lines.append(f"  {name.upper():12s} | {t_res['accuracy']:10.4f} | {t_res['f1_macro']:10.4f} | {t_res['f1_weighted']:11.4f}")

    lines.append("")

    if ablation_df is not None:
        lines.append("2. Feature Ablation Study Results (Macro F1-Score):")
        lines.append("-" * 60)
        lines.append(f"  {'Experiment Slice':25s} | {'Dims':5s} | {'Val F1':8s} | {'Test F1':8s}")
        lines.append(f"  {'-'*25} | {'-'*5} | {'-'*8} | {'-'*8}")
        for _, row in ablation_df.iterrows():
            lines.append(f"  {row['Experiment']:25s} | {int(row['Dimensions']):5d} | {row['Val_F1_Macro']:8.4f} | {row['Test_F1_Macro']:8.4f}")

    lines.append("")

    # Output files
    files_to_check = [
        "svm_classifier.pkl", "rf_classifier.pkl", "xgb_classifier.pkl",
        "mlp_classifier.pt", "cnn_end2end_classifier.pt", "style_predictor.pkl",
        "classifier_metrics.json", "ablation_results.csv"
    ]
    lines.append("Generated Deliverables:")
    for fname in files_to_check:
        p = os.path.join(FEATURES_DIR, fname)
        ok = os.path.exists(p)
        lines.append(f"  {'[OK]' if ok else '[MISSING]'} {fname}")

    lines.append("")
    lines.append("Explainability Deliverables:")
    vis_dir = Path(FEATURES_DIR).parent / "visualizations"
    plots = [
        "ablation_study.png", "tsne_clustering.png",
        "shap/global_importance.png", "shap/impressionism_beeswarm.png",
        "shap/baroque_beeswarm.png"
    ]
    for plot in plots:
        p = os.path.join(vis_dir, plot)
        ok = os.path.exists(p)
        lines.append(f"  {'[OK]' if ok else '[MISSING]'} visualizations/{plot}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("PHASE 2 STATUS: SUCCESS")

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(Path(FEATURES_DIR).parent, "phase2_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  [OK] Report saved -> {report_path}")


if __name__ == "__main__":
    t_start = time.time()
    header("ANTIGRAVITY Phase 2 — Master Orchestration Script", char="*")

    # Step 1: Train all 5 classifiers
    header("Step 1/4: Classifier Training Engine")
    run_training()

    # Step 2: Run ablation study
    header("Step 2/4: Slicing Feature Ablation Study")
    run_ablation()

    # Step 3: Run SHAP feature attribution
    header("Step 3/4: SHAP Explainability Calculations")
    run_shap_analysis()

    # Step 4: Run t-SNE representation projection
    header("Step 4/4: t-SNE Dimensionality Projection")
    run_tsne_analysis()

    # Compile final report
    compile_final_report()

    elapsed = time.time() - t_start
    print(f"\n  Total Phase 2 execution time: {elapsed/60:.1f} minutes")
    print("  [SUCCESS] Phase 2 COMPLETE — Ready for Phase 3!\n")

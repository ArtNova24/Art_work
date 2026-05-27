"""
ANTIGRAVITY Phase 4 — Configuration
Defines paths, constants, and evaluation settings for Phase 4.
All comments and strings are kept strictly in ASCII.
"""
import os
import sys

# Project root paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "wikiart")
FEATURES_DIR = os.path.join(ROOT, "features")
VIS_DIR = os.path.join(ROOT, "visualizations")
RECON_DIR = os.path.join(VIS_DIR, "reconstructions")

# Phase 4 specific deliverables
PHASE4_DIR = os.path.join(ROOT, "phase4")
PHASE4_METRICS_PATH = os.path.join(FEATURES_DIR, "phase4_metrics.json")
REPORT_PATH = os.path.join(ROOT, "phase4_report.txt")

# Image sizes & parameters
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 14 * 14 = 196

# Evaluation Settings
RANDOM_SEED = 42
EVAL_BATCH_SIZE = 16
FID_BATCH_SIZE = 32

# Verify and create folders
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(os.path.join(VIS_DIR, "phase4_visualizations"), exist_ok=True)

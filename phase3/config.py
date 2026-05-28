"""
Historic Image Restoration Phase 3 — central configuration for Style-Conditioned I-JEPA.
Defines hyperparameters, hardware settings, and file paths.
All prints and comments are kept strictly in ASCII.
"""
import os
import torch

# -- Project root paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "wikiart")
FEATURES_DIR = os.path.join(ROOT, "features")
VIS_DIR = os.path.join(ROOT, "visualizations")
RECON_DIR = os.path.join(VIS_DIR, "reconstructions")

# Create output directories if they do not exist
os.makedirs(RECON_DIR, exist_ok=True)

# -- Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Random Seed
RANDOM_SEED = 42

# -- Image & Patching Settings
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 14 * 14 = 196
CHANNELS = 3

# -- Masking settings (Block-wise mask ratio)
MASK_RATIO_MIN = 0.40
MASK_RATIO_MAX = 0.60

# -- Model Architecture Dimensions
HYBRID_DIM = 989     # Vector output from Phase 2.5
STYLE_EMBED_DIM = 256
VIT_EMBED_DIM = 768

# -- Context/Target Encoder (ViT-Small equivalent)
ENC_DEPTH = 6
ENC_HEADS = 8
ENC_MLP_RATIO = 4.0
ENC_DROPOUT = 0.1

# -- Style-Conditioned Predictor (Transformer Decoder)
PRED_DEPTH = 4
PRED_HEADS = 8
PRED_MLP_RATIO = 4.0
PRED_DROPOUT = 0.1

# -- Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4

# -- EMA Momentum (Target Encoder tracking)
EMA_MOMENTUM_BASE = 0.996
EMA_MOMENTUM_MAX = 1.0

# -- Loss Coefficients
LATENT_LOSS_WEIGHT = 1.0
PIXEL_LOSS_WEIGHT = 0.5

# -- Model Checkpoint Paths
PROJECTOR_PATH = os.path.join(FEATURES_DIR, "jepa_style_projector.pt")
ENCODER_PATH = os.path.join(FEATURES_DIR, "jepa_context_encoder.pt")
PREDICTOR_PATH = os.path.join(FEATURES_DIR, "jepa_predictor.pt")
DECODER_PATH = os.path.join(FEATURES_DIR, "jepa_pixel_decoder.pt")
TARGET_ENCODER_PATH = os.path.join(FEATURES_DIR, "jepa_target_encoder.pt")

"""
ANTIGRAVITY Phase 1 — Central Configuration
All paths, constants, and hyperparameters in one place.
"""
import os

# ── Project root ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(ROOT, "data", "wikiart")
INDIAN_DIR      = os.path.join(ROOT, "archive (5) - Copy")
FEATURES_DIR    = os.path.join(ROOT, "features")
VIS_DIR         = os.path.join(ROOT, "visualizations")

# ── WikiArt-10 style mapping (HuggingFace label → our class name) ────────────
WIKIART_STYLE_MAP = {
    "Impressionism":             "impressionism",
    "Cubism":                    "cubism",
    "Baroque":                   "baroque",
    "Abstract_Expressionism":    "abstract_expressionism",
    "Surrealism":                "surrealism",
    "Early_Renaissance":         "renaissance",
    "High_Renaissance":          "renaissance",
    "Northern_Renaissance":      "renaissance",
    "Romanticism":               "romanticism",
    "Art_Nouveau_(Modern)":      "art_nouveau",
    "Art_Nouveau":               "art_nouveau",
    "Minimalism":                "minimalism",
    "Pop_Art":                   "pop_art",
}

WIKIART_CLASSES = [
    "impressionism",
    "cubism",
    "baroque",
    "abstract_expressionism",
    "surrealism",
    "renaissance",
    "romanticism",
    "art_nouveau",
    "minimalism",
    "pop_art",
]

# ── Indian Art-8 style mapping (folder name → our class name) ─────────────────
INDIAN_FOLDER_MAP = {
    "gond painting":      "gond",
    "kalighat painting":  "kalighat",
    "kangra painting":    "kangra",
    "kerala mural":       "kerala_mural",
    "madhubani painting": "madhubani",
    "mandana art drawing":"mandana",
    "pichwai painting":   "pichwai",
    "warli painting":     "warli",
}

INDIAN_CLASSES = [
    "gond",
    "kalighat",
    "kangra",
    "kerala_mural",
    "madhubani",
    "mandana",
    "pichwai",
    "warli",
]

# ── All 17 classes (WikiArt first, then Indian) ──────────────────────────────
ALL_CLASSES = WIKIART_CLASSES + INDIAN_CLASSES  # indices 0-9 WikiArt, 10-17 Indian
CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

# ── Image settings ───────────────────────────────────────────────────────────
IMG_SIZE    = 224
MAX_WIKIART_PER_CLASS = 500   # cap WikiArt download to 500/class

# ── Feature dimensions ───────────────────────────────────────────────────────
GLCM_DIM  = 20
LBP_DIM   = 256
COLOR_DIM = 201
CNN_DIM   = 512
TOTAL_DIM = GLCM_DIM + LBP_DIM + COLOR_DIM + CNN_DIM  # 989

# ── GLCM settings ────────────────────────────────────────────────────────────
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES    = [0, 0.785398, 1.5708, 2.35619]   # 0, 45, 90, 135 degrees
GLCM_PROPS     = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']

# ── LBP settings ────────────────────────────────────────────────────────────
LBP_P      = 8
LBP_R      = 1
LBP_BINS   = 256     # 2^P bins using default method

# ── Color settings ───────────────────────────────────────────────────────────
HIST_BINS  = 64      # bins per color space (split across channels to = 64 total)
COLOR_KMEANS_K = 5   # for dominant palette (visualisation only)

# ── CNN / PCA settings ───────────────────────────────────────────────────────
CNN_BATCH_SIZE = 32
PCA_COMPONENTS = 512

# ── Train/val/test split ─────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

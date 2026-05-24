"""
ANTIGRAVITY Phase 1 — Step 8: Feature Visualisation
Generates sample visualisations for one image per style class:
  - GLCM co-occurrence matrix heatmap
  - LBP histogram
  - Color palette swatches (KMeans dominant colours)

Outputs saved to visualizations/<style>_<type>.png
"""

import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_DIR, VIS_DIR, ALL_CLASSES, COLOR_KMEANS_K,
    GLCM_DISTANCES, GLCM_ANGLES, IMG_SIZE, RANDOM_SEED
)
from preprocessing import safe_load

random.seed(RANDOM_SEED)

# Pretty style display names
DISPLAY_NAMES = {
    "impressionism":          "Impressionism",
    "cubism":                 "Cubism",
    "baroque":                "Baroque",
    "abstract_expressionism": "Abstract Expressionism",
    "surrealism":             "Surrealism",
    "renaissance":            "Renaissance",
    "romanticism":            "Romanticism",
    "art_nouveau":            "Art Nouveau",
    "minimalism":             "Minimalism",
    "pop_art":                "Pop Art",
    "gond":                   "Gond Painting",
    "kalighat":               "Kalighat Painting",
    "kangra":                 "Kangra Painting",
    "kerala_mural":           "Kerala Mural",
    "madhubani":              "Madhubani Painting",
    "mandana":                "Mandana Art",
    "pichwai":                "Pichwai Painting",
    "warli":                  "Warli Painting",
}


def pick_sample_per_class(image_index: list) -> dict:
    """Pick one random image path per class."""
    by_class = {}
    for item in image_index:
        cls = item['class']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(item['path'])
    samples = {}
    for cls, paths in by_class.items():
        samples[cls] = random.choice(paths)
    return samples


def plot_glcm(gray_img: np.ndarray, cls_name: str):
    """Plot GLCM co-occurrence matrix at angle=0, distance=1."""
    gray_64 = (gray_img // 4).astype(np.uint8)
    glcm = graycomatrix(gray_64, distances=[1], angles=[0],
                        levels=64, symmetric=True, normed=True)
    matrix = glcm[:, :, 0, 0]

    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    im = ax.imshow(np.log1p(matrix * 1000), cmap='magma', aspect='auto')
    ax.set_title(f'GLCM — {DISPLAY_NAMES.get(cls_name, cls_name)}',
                 color='white', fontsize=11, pad=10)
    ax.set_xlabel('Grey level j', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Grey level i', color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#888888')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#888888')
    cbar.set_label('log(1 + freq × 1000)', color='#aaaaaa', fontsize=8)

    out = os.path.join(VIS_DIR, f"{cls_name}_glcm.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return out


def plot_lbp_histogram(gray_img: np.ndarray, cls_name: str):
    """Plot 256-bin LBP histogram."""
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray_img, P=8, R=1, method='default')
    hist, bins = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=False)
    hist = hist / hist.sum()

    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    x = np.arange(256)
    ax.bar(x, hist, color='#7c6af7', width=1.0, alpha=0.85)
    ax.plot(x, hist, color='#c8b4ff', linewidth=0.6, alpha=0.7)

    ax.set_title(f'LBP Histogram — {DISPLAY_NAMES.get(cls_name, cls_name)}',
                 color='white', fontsize=11, pad=10)
    ax.set_xlabel('LBP pattern value (0–255)', color='#aaaaaa', fontsize=9)
    ax.set_ylabel('Normalised frequency',       color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.set_xlim(0, 255)

    out = os.path.join(VIS_DIR, f"{cls_name}_lbp.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return out


def plot_color_palette(img_rgb: np.ndarray, cls_name: str):
    """Plot original image alongside its K dominant colours."""
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=COLOR_KMEANS_K, random_state=RANDOM_SEED, n_init='auto')
    km.fit(pixels)
    palette = km.cluster_centers_.astype(np.uint8)  # (K, 3)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor='#0f0f1a',
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f'Color Palette — {DISPLAY_NAMES.get(cls_name, cls_name)}',
                 color='white', fontsize=12)

    # Left: original image
    axes[0].imshow(img_rgb)
    axes[0].set_title('Sample Image', color='#aaaaaa', fontsize=9)
    axes[0].axis('off')

    # Right: colour swatches
    axes[1].set_facecolor('#0f0f1a')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, COLOR_KMEANS_K)
    axes[1].axis('off')
    axes[1].set_title(f'Top {COLOR_KMEANS_K} Colours', color='#aaaaaa', fontsize=9)

    for i, colour in enumerate(palette):
        hex_col = '#{:02x}{:02x}{:02x}'.format(*colour)
        rect = mpatches.FancyBboxPatch(
            (0.05, i + 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.02",
            facecolor=np.array(colour) / 255.0,
            edgecolor='white', linewidth=0.5
        )
        axes[1].add_patch(rect)
        axes[1].text(0.5, i + 0.5, hex_col, color='white', fontsize=7,
                     ha='center', va='center',
                     fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.4, pad=1, linewidth=0))

    out = os.path.join(VIS_DIR, f"{cls_name}_palette.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    return out


def generate_all_visualisations(image_index: list):
    """Generate all three visualisation types for one sample per class."""
    os.makedirs(VIS_DIR, exist_ok=True)
    samples = pick_sample_per_class(image_index)

    results = []
    for cls_name in ALL_CLASSES:
        if cls_name not in samples:
            print(f"  WARNING: No sample found for class '{cls_name}'")
            continue

        path = samples[cls_name]
        print(f"  [{cls_name:30s}] {os.path.basename(path)}")

        result = safe_load(path, IMG_SIZE)
        if result is None:
            print(f"    → SKIPPED (load failed)")
            continue

        img_rgb, img_hsv, img_lab, gray, _ = result

        try:
            glcm_out    = plot_glcm(gray, cls_name)
            lbp_out     = plot_lbp_histogram(gray, cls_name)
            palette_out = plot_color_palette(img_rgb, cls_name)
            results.append((cls_name, glcm_out, lbp_out, palette_out))
            print(f"    → GLCM, LBP, Palette saved")
        except Exception as e:
            print(f"    → ERROR: {e}")

    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ANTIGRAVITY Phase 1 — Step 8: Visualisations")
    print("="*60)

    index_path = os.path.join(FEATURES_DIR, "image_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("Run download_dataset.py first to generate image_index.json")

    with open(index_path) as f:
        image_index = json.load(f)

    generate_all_visualisations(image_index)
    print(f"\n  ✓ All visualisations saved to: {VIS_DIR}")

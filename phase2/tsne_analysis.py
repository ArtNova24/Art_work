"""
ANTIGRAVITY Phase 2 — Step 4: Representation t-SNE Analysis
Applies t-SNE dimensionality reduction to the hybrid feature matrix
to project the multi-modal art representations down to 2D.

Generates a premium, high-resolution style clustering plot in dark theme
to visually inspect style separation and cluster proximity.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE

# Import paths from phase1 config
PHASE1_DIR = Path(__file__).parent.parent / "phase1"
sys.path.insert(0, str(PHASE1_DIR))

from config import FEATURES_DIR, ALL_CLASSES, RANDOM_SEED

# Pretty style display names for plotting
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


def run_tsne_analysis():
    print("\n" + "="*60)
    print("  ANTIGRAVITY Phase 2 — Step 4: Representation t-SNE Analysis")
    print("="*60)

    # 1. Load data
    # We use the test partition (887 samples) as it is lightweight and captures
    # the out-of-sample generalization space perfectly!
    X_test = np.load(os.path.join(FEATURES_DIR, "features_test.npy"))
    y_test = np.load(os.path.join(FEATURES_DIR, "labels_test.npy"))

    print(f"  Computing t-SNE projection on {X_test.shape[0]} hybrid vectors...")
    t0 = pd.Timestamp.now()

    # Perplexity 30 is standard for cluster visualization
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        early_exaggeration=12.0,
        learning_rate='auto',
        init='pca',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    X_2d = tsne.fit_transform(X_test)
    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    print(f"  ✓ t-SNE fit completed in {elapsed:.1f}s")

    # 2. Build plotting DataFrame
    df = pd.DataFrame(X_2d, columns=["t-SNE Dimension 1", "t-SNE Dimension 2"])
    df["Style Class"] = [DISPLAY_NAMES.get(ALL_CLASSES[idx], ALL_CLASSES[idx]) for idx in y_test]
    # Source category to check separation of Indian art vs WikiArt
    df["Category"] = ["Indian Heritage Art" if idx >= 10 else "Western WikiArt-10" for idx in y_test]

    # 3. Create high-resolution dark-themed scatter plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    # Color palette - use a bright, high-contrast categorical palette
    num_classes = len(ALL_CLASSES)
    palette = sns.color_palette("husl", num_classes)

    # Scatter plot
    sns.scatterplot(
        data=df,
        x="t-SNE Dimension 1",
        y="t-SNE Dimension 2",
        hue="Style Class",
        style="Category",
        palette=palette,
        alpha=0.85,
        s=45,
        edgecolor='none',
        ax=ax
    )

    ax.set_title('ANTIGRAVITY Style Embedding Projection (t-SNE)', color='white', fontsize=14, pad=15)
    ax.set_xlabel('t-SNE Dimension 1', color='#aaaaaa', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', color='#aaaaaa', fontsize=10)
    ax.tick_params(colors='#888888')

    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    ax.grid(color='#22223c', linestyle='--', linewidth=0.5)

    # Configure legend - place outside to prevent clutter
    legend = ax.legend(
        title="Art Styles & Categories",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        facecolor='#181828',
        edgecolor='#333355',
        labelcolor='white',
        title_fontsize=10,
        fontsize=9
    )
    legend.get_title().set_color('#aaaaaa')

    vis_out = os.path.join(Path(FEATURES_DIR).parent, "visualizations", "tsne_clustering.png")
    os.makedirs(os.path.dirname(vis_out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(vis_out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved t-SNE scatter plot → {vis_out}")


if __name__ == "__main__":
    run_tsne_analysis()

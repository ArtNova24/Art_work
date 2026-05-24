# ANTIGRAVITY — Art Reconstruction System
## Phase 1: Data Pipeline & Hybrid Feature Extraction

Welcome to the **ANTIGRAVITY** codebase. This repository contains the first stage of a two-stage AI research system designed to analyze and reconstruct damaged paintings using a combination of handcrafted visual descriptors, deep representation learning, and style-conditioned self-supervised models.

This document details **Phase 1**, which has been fully completed and verified. Phase 1 builds the entire data pipeline, preprocesses art images, extracts a premium **989-dimensional hybrid style descriptor**, and splits the dataset for classifier development (Phase 2).

---

## 🌌 Core Concept: Why Hybrid Features?
Standard computer vision models rely entirely on deep deep-layer features (e.g. ResNet-50 deep layers) which capture *semantic objects* (e.g., "dog", "person", "table"). However, in artistic classification and restoration, semantic objects are less important than the **artistic style** — defined by:
1. **Texture/Brushwork** (captured by GLCM and LBP descriptors)
2. **Color Palette & Light Contrast** (captured by HSV/LAB histograms and moments)
3. **Implicit Structure & Composition** (captured by multi-layer frozen CNN and Vision Transformer features)

By concatenating these multi-modal descriptors into a single **989-dimensional hybrid style vector**, we build a highly discriminative "fingerprint" that uniquely identifies an artist's style or movement. This vector serves as the conditioning signal for the reconstruction network in later phases.

---

## 🛠️ Pipeline Architecture

The entire Phase 1 pipeline is modularized under the `phase1/` directory. Below is the block flow diagram of how an image is processed:

```
[ Raw Image ]
      │
      ▼ (preprocessing.py)
┌────────────────────────────────────────────────────────┐
│ Resize to 224x224 (Bicubic)                            │
│ Generate: RGB Tensor, Grayscale, HSV, LAB              │
└────────────────────────────────────────────────────────┘
      │
      ├───────────────────────┬────────────────────────┬────────────────────────┐
      ▼ (extract_glcm.py)     ▼ (extract_lbp.py)       ▼ (extract_color.py)     ▼ (extract_cnn.py)
┌───────────┐           ┌───────────┐            ┌───────────┐            ┌───────────┐
│ GLCM      │           │ LBP       │            │ Color     │            │ CNN       │
│ Texture   │           │ Gradients │            │ Histogram │            │ DINOv2    │
│ 4 angles  │           │ Full image│            │ 64-bin    │            │ ResNet-50 │
│ 3 dists   │           │ + 4 quads │            │ Moments   │            │ PCA fit   │
└───────────┘           └───────────┘            └───────────┘            └───────────┘
      │                       │                        │                        │
  (20 dims)               (256 dims)               (201 dims)               (512 dims)
      │                       │                        │                        │
      └───────────────────────┴───────────────┬────────┴────────────────────────┘
                                              ▼ (assemble_features.py)
                                    ┌──────────────────┐
                                    │ Concatenate      │ ──► Hybrid Vector (989 dims)
                                    └──────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │ Stratified Split │ ──► Train (70%) / Val (15%) / Test (15%)
                                    │ Class Weights    │ ──► Save all outputs to features/
                                    └──────────────────┘
```

---

## 📁 Codebase Directory Structure

```
Image restoration using JEPA/
├── venv/                       # Python virtual environment
├── data/
│   └── wikiart/                # WikiArt-10 balanced images (~5,000 files)
├── archive (5) - Copy/          # Local raw Indian Art-8 images (~950 files)
├── features/                   # All Phase 1 computed arrays & metadata
│   ├── features_train.npy      # (4137, 989) training features
│   ├── features_val.npy        # (887, 989) validation features
│   ├── features_test.npy       # (887, 989) testing features
│   ├── labels_train.npy        # (4137,) training labels
│   ├── labels_val.npy          # (887,) validation labels
│   ├── labels_test.npy         # (887,) testing labels
│   ├── class_weights.npy       # (18,) class balancing weights
│   ├── pca_model.pkl           # Saved PCA transform for CNN features
│   ├── cnn_scaler.pkl          # Saved Standard Scaler for CNN features
│   ├── style_mapping.json      # Mapping of indices (0-17) to style names
│   ├── image_index.json        # Unified image indexing file
│   └── feature_summary.csv     # Descriptive stats of dimensions per class
├── visualizations/             # Auto-generated visual diagnostics
│   ├── <style>_glcm.png        # GLCM gray-level matrix heatmap
│   ├── <style>_lbp.png         # 256-bin LBP histogram
│   └── <style>_palette.png     # Image + Top-5 KMeans dominant color palette
├── phase1/                     # Python Source Modules
│   ├── config.py               # Hyperparameters, directories, mappings
│   ├── preprocessing.py        # Loading, resizing, color-space conversion
│   ├── extract_glcm.py         # GLCM extraction logic
│   ├── extract_lbp.py          # LBP extraction logic
│   ├── extract_color.py        # HSV/LAB/RGB histograms + color moments
│   ├── extract_cnn.py          # Multi-layer DINOv2 & ResNet-50 + PCA
│   ├── assemble_features.py    # Merging, splitting, and saving arrays
│   ├── visualize.py            # Diagnostic plot generator
│   └── run_phase1.py           # Master orchestration script
├── requirements.txt            # Main requirements file
└── phase1_report.txt           # Detailed pipeline summary report
```

---

## ⚙️ How the Feature Extractors Work

### 1. Grayscale Texture (GLCM — 20 dimensions)
We compute the Gray-Level Co-occurrence Matrix (GLCM) using `scikit-image`.
* **Angles**: $0^\circ, 45^\circ, 90^\circ, 135^\circ$
* **Offsets**: $1, 3, 5$ pixels
* **Properties**: We extract five statistical properties—*contrast, energy, homogeneity, correlation, and dissimilarity*—averaged across offsets to represent directional texture frequencies.

### 2. Micro-Texture Patterns (LBP — 256 dimensions)
We apply **Local Binary Patterns (LBP)** (`default` method) with parameters $P=8$ (points) and $R=1$ (radius) to capture micro-level paint strokes and edge directions.
* To retain rough spatial organization, we compute histograms over the **entire image** and over **four spatial quadrants**.
* The normalized histograms are concatenated into a **256-dimensional** vector.

### 3. Color Signature (201 dimensions)
* **Histograms (192 dims)**: Extracts 64-bin color distribution histograms across HSV, LAB, and RGB color spaces.
* **Moments (9 dims)**: Computes the first three statistical moments (*mean, standard deviation, and skewness*) for each channel in the HSV color space.
* **Dominant Palette (Visualization)**: Applies **KMeans (K=5)** clustering on the RGB pixels of sample images to render swatches of the top 5 dominant colors.

### 4. Deep Visual Representations (CNN — 512 dimensions)
Instead of relying on single-layer deep classifiers, we extract multi-scale representations using:
* **DINOv2 ViT-B/14** (self-supervised transformer)
* **ResNet-50** (supervised CNN)
To avoid out-of-memory issues and capture correct structural shapes:
* DINOv2 is dynamically resized to **518x518** (required for ViT-B patch layout)
* Intermediate activations are extracted from **early, mid, and deep** layers of both networks to capture raw edge details as well as high-level abstract features.
* The raw features are concatenated and reduced via **Principal Component Analysis (PCA)** to a compact **512-dimensional** vector, saving the fitted model (`pca_model.pkl`) to ensure deterministic inference later.

---

## 📈 Dataset Statistics & Classes

The pipeline runs on **5,911 total images** spanning **18 distinct classes** (10 Western Art movements and 8 Indian Cultural Heritage styles).

| Index | Class Code | Display Name | Category | Samples |
| :--- | :--- | :--- | :---: | :---: |
| 0 | `impressionism` | Impressionism | WikiArt-10 | 500 |
| 1 | `cubism` | Cubism | WikiArt-10 | 500 |
| 2 | `baroque` | Baroque | WikiArt-10 | 500 |
| 3 | `abstract_expressionism` | Abstract Expressionism | WikiArt-10 | 500 |
| 4 | `surrealism` | Surrealism | WikiArt-10 | 500 |
| 5 | `renaissance` | Renaissance | WikiArt-10 | 500 |
| 6 | `romanticism` | Romanticism | WikiArt-10 | 500 |
| 7 | `art_nouveau` | Art Nouveau | WikiArt-10 | 500 |
| 8 | `minimalism` | Minimalism | WikiArt-10 | 460 |
| 9 | `pop_art` | Pop Art | WikiArt-10 | 500 |
| 10 | `gond` | Gond Painting | Indian Art | 98 |
| 11 | `kalighat` | Kalighat Painting | Indian Art | 244 |
| 12 | `kangra` | Kangra Painting | Indian Art | 87 |
| 13 | `kerala_mural` | Kerala Mural | Indian Art | 94 |
| 14 | `madhubani` | Madhubani Painting | Indian Art | 95 |
| 15 | `mandana` | Mandana Art | Indian Art | 140 |
| 16 | `pichwai` | Pichwai Painting | Indian Art | 96 |
| 17 | `warli` | Warli Painting | Indian Art | 97 |

---

## 🚀 Execution & Reproducibility

### 1. Requirements Installation
To establish the exact Python virtual environment, execute:
```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Run the Full Orchestration
To execute the pipeline from start to finish, run:
```powershell
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe phase1\run_phase1.py
```

### 3. Pipeline Smart Caching
To protect computer resources and enable quick iterative testing, each step features automatic **numpy caching**. If the feature file exists in the `features/` directory and matches the image index size, the step is bypassed. To trigger a full re-computation, simply delete the target `.npy` file.

---

## 📈 Verification Metrics

At the end of execution, the pipeline runs automated verification assertions:
* **Deliverable Presence**: Asserts that all **12 outputs** (train/val/test splits, labels, scalars, mappings, summaries, indices) are present and non-empty.
* **Shape Verification**: Asserts that feature matrices match `(N, 989)`.
* **Total Dimension Split**: Verified exactly:
  $$\text{Texture }(276) + \text{Color }(201) + \text{CNN }(512) = 989\text{ dimensions}$$
* **Class Balancing**: Automatically computes and saves `class_weights.npy` to penalize minority art style errors proportionately during model training.

The Phase 1 pipeline has successfully concluded with an end-to-end **`SUCCESS`** status, prepped and ready for Phase 2!

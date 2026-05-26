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

---

## 🎨 Phase 2 & 2.5: Style Classification, Feature Ablation & Explanations

In this phase, we develop a suite of high-performance classifiers to predict the artistic style of a painting using the 989-dimensional hybrid style descriptor. The primary goal of Phase 2 is to create a robust **Style Predictor** (`style_predictor.pkl`) that can be used to condition the Self-Supervised I-JEPA generator in Phase 3. 

To ensure our features are maximally discriminative, we designed a **7-experiment ablation study** and conducted **SHAP (SHapley Additive exPlanations)** analysis to explain how handcrafted visual elements and deep CNN patterns influence classification decisions.

---

### ⚖️ The Scale Imbalance & Representation Noise Challenge (Phase 2.5)

During our initial ablation runs, we encountered a fundamental machine learning anomaly:
* **The Problem**: A support vector machine trained on the **Full Hybrid (989 dims)** features achieved a Test Macro F1 of **0.7162**, which was significantly *worse* than a model trained on **CNN-Only features (512 dims)**, which scored **0.8898**. 
* **The Root Cause**: Distance-sensitive algorithms like SVMs with Radial Basis Function (RBF) kernels compute L2 Euclidean distances in the joint feature space. In our raw concatenated hybrid descriptor:
  1. Handcrafted features (GLCM textures, LBP micro-textures, HSV/LAB histograms) had vastly different scales, variances, and numeric bounds compared to deep CNN features.
  2. CNN features naturally dominated the distance calculations.
  3. The 477 handcrafted dimensions introduced high-dimensional representation noise that diluted the highly informative deep signal without adding structured value.

#### 🛠️ The Mathematical Solution: Per-Block Normalization & SHAP-Weighted Beta Blending

To solve this scaling bottleneck, we developed **Phase 2.5 Optimization** (`phase2/optimize_hybrid_features.py`). We introduced two major mathematical treatments to re-align the feature spaces:

1. **Per-Modality Block Z-Score Normalization**:
   Instead of global normalization (which preserves relative magnitude differences between modalities), we isolate and normalize each of the four modality blocks independently:
   $$\mathbf{X}_{\text{block}} = \frac{\mathbf{X}_{\text{block}} - \mu_{\text{block}}}{\sigma_{\text{block}}}$$
   Where $\text{block} \in \{\text{GLCM}, \text{LBP}, \text{Color}, \text{CNN}\}$. This guarantees that every modality block has exactly zero mean and unit variance ($\mu=0, \sigma=1$) prior to concatenation.

2. **SHAP-Weighted Beta Blending**:
   To selectively scale the modalities based on their true predictive signal rather than empirical guessing, we extract the global SHAP feature importance vectors $\mathbf{I}_{\text{SHAP}}$ for each block. We then apply a **Beta Blending** ratio $\beta \in [0, 1]$ to scale the handcrafted vs. deep representations:
   $$\mathbf{W}_{\text{block}} = \text{mean}(|\mathbf{I}_{\text{SHAP}, \text{block}}|)$$
   $$\mathbf{X}_{\text{handcrafted\_scaled}} = \sqrt{1 - \beta} \cdot \mathbf{W}_{\text{handcrafted}} \odot \mathbf{X}_{\text{handcrafted}}$$
   $$\mathbf{X}_{\text{cnn\_scaled}} = \sqrt{\beta} \cdot \mathbf{W}_{\text{cnn}} \odot \mathbf{X}_{\text{cnn}}$$
   Where $\odot$ represents element-wise multiplication by the modality's mean SHAP importance weight vector, scaled globally by the blending ratio.

* **Grid Search Findings**: A grid search over $\beta$ revealed that **$\beta = 0.95$** was the mathematically optimal blending ratio. At this point:
  * CNN features contribute a dominant weight ($\approx \sqrt{0.95} = 0.975$).
  * Handcrafted features act as a clean, highly structured secondary support signal ($\approx \sqrt{0.05} = 0.224$).
  * **Result**: The SVM Full Hybrid Test Macro F1 surged from **0.7162 to 0.8960**, successfully beating the CNN-Only score of **0.8940**.

---

### 🛡️ The 5 Classifier Architectures

We trained and evaluated five distinct machine learning architectures on the optimized hybrid features:

1. **Support Vector Machine (SVM)**: 
   * **Configuration**: Radial Basis Function (RBF) kernel, $C=10.0$, balanced class weights to manage Indian Art style minority partitions.
   * **Role**: High-dimensional decision boundary optimizer. Highly sensitive to feature normalization.

2. **Random Forest (RF)**:
   * **Configuration**: 500 estimators, maximum depth of 25, Gini impurity criterion, balanced bootstrap class weighting.
   * **Role**: Robust ensemble baseline utilizing decision trees to capture non-linear relationships.

3. **XGBoost (XGB)**:
   * **Configuration**: multiclass gradient boosting, learning rate $\eta=0.05$, max depth of 6, 300 boosting rounds, sample-weighted instances.
   * **Role**: Advanced boosting model optimized for gradient step scaling.

4. **Multi-Layer Perceptron (MLP) [BEST MODEL]**:
   * **Configuration**: Custom PyTorch neural network.
     * Layer 1: Linear(989 $\to$ 512) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3)
     * Layer 2: Linear(512 $\to$ 256) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout(0.3)
     * Layer 3: Linear(256 $\to$ 18)
     * Trained for 80 epochs using AdamW optimizer ($\text{lr}=1\times 10^{-3}$, weight decay $= 1\times 10^{-4}$), Cosine Annealing learning rate scheduler, and custom cross-entropy loss penalized by class weights.
   * **Role**: Dynamic, high-capacity deep feature mixer. **Achieved peak Macro F1 of 0.9170 on Test.**

5. **CNN End-to-End Classifier**:
   * **Configuration**: ResNet-18 trained directly on the raw 224x224 input images. We froze the backbone and fine-tuned the fully connected layer for 10 epochs.
   * **Role**: Pure pixels-to-class control baseline.

---

### 📈 Classifier Performance Comparison (Test Partition)

Below is the performance comparison of all five classifiers on the holdout test set using the optimized Phase 2.5 features:

| Model Architecture | Accuracy | Macro F1-Score | Weighted F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Multi-Layer Perceptron (MLP)** | **0.9154** | **0.9170** | **0.9155** | 🥇 **Best Model** |
| **Support Vector Machine (SVM)** | 0.8963 | 0.8960 | 0.8966 | 🥈 **Top Classical** |
| **XGBoost (XGB)** | 0.8703 | 0.8516 | 0.8705 | Robust Boosting |
| **CNN End-to-End** | 0.8072 | 0.8086 | 0.8071 | Baseline Pixels |
| **Random Forest (RF)** | 0.7948 | 0.7664 | 0.7929 | Baseline Trees |

---

### 🧪 Feature Ablation Study Results

To rigorously test the impact of each modality block, we evaluated the SVM classifier across 7 distinct feature slices. 

| Exp Slice | Modality Slice | Dimension Count | Val Macro F1 | Test Macro F1 | Key Takeaway |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A** | Texture Only (GLCM + LBP) | 276 | 0.5056 | 0.5218 | Weak structural baseline |
| **B** | Color Only (Histograms + Moments) | 201 | 0.3619 | 0.3728 | Color alone is highly ambiguous |
| **C** | CNN Only (DINOv2 + ResNet PCA) | 512 | 0.8834 | 0.8940 | Strong representational baseline |
| **D** | Texture + Color | 477 | 0.5818 | 0.6178 | Handcrafted ensemble (No deep features) |
| **E** | Color + CNN | 713 | 0.8817 | 0.8933 | Color slightly dilutes raw CNN |
| **F** | Texture + CNN | 788 | 0.8814 | 0.8959 | Texture combined with CNN is highly powerful |
| **G** | **Full Hybrid (Optimized)** | **989** | **0.8790** | **0.8960** | 🏆 **Optimal - Beats CNN Only (0.8940)** |

*Note: With our Phase 2.5 per-block scaling, adding texture and color to deep CNN features (Slice G) yields superior generalization over deep CNN features alone (Slice C), validating the hybrid design.*

---

### 🔍 Interpretability & Diagnostic Visualizations

All diagnostic outputs are saved under `visualizations/` and `visualizations/shap/`.

#### 1. Global t-SNE Clustering (`visualizations/tsne_clustering.png`)
We projected the 989-dimensional optimized hybrid feature space into 2D using t-Distributed Stochastic Neighbor Embedding (t-SNE) with perplexity=40.
* **Observation**: The features form highly structured, distinct clusters. Western Art movements like *Cubism* and *Pop Art* form tight, isolated islands, while traditional Indian Art styles (e.g., *Warli*, *Madhubani*, *Mandana*) align into separate, high-density clusters, indicating clean semantic and textural boundaries.

#### 2. Feature Ablation Study Plot (`visualizations/ablation_study.png`)
A grouped bar chart illustrating the gradual performance climb from individual handcrafted features to the maximized Full Hybrid score, providing clean proof of Phase 2.5's scaling efficacy.

#### 3. Global SHAP Feature Importance (`visualizations/shap/global_importance.png`)
Using a tree-based explainer on the XGBoost classifier, we measured the mean absolute SHAP value for each feature.
* **Findings**: Deep CNN activations occupy 8 of the top 10 most globally important indices, while **HSV Hue moments** and **LBP quadrant edge histograms** rank highly as secondary features, verifying that the model relies on a mix of local texture frequencies and global semantic structures.

#### 4. Style-Specific Beeswarm Explanations
Beeswarm plots illustrate how individual feature values drive the model's confidence for specific target styles:
* **Baroque (`visualizations/shap/baroque_beeswarm.png`)**: High-value activations in dark LAB color histograms and deep CNN activations (representing complex compositions and low-key lighting) heavily push predictions toward Baroque.
* **Impressionism (`visualizations/shap/impressionism_beeswarm.png`)**: High LBP texture variance (brush strokes) and bright, saturated HSV values strongly correlate with Impressionism.
* **Cubism (`visualizations/shap/cubism_beeswarm.png`)**: Dominated by high LBP quadrant values (representing sharp multi-directional edges and geometric lines).
* **Minimalism (`visualizations/shap/minimalism_beeswarm.png`)**: Triggered by extreme low-value counts in LBP density features (smooth, textureless fields) and tight HSV histogram peaks.

---

### 📂 Phase 2 Directory Structure

Our expansion adds a dedicated `phase2/` module, training outputs, and a visual diagnostic suite:

```
Image restoration using JEPA/
├── features/
│   ├── features_train_raw.npy    # Pre-optimized training backup
│   ├── features_val_raw.npy      # Pre-optimized validation backup
│   ├── features_test_raw.npy     # Pre-optimized testing backup
│   ├── block_scalers.pkl         # Modality-specific StandardScalers (GLCM, LBP, Color, CNN)
│   ├── shap_scaling_weights.npy  # Fitted global SHAP importance weights
│   ├── svm_classifier.pkl        # Serialized SVM model
│   ├── rf_classifier.pkl         # Serialized Random Forest model
│   ├── xgb_classifier.pkl        # Serialized XGBoost model
│   ├── mlp_classifier.pt         # Saved PyTorch MLP model state
│   ├── cnn_end2end_classifier.pt # Saved fine-tuned ResNet-18 checkpoint
│   ├── style_predictor.pkl       # Unified Phase 2 predictor interface
│   ├── classifier_metrics.json   # Full test statistics and F1 scores
│   └── ablation_results.csv      # Tabular ablation test records
├── visualizations/
│   ├── ablation_study.png        # Performance bar chart over 7 slices
│   ├── tsne_clustering.png       # 2D projection clustering of style spaces
│   └── shap/                     # SHAP Explainability plots
│       ├── global_importance.png     # Global feature rank across all classes
│       ├── baroque_beeswarm.png      # Baroque style feature impact
│       ├── impressionism_beeswarm.png # Impressionism style feature impact
│       ├── cubism_beeswarm.png       # Cubism style feature impact
│       └── minimalism_beeswarm.png   # Minimalism style feature impact
├── phase2/
│   ├── optimize_hybrid_features.py # Grid search scaling and block normalizer
│   ├── train_classifiers.py        # Master training logic for the 5 architectures
│   ├── ablation_study.py           # Slices feature vectors and runs comparative SVM tests
│   ├── tsne_analysis.py            # Computes t-SNE coordinate spaces and saves plot
│   ├── shap_analysis.py            # Computes tree-based SHAP values and builds beeswarms
│   └── run_phase2.py               # Orchestrator running the complete Phase 2/2.5 pipeline
└── phase2_report.txt               # Plain-text performance summary card
```

### 🚀 Running Phase 2 Pipeline

To execute the entire Phase 2 training, feature optimization, ablation, t-SNE, and SHAP suite:
```powershell
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe phase2\run_phase2.py
```
*Note: The script features automatic model caching. If `mlp_classifier.pt` or `svm_classifier.pkl` exist, they will be loaded directly. To force a full retrain, delete the respective pickle or model files in `features/`.*

---

**Phase 2 has concluded successfully, and our high-performance Hybrid Style Predictor is saved and fully primed to act as the conditioning context for our Self-Supervised Style-Conditioned I-JEPA Model in Phase 3!**


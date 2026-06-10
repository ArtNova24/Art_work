# Graph Report - Image restoration using JEPA  (2026-05-28)

## Corpus Check
- 32 files · ~159,566 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 460 nodes · 629 edges · 35 communities (29 shown, 6 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f6714014`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]

## God Nodes (most connected - your core abstractions)
1. `Phase4Evaluator` - 17 edges
2. `run_inference()` - 15 edges
3. `PerceptualStyleLoss` - 13 edges
4. `run_training()` - 13 edges
5. `FIDCalculator` - 13 edges
6. `ANTIGRAVITY — Reconstruction Quality Improvement Plan` - 13 edges
7. `safe_load()` - 12 edges
8. `StyleJEPAImageDataset` - 12 edges
9. `BlockMaskGenerator` - 12 edges
10. `PixelDecoder` - 12 edges

## Surprising Connections (you probably didn't know these)
- `FIDCalculator` --uses--> `StyleJEPAImageDataset`  [INFERRED]
  phase4/evaluator.py → phase3/dataset.py
- `Phase4Evaluator` --uses--> `StyleJEPAImageDataset`  [INFERRED]
  phase4/evaluator.py → phase3/dataset.py
- `FIDCalculator` --uses--> `BlockMaskGenerator`  [INFERRED]
  phase4/evaluator.py → phase3/masking.py
- `Phase4Evaluator` --uses--> `BlockMaskGenerator`  [INFERRED]
  phase4/evaluator.py → phase3/masking.py
- `FIDCalculator` --uses--> `StyleProjector`  [INFERRED]
  phase4/evaluator.py → phase3/models.py

## Communities (35 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (46): 3. Approach B — FiLM Style Conditioning, 4. Approach C — Convolutional Pixel Decoder, 5. Approach D — Perceptual + Style Loss, 6. Approach E — Diffusion Decoder (Advanced), 7. Approach F — Scale the Dataset, 8. Implementation Roadmap, 9. Expected Results After Each Fix, ANTIGRAVITY — Reconstruction Quality Improvement Plan (+38 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (44): 1. Global t-SNE Clustering (`visualizations/tsne_clustering.png`), 1. Grayscale Texture (GLCM — 20 dimensions), 1. Requirements Installation, 2. Feature Ablation Study Plot (`visualizations/ablation_study.png`), 2. Micro-Texture Patterns (LBP — 256 dimensions), 2. Run the Full Orchestration, 3. Color Signature (201 dimensions), 3. Global SHAP Feature Importance (`visualizations/shap/global_importance.png`) (+36 more)

### Community 2 - "Community 2"
Cohesion: 0.10
Nodes (29): extract_patches(), Folds a batch of patches of shape (B, 196, 3, 16, 16)     back into a batch of, Slices a batch of images of shape (B, 3, 224, 224)     into non-overlapping pat, reconstruct_image(), _compute_metrics(), detect_corrupted_patches(), _extract_dino(), extract_hybrid() (+21 more)

### Community 3 - "Community 3"
Cohesion: 0.11
Nodes (26): assemble_hybrid(), compute_weights(), load_all_features(), ndarray, Historic Image Restoration Phase 1 — Step 7: Feature Assembly & Splitting Conca, Save feature_summary.csv: mean and std of each dimension per style class.     S, Save all npy files to features/ directory., Load all intermediate feature arrays and the image index. (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (15): Dataset, ArtImageDataset, evaluate_model(), HybridFeatureDataset, load_data(), main(), Historic Image Restoration Phase 2 — Step 1: Model Training Trains 5 style clas, Compute comprehensive accuracy, macro F1, and weighted F1 metrics. (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (22): 1. Extract Features (Phase 1), 2. Train Classifiers & Run Ablations (Phase 2), 3. Train Style-Conditioned I-JEPA (Phase 3), 4. Run Model Evaluation Suite (Phase 4), 5. Launch Live Gradio Web Server (Phase 4), code:mermaid (flowchart TD), code:powershell ($env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe pha), code:powershell ($env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe -u ) (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (12): ConditionalResBlock, DDPM, DiffusionPatchDecoder, Historic Image Restoration Phase 3 — Conditional Diffusion Decoder. Implements D, DDPM noise scheduler for patch-level diffusion., Add noise to clean patches x0 at timestep t., Standard sinusoidal time embedding for diffusion timestep., DDIM-style fast sampling from the diffusion decoder. (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (19): extract_glcm(), extract_glcm_all(), ndarray, Historic Image Restoration Phase 1 — Step 3: GLCM Feature Extraction Extracts 2, Extract 20-dim GLCM feature vector from a grayscale image.      Args:, Extract GLCM features for all images in the index.      Args:         image_i, extract_lbp(), extract_lbp_all() (+11 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (19): 1. Baseline Classifiers, 1. Quantitative Evaluator, 1. Texture Features (GLCM), 1. The Masking Pipeline, 2. Architecture & Components ([`phase3/models.py`](file:///c:/Users/SUBHAM/Downloads/Image%20restoration%20using%20JEPA/phase3/models.py)), 2. Local Structures (LBP), 2. The 5-Mode Gradio Web Application ([`phase4/gradio_app.py`](file:///c:/Users/SUBHAM/Downloads/Image%20restoration%20using%20JEPA/phase4/gradio_app.py)), 2. Visualizations & Explainability (+11 more)

### Community 9 - "Community 9"
Cohesion: 0.16
Nodes (7): Attention, FiLMLayer, FiLMTransformerBlock, Historic Image Restoration Phase 3 — Neural Network Models. Implements:   1. S, Feature-wise Linear Modulation.     Given a style vector s of shape (B, style_d, x     : (B, N, feat_dim)         style : (B, style_dim), TransformerBlock

### Community 10 - "Community 10"
Cohesion: 0.12
Nodes (9): FIDCalculator, Runs batch reconstruction using the style-conditioned predictor & decoder., Runs OpenCV Fast Inpainting (Telea) as classical baseline.         imgs: torch., Runs the complete Phase 4 evaluation pipeline., Generates a detailed, beautiful ASCII evaluation report., Computes Fréchet Inception Distance using PyTorch's InceptionV3 model., Extract Inception V3 features.         images_tensor: shape (B, 3, 224, 224) in, Computes mean and covariance of extracted features. (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.19
Nodes (14): bool, apply_pca(), extract_dino_features(), extract_raw_cnn_features(), extract_resnet_features(), load_models(), ndarray, Historic Image Restoration Phase 1 — Step 6: CNN Feature Extraction Extracts 51 (+6 more)

### Community 12 - "Community 12"
Cohesion: 0.23
Nodes (13): generate_all_visualisations(), pick_sample_per_class(), plot_color_palette(), plot_glcm(), plot_lbp_histogram(), ndarray, str, Historic Image Restoration Phase 1 — Step 8: Feature Visualisation Generates sa (+5 more)

### Community 13 - "Community 13"
Cohesion: 0.20
Nodes (9): PixelDecoder, Convolutional pixel decoder.     Input : (B, N_tgt, embed_dim) latent predictio, z_tgt_pred: (B, N_tgt, embed_dim)         Returns   : (B, N_tgt, 3, 16, 16), main(), Historic Image Restoration Phase 3 — Style-Conditioned I-JEPA Reconstruction Orc, # NOTE: Do NOT wrap sys.stdout with TextIOWrapper here — it re-buffers, Historic Image Restoration Phase 3 — Style-Conditioned I-JEPA Training Suite. I, Executes the self-supervised style-conditioned I-JEPA training run. (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (11): 1. Current State: What Is Actually Broken, code:python (# CURRENT (BROKEN): Randomly initialized), code:python (# CURRENT (BROKEN): Style is just one token in a 197-token s), code:python (# CURRENT (BROKEN): 2-layer MLP decoding art texture), code:python (# CURRENT (BROKEN): L2 averages all plausible reconstruction), Phase 4 Results (The Problem in Numbers), Root Cause 1 — Training From Scratch on 4,000 Images, Root Cause 2 — Style Conditioning Is Being Ignored (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.20
Nodes (9): build_image_index(), download_wikiart(), generate_style_mapping(), Historic Image Restoration Phase 1 — Step 1: Dataset Download & Preparation (Str, Verify and index the Indian Art-8 dataset from the local folder., Generate style_mapping.json mapping integer index → class name., Build a master list of all (image_path, class_idx) pairs.     Saves as image_in, Download missing WikiArt images from HuggingFace Artificio/WikiArt and save to d (+1 more)

### Community 16 - "Community 16"
Cohesion: 0.29
Nodes (9): _channel_hist(), extract_color(), extract_color_all(), int, ndarray, Historic Image Restoration Phase 1 — Step 5: Color Feature Extraction Extracts, Compute a normalised histogram for one image channel (uint8)., Extract 201-dim colour feature vector.      Args:         img_rgb : (H, W, 3) (+1 more)

### Community 17 - "Community 17"
Cohesion: 0.29
Nodes (5): z_ctx: (B, N_ctx, embed_dim) Context latents from encoder.         s_emb: (B, s, StyleConditionedPredictor, StyleProjector, Phase4Evaluator, Historic Image Restoration Phase 4 — Core Evaluation and Metrics Engine Compute

### Community 18 - "Community 18"
Cohesion: 0.25
Nodes (8): 2. Approach A — Pretrained ViT Backbone (Highest Impact), code:python (import timm), code:python (# Update these constants), code:python (# Differential learning rates for pretrained vs new componen), Exact Code Change, Expected Impact, What It Is, Why It Fixes the Problem

### Community 19 - "Community 19"
Cohesion: 0.29
Nodes (4): BlockMaskGenerator, Historic Image Restoration Phase 3 — Block-Wise Masking Pipeline. Generates con, Generates a block-wise random mask with exactly target_masked patches., Generates a batch of masks.         Returns:             masks: BoolTensor of

### Community 20 - "Community 20"
Cohesion: 0.38
Nodes (6): get_slices(), load_data(), Historic Image Restoration Phase 2 — Step 2: Feature Ablation Study Evaluates c, Load pre-split numpy arrays from features/., Define slicing indices for the 4 feature groups., run_ablation()

### Community 21 - "Community 21"
Cohesion: 0.29
Nodes (3): Historic Image Restoration Phase 3 — Style JEPA Dataset. Loads preprocessed raw, split: 'train', 'val', or 'test', StyleJEPAImageDataset

### Community 22 - "Community 22"
Cohesion: 0.38
Nodes (3): PerceptualStyleLoss, Computes VGG-based perceptual loss + Gram matrix style loss.     Uses VGG-16 fe, pred, target: (B, 3, H, W) in [-1, 1]         Returns: perceptual_loss, style_l

### Community 23 - "Community 23"
Cohesion: 0.48
Nodes (6): build_app(), main(), Historic Image Restoration Phase 4 -- Master Orchestrator Supports two modes vi, run_demo(), run_evaluate(), set_seeds()

### Community 24 - "Community 24"
Cohesion: 0.47
Nodes (5): get_feature_names(), load_data(), Historic Image Restoration Phase 2 — Step 3: SHAP Feature Attribution Computes, Generate meaningful feature names instead of simple indices., run_shap_analysis()

### Community 25 - "Community 25"
Cohesion: 0.33
Nodes (3): Extract DINOv2 intermediate features for style fidelity checks., Extract ResNet-50 intermediate features for style fidelity checks., Extracts 989-dim hybrid feature vector for a reconstructed image.         img_t

### Community 26 - "Community 26"
Cohesion: 0.50
Nodes (4): load_data(), main(), Historic Image Restoration Phase 2.5 — Feature Normalization & Validation-Tuned, Load raw split arrays.

### Community 27 - "Community 27"
Cohesion: 0.67
Nodes (3): compile_final_report(), header(), Historic Image Restoration Phase 2 — Master Orchestration Script Runs all Phase

## Knowledge Gaps
- **99 isolated node(s):** `bool`, `int`, `From Broken Baselines to Style-Faithful Art Restoration`, `Table of Contents`, `Phase 4 Results (The Problem in Numbers)` (+94 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `safe_load()` connect `Community 7` to `Community 2`, `Community 11`, `Community 12`, `Community 16`, `Community 17`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `StyleJEPAImageDataset` connect `Community 21` to `Community 4`, `Community 10`, `Community 13`, `Community 17`, `Community 22`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `Phase4Evaluator` (e.g. with `StyleJEPAImageDataset` and `BlockMaskGenerator`) actually correct?**
  _`Phase4Evaluator` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `PerceptualStyleLoss` (e.g. with `StyleJEPAImageDataset` and `BlockMaskGenerator`) actually correct?**
  _`PerceptualStyleLoss` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `FIDCalculator` (e.g. with `StyleJEPAImageDataset` and `BlockMaskGenerator`) actually correct?**
  _`FIDCalculator` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Historic Image Restoration Phase 1 — Step 7: Feature Assembly & Splitting Conca`, `Load all intermediate feature arrays and the image index.`, `Dimension sanity check — raise if any mismatch.` to the rest of the system?**
  _205 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0425531914893617 - nodes in this community are weakly interconnected._
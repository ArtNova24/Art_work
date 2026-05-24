"""
ANTIGRAVITY Phase 1 — Step 6: CNN Feature Extraction
Extracts 512-dimensional deep CNN features using DINOv2 + ResNet-50.

Output:
  features/cnn_features.npy   shape (N, 512)
  features/pca_model.pkl      fitted PCA model (reused in all phases)

Pipeline:
  1. DINOv2-ViT-B/14 → CLS token (768-dim) + early/mid layer patch averages
  2. ResNet-50        → layer2 (512-dim avg pool) + layer4 (2048-dim avg pool)
  3. Concatenate all  → high-dim raw vector
  4. PCA reduce       → 512-dim final CNN feature
"""

import os
import json
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_DIR, IMG_SIZE, CNN_DIM, CNN_BATCH_SIZE,
    PCA_COMPONENTS, RANDOM_SEED
)
from preprocessing import safe_load

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ImageNet normalisation ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_models():
    """Load frozen DINOv2 and ResNet-50 models."""
    import timm
    import torchvision.models as models

    print(f"  Device: {DEVICE}")

    # ── DINOv2 ViT-B/14 ──────────────────────────────────────────────────────
    print("  Loading DINOv2-ViT-B/14...")
    dino = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
    dino.eval()
    dino = dino.to(DEVICE)
    for p in dino.parameters():
        p.requires_grad_(False)

    # ── ResNet-50 ─────────────────────────────────────────────────────────────
    print("  Loading ResNet-50...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.eval()
    resnet = resnet.to(DEVICE)
    for p in resnet.parameters():
        p.requires_grad_(False)

    return dino, resnet


def extract_resnet_features(resnet, batch: torch.Tensor) -> np.ndarray:
    """
    Extract multi-layer ResNet-50 features.
    Returns concatenation of layer2 + layer4 global avg pools.
    """
    x = batch
    x = resnet.conv1(x)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)

    x = resnet.layer1(x)
    l2 = resnet.layer2(x)   # (B, 512, 28, 28)
    l2_pool = l2.mean(dim=[2, 3])  # (B, 512)

    l3 = resnet.layer3(l2)
    l4 = resnet.layer4(l3)  # (B, 2048, 7, 7)
    l4_pool = l4.mean(dim=[2, 3])  # (B, 2048)

    combined = torch.cat([l2_pool, l4_pool], dim=1)  # (B, 2560)
    return combined.cpu().numpy()


def extract_dino_features(dino, batch: torch.Tensor) -> np.ndarray:
    """
    Extract DINOv2 features: CLS token + intermediate layer patch averages.
    Returns (B, 768*3) = (B, 2304) from 3 selected layers.
    """
    # Register hooks to capture intermediate layer outputs
    intermediate = {}

    def make_hook(name):
        def hook(module, input, output):
            # output shape: (B, num_patches+1, dim) for ViT blocks
            if isinstance(output, torch.Tensor):
                # Average patch tokens (exclude CLS at position 0)
                intermediate[name] = output[:, 1:, :].mean(dim=1)  # (B, 768)
        return hook

    # Hook early (block 3), mid (block 7), deep (block 11) layers
    n_blocks = len(dino.blocks)
    early_idx = n_blocks // 4
    mid_idx   = n_blocks // 2
    deep_idx  = n_blocks - 1

    handles = [
        dino.blocks[early_idx].register_forward_hook(make_hook('early')),
        dino.blocks[mid_idx  ].register_forward_hook(make_hook('mid')),
        dino.blocks[deep_idx ].register_forward_hook(make_hook('deep')),
    ]

    try:
        # DINOv2 ViT-B/14 (lvd142m) expects 518×518 (37×37 patches × 14px = 518)
        dino_input_size = dino.patch_embed.img_size
        if isinstance(dino_input_size, (list, tuple)):
            dino_h, dino_w = dino_input_size[0], dino_input_size[1]
        else:
            dino_h = dino_w = dino_input_size
        batch_resized = torch.nn.functional.interpolate(
            batch, size=(dino_h, dino_w), mode='bilinear', align_corners=False
        )
        out = dino.forward_features(batch_resized)
        # CLS token from final layer
        if isinstance(out, dict):
            cls_token = out['x_norm_clstoken']  # (B, 768)
        else:
            cls_token = out[:, 0, :]  # (B, 768)
    finally:
        for h in handles:
            h.remove()

    # Combine: early(768) + mid(768) + cls(768) = 2304
    combined = torch.cat([
        intermediate.get('early', torch.zeros(batch.shape[0], 768, device=DEVICE)),
        intermediate.get('mid',   torch.zeros(batch.shape[0], 768, device=DEVICE)),
        cls_token,
    ], dim=1)  # (B, 2304)

    return combined.cpu().numpy()


def extract_raw_cnn_features(image_index: list, dino, resnet) -> np.ndarray:
    """
    Extract raw (pre-PCA) CNN features for all images.
    Returns (N, 4864) = DINOv2(2304) + ResNet(2560)
    """
    all_raw = []
    failed_count = 0

    # Process in batches
    batch_paths = [item['path'] for item in image_index]
    n = len(batch_paths)

    for start in tqdm(range(0, n, CNN_BATCH_SIZE), desc="  Extracting CNN features ", unit="batch"):
        batch_paths_chunk = batch_paths[start: start + CNN_BATCH_SIZE]
        tensors = []
        valid_indices = []

        for i, path in enumerate(batch_paths_chunk):
            result = safe_load(path, IMG_SIZE)
            if result is None:
                failed_count += 1
                continue
            img_rgb_norm = result[4]  # float32 (H,W,3)
            tensor = TRANSFORM(img_rgb_norm)  # apply ImageNet norm
            tensors.append(tensor)
            valid_indices.append(i)

        if not tensors:
            # All failed — push zeros for the whole batch
            for _ in batch_paths_chunk:
                all_raw.append(np.zeros(2304 + 2560, dtype=np.float32))
            continue

        batch_tensor = torch.stack(tensors).to(DEVICE)  # (B, 3, H, W)

        with torch.no_grad():
            dino_feats   = extract_dino_features(dino, batch_tensor)    # (B, 2304)
            resnet_feats = extract_resnet_features(resnet, batch_tensor) # (B, 2560)

        raw = np.concatenate([dino_feats, resnet_feats], axis=1)  # (B, 4864)

        # Insert back (zeros for failed images)
        ptr = 0
        for i in range(len(batch_paths_chunk)):
            if i in valid_indices:
                all_raw.append(raw[ptr])
                ptr += 1
            else:
                all_raw.append(np.zeros(2304 + 2560, dtype=np.float32))

    if failed_count:
        print(f"\n  WARNING: {failed_count} images failed during CNN extraction.")

    return np.array(all_raw, dtype=np.float32)


def apply_pca(raw_features: np.ndarray, fit: bool = True):
    """
    Fit (or load) PCA and reduce raw CNN features to 512-dim.

    Returns:
        reduced   : (N, 512) float32
        pca_model : fitted sklearn PCA object
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca_path    = os.path.join(FEATURES_DIR, "pca_model.pkl")
    scaler_path = os.path.join(FEATURES_DIR, "cnn_scaler.pkl")

    if fit:
        print(f"\n  Fitting StandardScaler + PCA({PCA_COMPONENTS}) on {raw_features.shape}...")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw_features)

        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
        reduced = pca.fit_transform(scaled).astype(np.float32)

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA explained variance: {explained:.1f}%")

        joblib.dump(pca, pca_path)
        joblib.dump(scaler, scaler_path)
        print(f"  Saved pca_model.pkl and cnn_scaler.pkl → {FEATURES_DIR}")
    else:
        print("  Loading existing PCA model...")
        scaler = joblib.load(scaler_path)
        pca    = joblib.load(pca_path)
        scaled  = scaler.transform(raw_features)
        reduced = pca.transform(scaled).astype(np.float32)

    return reduced, pca


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ANTIGRAVITY Phase 1 — Step 6: CNN Extraction")
    print("="*60)

    index_path = os.path.join(FEATURES_DIR, "image_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("Run download_dataset.py first to generate image_index.json")

    with open(index_path) as f:
        image_index = json.load(f)

    print(f"  Processing {len(image_index)} images on {DEVICE}...")

    # Load models
    dino, resnet = load_models()

    # Extract raw features
    raw_features = extract_raw_cnn_features(image_index, dino, resnet)
    raw_path = os.path.join(FEATURES_DIR, "cnn_raw_features.npy")
    np.save(raw_path, raw_features)
    print(f"\n  Raw CNN features: {raw_features.shape} → {raw_path}")

    # PCA reduction to 512-dim
    cnn_features, _ = apply_pca(raw_features, fit=True)

    out_path = os.path.join(FEATURES_DIR, "cnn_features.npy")
    np.save(out_path, cnn_features)

    print(f"\n  ✓ CNN features saved: {cnn_features.shape} → {out_path}")
    print(f"  Min: {cnn_features.min():.4f}  Max: {cnn_features.max():.4f}")

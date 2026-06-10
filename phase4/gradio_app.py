"""
Historic Image Restoration Phase 4 -- Interactive Gradio Web Application
End-to-end demo: Upload damaged painting -> Style detected -> Style-Conditioned I-JEPA reconstructs.
Features:
  - Premium dark-mode glassmorphic theme.
  - Side-by-side visual comparison: Original | Corrupted | Conditioned JEPA | Vanilla JEPA | OpenCV.
  - Style classification confidence bar chart.
  - Real-time SSIM, PSNR metrics, and style fidelity check.
  - Pre-loaded gallery examples from WikiArt-10 and Indian art forms.
All prints and comments kept strictly in ASCII.
"""
import os
import sys
import cv2
import math
import joblib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn.functional as F
import torchvision.transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase1"))
sys.path.insert(0, os.path.join(ROOT, "phase3"))

from phase4.config import FEATURES_DIR, IMG_SIZE, RANDOM_SEED
from phase1.config import ALL_CLASSES, CLASS_TO_IDX, TOTAL_DIM
from phase1.preprocessing import safe_load
from phase1.extract_glcm import extract_glcm
from phase1.extract_lbp import extract_lbp
from phase1.extract_color import extract_color
from phase1.extract_cnn import TRANSFORM as IMAGENET_TRANSFORM

from phase3.config import DEVICE
from phase3.masking import BlockMaskGenerator
from phase3.models import (
    StyleProjector, ViTContextEncoder, StyleConditionedPredictor, PixelDecoder
)
from phase3.train_jepa import extract_patches, reconstruct_image

import gradio as gr

# ---------------------------------------------------------------------------
# Model Registry -- loaded once at startup
# ---------------------------------------------------------------------------
print("  [Historic Image Restoration] Initializing model registry for Gradio demo...")

_projector = StyleProjector().to(DEVICE)
_projector.load_state_dict(torch.load(os.path.join(FEATURES_DIR, "jepa_style_projector.pt"), map_location=DEVICE))
_projector.eval()

_context_encoder = ViTContextEncoder().to(DEVICE)
_context_encoder.load_state_dict(torch.load(os.path.join(FEATURES_DIR, "jepa_context_encoder.pt"), map_location=DEVICE))
_context_encoder.eval()

_predictor = StyleConditionedPredictor().to(DEVICE)
_predictor.load_state_dict(torch.load(os.path.join(FEATURES_DIR, "jepa_predictor.pt"), map_location=DEVICE))
_predictor.eval()

_pixel_decoder = PixelDecoder().to(DEVICE)
_pixel_decoder.load_state_dict(torch.load(os.path.join(FEATURES_DIR, "jepa_pixel_decoder.pt"), map_location=DEVICE))
_pixel_decoder.eval()

_classifier = joblib.load(os.path.join(FEATURES_DIR, "style_predictor.pkl"))
_pca_model  = joblib.load(os.path.join(FEATURES_DIR, "pca_model.pkl"))
_scaler     = joblib.load(os.path.join(FEATURES_DIR, "cnn_scaler.pkl"))

_mask_gen   = BlockMaskGenerator(grid_size=14, target_masked=98)

# DINOv2 and ResNet-50 for feature extraction
from extract_cnn import load_models as _load_cnn_models
_dino, _resnet = _load_cnn_models()

print("  [Historic Image Restoration] Model registry loaded successfully.")

# ---------------------------------------------------------------------------
# Pre-Processing Utilities
# ---------------------------------------------------------------------------
_to_tensor = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def pil_to_tensor(pil_img):
    """PIL RGB -> torch (1, 3, 224, 224) [-1, 1]."""
    pil_img = pil_img.convert("RGB")
    return _to_tensor(pil_img).unsqueeze(0)

def tensor_to_pil(t):
    """torch (3, 224, 224) [-1, 1] -> PIL RGB."""
    arr = (t.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(arr)

def tensor_to_np(t):
    """torch (3, 224, 224) [-1, 1] -> float32 H x W x 3 in [0,1]."""
    arr = t.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    return np.clip(arr, 0.0, 1.0)

# ---------------------------------------------------------------------------
# In-memory 989-dim hybrid feature extraction
# ---------------------------------------------------------------------------
def _extract_dino(batch_tensor):
    intermediate = {}
    def _hook(name):
        def fn(mod, inp, out):
            if isinstance(out, torch.Tensor):
                intermediate[name] = out[:, 1:, :].mean(dim=1)
        return fn
    n = len(_dino.blocks)
    handles = [
        _dino.blocks[n // 4].register_forward_hook(_hook('early')),
        _dino.blocks[n // 2].register_forward_hook(_hook('mid')),
        _dino.blocks[n - 1 ].register_forward_hook(_hook('deep')),
    ]
    try:
        sz = _dino.patch_embed.img_size
        if isinstance(sz, (list, tuple)):
            h, w = sz[0], sz[1]
        else:
            h = w = sz
        resized = F.interpolate(batch_tensor, size=(h, w), mode='bilinear', align_corners=False)
        out = _dino.forward_features(resized)
        cls = out['x_norm_clstoken'] if isinstance(out, dict) else out[:, 0, :]
    finally:
        for h_handle in handles:
            h_handle.remove()
    combined = torch.cat([
        intermediate.get('early', torch.zeros(batch_tensor.shape[0], 768, device=DEVICE)),
        intermediate.get('mid',   torch.zeros(batch_tensor.shape[0], 768, device=DEVICE)),
        cls,
    ], dim=1)
    return combined.cpu().numpy()

def _extract_resnet(batch_tensor):
    x = _resnet.conv1(batch_tensor)
    x = _resnet.bn1(x); x = _resnet.relu(x); x = _resnet.maxpool(x)
    x = _resnet.layer1(x)
    l2 = _resnet.layer2(x).mean(dim=[2, 3])
    l3 = _resnet.layer3(_resnet.layer2(x))
    l4 = _resnet.layer4(l3).mean(dim=[2, 3])
    return torch.cat([l2, l4], dim=1).cpu().numpy()

def extract_hybrid(img_tensor_minus1_to_1):
    """img_tensor: torch (3, 224, 224) in [-1, 1]. Returns (989,) float32."""
    arr = (img_tensor_minus1_to_1.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    bgr  = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    glcm_f  = extract_glcm(gray)
    lbp_f   = extract_lbp(gray)
    color_f = extract_color(arr, hsv, lab)

    img_norm = arr.astype(np.float32) / 255.0
    t = IMAGENET_TRANSFORM(img_norm).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dino_f   = _extract_dino(t)
        resnet_f = _extract_resnet(t)
    cnn_raw = np.concatenate([dino_f, resnet_f], axis=1)
    cnn_scaled = _scaler.transform(cnn_raw)
    cnn_f = _pca_model.transform(cnn_scaled).astype(np.float32)[0]

    return np.concatenate([glcm_f, lbp_f, color_f, cnn_f]).astype(np.float32)

# ---------------------------------------------------------------------------
# Core reconstruction function
# ---------------------------------------------------------------------------
def _reconstruct(img_tensor, style_vec_np, mask_1d, ablation=None):
    """
    img_tensor : (1, 3, 224, 224) on CPU, range [-1, 1]
    style_vec_np: (989,) float32 numpy
    mask_1d     : (196,) bool torch BoolTensor
    ablation    : None | 'zeros' | 'color_only' | 'texture_only'
    Returns (3, 224, 224) float tensor on CPU, range [-1, 1].
    """
    imgs   = img_tensor.to(DEVICE)
    masks  = mask_1d.unsqueeze(0).to(DEVICE)   # (1, 196)
    sv     = style_vec_np.copy()

    if ablation == 'zeros':
        sv[:] = 0.0
    elif ablation == 'color_only':
        sv[:276] = 0.0; sv[477:] = 0.0
    elif ablation == 'texture_only':
        sv[276:] = 0.0

    sv_t = torch.tensor(sv, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 989)

    with torch.no_grad():
        patches = extract_patches(imgs)                            # (1, 196, 3, 16, 16)
        num_tgt = int(mask_1d.sum().item())
        sorted_idx = torch.argsort(masks.to(torch.int32), dim=1)
        tgt_idx    = sorted_idx[:, -num_tgt:] if num_tgt > 0 else sorted_idx[:, :0]

        s_emb      = _projector(sv_t)                              # (1, 256)
        z_ctx, _   = _context_encoder(imgs, mask=masks)            
        z_pred, _  = _predictor(z_ctx, s_emb, mask=masks)         
        pix_pred   = _pixel_decoder(z_pred)                        

        recon_patches = patches.clone()
        if num_tgt > 0:
            recon_patches[0, tgt_idx[0]] = pix_pred[0]
        recon_img = reconstruct_image(recon_patches)               # (1, 3, 224, 224)

    return recon_img[0].cpu()

def _opencv_inpaint(img_tensor, mask_1d):
    """Classical OpenCV Telea inpainting. Returns PIL image."""
    arr = (img_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    mask_2d = mask_1d.numpy().reshape(14, 14)
    mask_2d = np.repeat(np.repeat(mask_2d, 16, axis=0), 16, axis=1).astype(np.uint8) * 255
    inp = cv2.inpaint(arr, mask_2d, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inp)

# ---------------------------------------------------------------------------
# Compute metrics helpers
# ---------------------------------------------------------------------------
def _compute_metrics(orig_np, recon_np):
    from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
    ssim_val = ssim_fn(orig_np, recon_np, channel_axis=2, data_range=1.0)
    psnr_val = psnr_fn(orig_np, recon_np, data_range=1.0)
    if math.isinf(psnr_val) or math.isnan(psnr_val):
        psnr_val = 80.0
    return round(ssim_val, 4), round(psnr_val, 2)

# ---------------------------------------------------------------------------
# Style confidence bar chart
# ---------------------------------------------------------------------------
def _style_chart(probs, pred_idx, title="Style Classification"):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    colors = ["#6c63ff" if i != pred_idx else "#00e5ff" for i in range(len(ALL_CLASSES))]
    y_pos  = np.arange(len(ALL_CLASSES))
    ax.barh(y_pos, probs, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace("_", " ").title() for c in ALL_CLASSES],
                       color="#e0e0e0", fontsize=8)
    ax.set_xlabel("Confidence", color="#888888", fontsize=9)
    ax.set_title(title, color="#ffffff", fontsize=11, pad=10)
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#222233")
    ax.set_xlim(0, 1)
    ax.axvline(0, color="#333344", linewidth=0.5)

    patch = mpatches.Patch(color="#00e5ff", label=f"Predicted: {ALL_CLASSES[pred_idx].replace('_',' ').title()}")
    ax.legend(handles=[patch], loc='lower right', facecolor="#1a1a2e", labelcolor="#ffffff", fontsize=8)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Main inference function called by Gradio
# ---------------------------------------------------------------------------
def detect_corrupted_patches(img_tensor, threshold=0.008):
    """
    img_tensor: (1, 3, 224, 224) on CPU, range [-1, 1]
    Returns a (196,) BoolTensor.
    """
    img_np = (img_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # (224, 224, 3) in [0, 1]
    mask_1d = torch.zeros(196, dtype=torch.bool)
    
    for i in range(14):
        for j in range(14):
            patch = img_np[i*16:(i+1)*16, j*16:(j+1)*16]
            var = np.var(patch)
            
            # Artificially corrupted regions (black/grey/white flat color) have extremely low variance
            if var < threshold:
                mask_1d[i*14 + j] = True
                
    # Fallback: if no flat patches detected, select 49 lowest variance patches (25%)
    if not torch.any(mask_1d):
        variances = []
        for i in range(14):
            for j in range(14):
                patch = img_np[i*16:(i+1)*16, j*16:(j+1)*16]
                variances.append((np.var(patch), i*14 + j))
        variances.sort(key=lambda x: x[0])
        for k in range(49):
            mask_1d[variances[k][1]] = True
            
    return mask_1d

def overlay_mask(pil_img, mask_1d, color=(255, 0, 0, 100)):
    """
    Overlays a semi-transparent color on the masked 16x16 patches.
    pil_img: PIL Image of shape (224, 224)
    mask_1d: (196,) BoolTensor
    Returns: PIL Image with overlay
    """
    from PIL import ImageDraw
    img_rgba = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    for i in range(14):
        for j in range(14):
            if mask_1d[i*14 + j]:
                x0 = j * 16
                y0 = i * 16
                x1 = (j + 1) * 16
                y1 = (i + 1) * 16
                draw.rectangle([x0, y0, x1, y1], fill=color)
                
    return Image.alpha_composite(img_rgba, overlay).convert("RGB")

def resolve_image_path(path_str):
    if not path_str:
        return None
    if os.path.exists(path_str):
        return path_str
    # Try finding it relative to ROOT
    parts = path_str.replace("\\", "/").split("/Image restoration using JEPA/")
    if len(parts) > 1:
        rel_path = parts[1]
        local_path = os.path.join(ROOT, rel_path.replace("/", os.sep))
        if os.path.exists(local_path):
            return local_path
    return None

def extract_pil_image(value):
    """
    Extracts a PIL Image robustly from different Gradio outputs
    (dict, filepath, numpy array, or PIL Image) for clipboard, webcam, and upload support.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return Image.open(value)
        except Exception as e:
            print(f"Error opening image filepath {value}: {e}")
            return None
    if isinstance(value, dict):
        if "path" in value:
            return extract_pil_image(value["path"])
        img = value.get("background")
        if img is None:
            img = value.get("composite")
        if img is None and value.get("layers") and len(value["layers"]) > 0:
            img = value["layers"][0]
        return extract_pil_image(img)
    if isinstance(value, np.ndarray):
        return Image.fromarray(value)
    if isinstance(value, Image.Image):
        return value
    return value

# ---------------------------------------------------------------------------
# Main inference function called by Gradio
# ---------------------------------------------------------------------------
def run_inference(mode, img_mode1, editor_clean, editor_corrupted, show_metrics):
    """
    Gradio callback supporting 5 modes.
    """
    print(f"\n[run_inference] Mode: {mode}")
    try:
        layers = []
        # 1. Extract base image and drawing layers
        if mode == "Mode 1: Auto-Generated 50% Mask (Validation)":
            pil_image = extract_pil_image(img_mode1)
        elif mode == "Mode 2: Interactive Eraser (Draw Mask on Clean Image)":
            if isinstance(editor_clean, dict):
                pil_image = extract_pil_image(editor_clean.get("background"))
                layers = editor_clean.get("layers", [])
            else:
                pil_image = extract_pil_image(editor_clean)
        elif mode == "Mode 3: Interactive Paint Corruption (Draw Corruption directly)":
            if isinstance(editor_clean, dict):
                pil_image = extract_pil_image(editor_clean.get("composite"))
                layers = editor_clean.get("layers", [])
            else:
                pil_image = extract_pil_image(editor_clean)
        elif mode == "Mode 4: Auto-Detect on Corrupted Image (In-the-Wild)":
            if isinstance(editor_corrupted, dict):
                pil_image = extract_pil_image(editor_corrupted.get("background"))
                if pil_image is None:
                    pil_image = extract_pil_image(editor_corrupted.get("composite"))
            else:
                pil_image = extract_pil_image(editor_corrupted)
        else:  # Mode 5: Manual Mask on Corrupted Image (Precision)
            if isinstance(editor_corrupted, dict):
                pil_image = extract_pil_image(editor_corrupted.get("background"))
                if pil_image is None:
                    pil_image = extract_pil_image(editor_corrupted.get("composite"))
                layers = editor_corrupted.get("layers", [])
            else:
                pil_image = extract_pil_image(editor_corrupted)

        if pil_image is None:
            print("  [run_inference] Error: pil_image is None.")
            return [None] * 5 + [None, "**Please upload or select an image first.**"]

        # Normalize base image
        pil_image = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_tensor = pil_to_tensor(pil_image)   # (1, 3, 224, 224) [-1, 1]

        # Initialize mask and corrupted preview
        mask_1d = torch.zeros(196, dtype=torch.bool)
        corrupted_pil = None

        # Determine original/input image to display
        if mode == "Mode 3: Interactive Paint Corruption (Draw Corruption directly)":
            if isinstance(editor_clean, dict):
                out_orig_pil = extract_pil_image(editor_clean.get("background"))
                if out_orig_pil is not None:
                    out_orig_pil = out_orig_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                else:
                    out_orig_pil = pil_image
            else:
                out_orig_pil = pil_image
        else:
            out_orig_pil = pil_image

        # 2. Generate mask and corrupted preview based on mode
        if mode == "Mode 1: Auto-Generated 50% Mask (Validation)":
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            mask_1d = _mask_gen.generate_mask()
            
            # Build corrupted image for display
            patches = extract_patches(img_tensor.to(DEVICE))
            sorted_idx = torch.argsort(mask_1d.unsqueeze(0).to(torch.int32).to(DEVICE), dim=1)
            num_tgt = int(mask_1d.sum().item())
            tgt_idx = sorted_idx[:, -num_tgt:] if num_tgt > 0 else sorted_idx[:, :0]
            
            corr_patches = patches.clone()
            if num_tgt > 0:
                corr_patches[0, tgt_idx[0]] = -0.8
            corrupted_tensor = reconstruct_image(corr_patches)[0].cpu()
            corrupted_pil = tensor_to_pil(corrupted_tensor)

        elif mode in [
            "Mode 2: Interactive Eraser (Draw Mask on Clean Image)",
            "Mode 3: Interactive Paint Corruption (Draw Corruption directly)",
            "Mode 5: Manual Mask on Corrupted Image (Precision)"
        ]:
            custom_mask_2d = None
            
            # Method 1: Compare composite vs background (highly robust for brush/paint/erasures)
            editor_dict = editor_clean if mode in ["Mode 2: Interactive Eraser (Draw Mask on Clean Image)", "Mode 3: Interactive Paint Corruption (Draw Corruption directly)"] else editor_corrupted
            if isinstance(editor_dict, dict):
                bg = extract_pil_image(editor_dict.get("background"))
                comp = extract_pil_image(editor_dict.get("composite"))
                if bg is not None and comp is not None:
                    bg_rgb = bg.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    comp_rgb = comp.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                    bg_np = np.array(bg_rgb).astype(np.float32)
                    comp_np = np.array(comp_rgb).astype(np.float32)
                    diff_max = np.max(np.abs(comp_np - bg_np), axis=2)
                    # Use a small threshold of 8 to capture color changes while ignoring minor compression noise
                    diff_mask = (diff_max > 8).astype(np.uint8) * 255
                    if np.any(diff_mask > 0):
                        custom_mask_2d = diff_mask
                        print(f"  [run_inference] Detected mask via composite difference: {np.sum(diff_mask > 0)} pixels.")

            # Method 2: Fallback to layers alpha channels
            if custom_mask_2d is None and layers:
                combined_alpha = None
                for layer in layers:
                    if layer is not None:
                        try:
                            layer_pil = extract_pil_image(layer)
                            if layer_pil is not None:
                                layer_rgba = layer_pil.convert("RGBA")
                                np_layer = np.array(layer_rgba)
                                alpha = np_layer[:, :, 3]
                                if combined_alpha is None:
                                    combined_alpha = (alpha > 10).astype(np.uint8) * 255
                                else:
                                    combined_alpha = np.maximum(combined_alpha, (alpha > 10).astype(np.uint8) * 255)
                        except Exception as e:
                            print(f"  [run_inference] Error parsing layer: {e}")
                if combined_alpha is not None and np.any(combined_alpha > 0):
                    custom_mask_2d = combined_alpha
                    print(f"  [run_inference] Detected mask via layers alpha: {np.sum(combined_alpha > 0)} pixels.")

            if custom_mask_2d is not None:
                mask_pil = Image.fromarray(custom_mask_2d).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST)
                np_mask_224 = np.array(mask_pil)
                if len(np_mask_224.shape) == 3:
                    np_mask_224 = np_mask_224[:, :, 0]
                
                for i in range(14):
                    for j in range(14):
                        patch_pixels = np_mask_224[i*16:(i+1)*16, j*16:(j+1)*16]
                        # Sensitive threshold of 2 pixels (about 1% of a 16x16 patch) to catch thin brush strokes
                        if np.sum(patch_pixels > 128) > 2:
                            mask_1d[i*14 + j] = True
                print(f"  [run_inference] Custom mask registered: {int(mask_1d.sum().item())} patches.")
            
            if not torch.any(mask_1d):
                if mode == "Mode 5: Manual Mask on Corrupted Image (Precision)":
                    print("  [run_inference] No custom mask detected, falling back to auto-detect.")
                    mask_1d = detect_corrupted_patches(img_tensor)
                else:
                    print("  [run_inference] No custom mask detected, falling back to random 50% mask.")
                    np.random.seed(RANDOM_SEED)
                    torch.manual_seed(RANDOM_SEED)
                    mask_1d = _mask_gen.generate_mask()
            
            if mode == "Mode 2: Interactive Eraser (Draw Mask on Clean Image)":
                patches = extract_patches(img_tensor.to(DEVICE))
                sorted_idx = torch.argsort(mask_1d.unsqueeze(0).to(torch.int32).to(DEVICE), dim=1)
                num_tgt = int(mask_1d.sum().item())
                tgt_idx = sorted_idx[:, -num_tgt:] if num_tgt > 0 else sorted_idx[:, :0]
                
                corr_patches = patches.clone()
                if num_tgt > 0:
                    corr_patches[0, tgt_idx[0]] = -0.8
                corrupted_tensor = reconstruct_image(corr_patches)[0].cpu()
                corrupted_pil = tensor_to_pil(corrupted_tensor)
            else:
                corrupted_pil = overlay_mask(pil_image, mask_1d)

        else:  # Mode 4: Auto-Detect on Corrupted Image (In-the-Wild)
            mask_1d = detect_corrupted_patches(img_tensor)
            corrupted_pil = overlay_mask(pil_image, mask_1d)
            print(f"  [run_inference] Auto-detected mask: {int(mask_1d.sum().item())} patches.")

        # 3. Extract style features (with de-poisoning for Modes 3, 4, 5)
        if mode in [
            "Mode 3: Interactive Paint Corruption (Draw Corruption directly)",
            "Mode 4: Auto-Detect on Corrupted Image (In-the-Wild)",
            "Mode 5: Manual Mask on Corrupted Image (Precision)"
        ]:
            style_inpainted_pil = _opencv_inpaint(img_tensor, mask_1d)
            style_inpainted_tensor = pil_to_tensor(style_inpainted_pil)
            with torch.no_grad():
                style_vec = extract_hybrid(style_inpainted_tensor[0])
        else:
            with torch.no_grad():
                style_vec = extract_hybrid(img_tensor[0])  # (989,)

        # 4. Classify style
        try:
            probs = _classifier.predict_proba(style_vec.reshape(1, -1))[0]
        except Exception:
            probs = np.ones(len(ALL_CLASSES)) / len(ALL_CLASSES)
        pred_idx = int(np.argmax(probs))

        chart = _style_chart(probs, pred_idx, title="Input Artwork -- Style Classification")

        # 5. Run reconstructions
        cond_t = _reconstruct(img_tensor, style_vec, mask_1d, ablation=None)
        vanilla_t = _reconstruct(img_tensor, style_vec, mask_1d, ablation='zeros')
        opencv_pil = _opencv_inpaint(img_tensor, mask_1d)

        cond_pil = tensor_to_pil(cond_t)
        vanilla_pil = tensor_to_pil(vanilla_t)

        # 6. Compute metrics
        metrics_md = ""
        if show_metrics:
            clean_ref_pil = None
            if mode == "Mode 1: Auto-Generated 50% Mask (Validation)":
                clean_ref_pil = pil_image
            elif mode == "Mode 2: Interactive Eraser (Draw Mask on Clean Image)":
                clean_ref_pil = pil_image
            elif mode == "Mode 3: Interactive Paint Corruption (Draw Corruption directly)":
                if isinstance(editor_clean, dict):
                    clean_ref_pil = extract_pil_image(editor_clean.get("background"))

            if clean_ref_pil is not None:
                clean_ref_pil = clean_ref_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                clean_ref_tensor = pil_to_tensor(clean_ref_pil)
                orig_np = tensor_to_np(clean_ref_tensor[0])
                cond_np = tensor_to_np(cond_t)
                vanilla_np = tensor_to_np(vanilla_t)
                opencv_np = np.array(opencv_pil).astype(np.float32) / 255.0

                ssim_c, psnr_c = _compute_metrics(orig_np, cond_np)
                ssim_v, psnr_v = _compute_metrics(orig_np, vanilla_np)
                ssim_o, psnr_o = _compute_metrics(orig_np, opencv_np)

                try:
                    recon_feat = extract_hybrid(cond_t)
                    recon_pred = int(_classifier.predict(recon_feat.reshape(1, -1))[0])
                    fidelity_str = ("✅ Style Preserved" if recon_pred == pred_idx
                                    else f"⚠ Reconstructed as: {ALL_CLASSES[recon_pred].replace('_',' ').title()}")
                except Exception:
                    fidelity_str = "N/A"

                metrics_md = f"""
### 📊 Reconstruction Metrics

| Method | SSIM ↑ | PSNR (dB) ↑ |
|---|---|---|
| **Style-Conditioned I-JEPA** | **{ssim_c}** | **{psnr_c}** |
| Vanilla I-JEPA (no style) | {ssim_v} | {psnr_v} |
| OpenCV Inpainting | {ssim_o} | {psnr_o} |

### 🎨 Style Fidelity
**Detected Style:** {ALL_CLASSES[pred_idx].replace('_', ' ').title()}

**Conditioned Reconstruction:** {fidelity_str}
                """
            else:
                try:
                    recon_feat = extract_hybrid(cond_t)
                    recon_pred = int(_classifier.predict(recon_feat.reshape(1, -1))[0])
                    fidelity_str = f"🎨 Reconstructed Style: **{ALL_CLASSES[recon_pred].replace('_',' ').title()}**"
                except Exception:
                    fidelity_str = "N/A"

                num_detected = int(mask_1d.sum().item())
                pct_detected = round((num_detected / 196) * 100, 1)

                metrics_md = f"""
### 📊 Reconstruction Status
*Metrics (SSIM/PSNR) are not available because no ground-truth clean image was provided.*

- **Detected Corrupted Patches:** {num_detected} / 196 ({pct_detected}%)
- **Input Style Prediction:** {ALL_CLASSES[pred_idx].replace('_', ' ').title()}
- **Restored Art Style:** {fidelity_str}

> [!NOTE]
> In this in-the-wild mode, the model auto-detected or manually masked the damaged regions shown in red on the **Corrupted / Mask Visualized** panel and predicted their contents using I-JEPA conditioned on the art style detected from the remaining parts of the image.
                """

        return (
            out_orig_pil,
            corrupted_pil,
            cond_pil,
            vanilla_pil,
            opencv_pil,
            chart,
            metrics_md
        )
    except Exception as e:
        import traceback
        print(f"  [run_inference] ERROR: Exception occurred during inference:")
        traceback.print_exc()
        error_msg = f"""
### ❌ Exception During Reconstruction
An error occurred during inference in the Python backend.

**Error Details:**
```text
{str(e)}
```

**Traceback:**
```text
{traceback.format_exc()}
```
"""
        return [None] * 5 + [None, error_msg]

# ---------------------------------------------------------------------------
# Gradio UI construction
# ---------------------------------------------------------------------------
DARK_THEME_CSS = """
* { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0a16 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    color: #e0e0f0 !important;
}

.gr-box, .gr-panel, .gr-form {
    background: rgba(20, 20, 40, 0.7) !important;
    border: 1px solid rgba(108, 99, 255, 0.25) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
}

h1, h2, h3 {
    background: linear-gradient(135deg, #6c63ff 0%, #00e5ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

button.primary {
    background: linear-gradient(135deg, #6c63ff, #00e5ff) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 229, 255, 0.5) !important;
}

.gr-image { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.07) !important; }

label { color: #aaaacc !important; font-weight: 600; font-size: 0.85rem; }

.gr-input, textarea, input[type="text"], input[type="number"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(108, 99, 255, 0.3) !important;
    color: #e0e0f0 !important;
    border-radius: 10px !important;
}

.gr-slider input[type="range"] { accent-color: #6c63ff; }

footer { display: none !important; }
"""

_EXAMPLE_IMAGES = []
# Collect one sample from each class from image_index
import json as _json
_idx_path = os.path.join(FEATURES_DIR, "image_index.json")
if os.path.exists(_idx_path):
    with open(_idx_path) as _f:
        _img_idx = _json.load(_f)
    _seen_classes = set()
    for _item in _img_idx:
        _cls = _item.get('class')
        resolved = resolve_image_path(_item.get('path', ''))
        if _cls not in _seen_classes and resolved:
            _EXAMPLE_IMAGES.append([resolved])
            _seen_classes.add(_cls)
        if len(_seen_classes) >= 17:
            break


def build_app():
    _theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )
    with gr.Blocks(
        title="Historic Image Restoration -- Art Reconstruction System",
    ) as demo:

        gr.HTML("""
        <div style="text-align:center; padding: 32px 0 16px;">
            <h1 style="font-size:2.6rem; margin-bottom: 8px;">
                ✦ Historic Image Restoration
            </h1>
            <p style="color:#8888aa; font-size:1.05rem; max-width:680px; margin:0 auto;">
                Style-Conditioned Art Reconstruction via I-JEPA &amp; Hybrid Feature Analysis.
                Support for 50% Auto Masking, Interactive Brush Drawing, and Custom Pre-Corrupted Image uploads.
            </p>
        </div>
        """)

        with gr.Row():
            # ── Left column: Upload & Controls
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### ⚙ Controls")
                mode_selector = gr.Radio(
                    choices=[
                        "Mode 1: Auto-Generated 50% Mask (Validation)",
                        "Mode 2: Interactive Eraser (Draw Mask on Clean Image)",
                        "Mode 3: Interactive Paint Corruption (Draw Corruption directly)",
                        "Mode 4: Auto-Detect on Corrupted Image (In-the-Wild)",
                        "Mode 5: Manual Mask on Corrupted Image (Precision)"
                    ],
                    value="Mode 1: Auto-Generated 50% Mask (Validation)",
                    label="Select Restoration Mode"
                )

                gr.Markdown("---")
                gr.Markdown("### 🖼 Input Artwork")
                
                # Mode 1: Auto-Mask Image Input
                img_mode1 = gr.Image(
                    label="Clean Image (Auto-Masked)",
                    type="pil",
                    height=320,
                    visible=True,
                    sources=["upload", "webcam", "clipboard"]
                )
                
                # Mode 2 & 3: Interactive Canvas for Clean Image
                editor_clean = gr.ImageEditor(
                    label="Clean Painting Canvas (Brush to mask / corrupt)",
                    type="pil",
                    height=320,
                    visible=False,
                    sources=["upload", "webcam", "clipboard"],
                    transforms=[]
                )
                
                # Mode 4 & 5: Corrupted Canvas
                editor_corrupted = gr.ImageEditor(
                    label="Corrupted Painting Canvas (Brush to mask)",
                    type="pil",
                    height=320,
                    visible=False,
                    sources=["upload", "webcam", "clipboard"],
                    transforms=[]
                )

                show_metrics_cb = gr.Checkbox(
                    label="📐 Show Reconstruction Metrics",
                    value=True
                )
                run_btn = gr.Button("🔮  Reconstruct Painting", variant="primary")

                # Container for Mode 1 Examples
                with gr.Group(visible=True) as examples_group:
                    examples = gr.Examples(
                        examples=_EXAMPLE_IMAGES if _EXAMPLE_IMAGES else [[]],
                        inputs=[img_mode1],
                        label="Sample Clean Paintings (Mode 1)",
                    )

            # ── Right column: Results
            with gr.Column(scale=3):
                gr.Markdown("### 🔬 Reconstruction Comparison")
                with gr.Row():
                    out_orig     = gr.Image(label="Original / Input", height=224, width=224)
                    out_corrupt  = gr.Image(label="Corrupted / Mask Visualized", height=224, width=224)
                    out_cond     = gr.Image(label="✅ Style-Conditioned I-JEPA", height=224, width=224)
                    out_vanilla  = gr.Image(label="Vanilla I-JEPA (no style)", height=224, width=224)
                    out_opencv   = gr.Image(label="OpenCV Inpainting", height=224, width=224)

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=2):
                        chart_out  = gr.Plot(label="Style Classification Confidence")
                    with gr.Column(scale=1):
                        metrics_md = gr.Markdown(value="*Run inference to see metrics.*")

        # Mode switching visibility callback
        def on_mode_change(mode):
            if mode == "Mode 1: Auto-Generated 50% Mask (Validation)":
                return (
                    gr.update(visible=True),   # img_mode1
                    gr.update(visible=False),  # editor_clean
                    gr.update(visible=False),  # editor_corrupted
                    gr.update(visible=True)    # examples_group
                )
            elif mode in [
                "Mode 2: Interactive Eraser (Draw Mask on Clean Image)",
                "Mode 3: Interactive Paint Corruption (Draw Corruption directly)"
            ]:
                return (
                    gr.update(visible=False),  # img_mode1
                    gr.update(visible=True),   # editor_clean
                    gr.update(visible=False),  # editor_corrupted
                    gr.update(visible=False)   # examples_group
                )
            else:  # Mode 4 & 5
                return (
                    gr.update(visible=False),  # img_mode1
                    gr.update(visible=False),  # editor_clean
                    gr.update(visible=True),   # editor_corrupted
                    gr.update(visible=False)   # examples_group
                )

        mode_selector.change(
            fn=on_mode_change,
            inputs=[mode_selector],
            outputs=[img_mode1, editor_clean, editor_corrupted, examples_group]
        )

        # Wire controls
        run_btn.click(
            fn=run_inference,
            inputs=[mode_selector, img_mode1, editor_clean, editor_corrupted, show_metrics_cb],
            outputs=[out_orig, out_corrupt, out_cond, out_vanilla, out_opencv, chart_out, metrics_md]
        )

        gr.HTML("""
        <div style="text-align:center; padding:24px; color:#44445a; font-size:0.8rem;">
            Historic Image Restoration &mdash; AI Art Restoration Research &nbsp;|&nbsp;
            WikiArt-10 + Indian Cultural Heritage Art &nbsp;|&nbsp;
            I-JEPA + Hybrid Feature Analysis
        </div>
        """)

    return demo, _theme, DARK_THEME_CSS


if __name__ == "__main__":
    app, theme, css = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme,
        css=css
    )

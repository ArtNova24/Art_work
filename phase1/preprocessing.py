"""
ANTIGRAVITY Phase 1 — Step 2: Preprocessing Utilities
Shared image loading and conversion functions used by all feature extractors.
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def load_image_all_spaces(image_path: str, size: int = 224):
    """
    Load an image and return it in all required colour spaces.

    Returns:
        img_rgb  : np.ndarray (H, W, 3) uint8  — RGB
        img_hsv  : np.ndarray (H, W, 3) uint8  — HSV (OpenCV scale: H 0-179, S/V 0-255)
        img_lab  : np.ndarray (H, W, 3) uint8  — LAB (L 0-255, A/B 0-255)
        img_gray : np.ndarray (H, W)    uint8  — Grayscale
        img_rgb_norm : np.ndarray (H, W, 3) float32 — RGB normalised [0,1]
    """
    # Load with PIL for robustness (handles EXIF, palette modes, etc.)
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = pil_img.resize((size, size), Image.BICUBIC)

    img_rgb = np.array(pil_img, dtype=np.uint8)  # (H,W,3) RGB

    # Convert via OpenCV (needs BGR input)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_rgb_norm = img_rgb.astype(np.float32) / 255.0

    return img_rgb, img_hsv, img_lab, img_gray, img_rgb_norm


def safe_load(image_path: str, size: int = 224):
    """
    Wrapper around load_image_all_spaces with error handling.
    Returns None on failure.
    """
    try:
        return load_image_all_spaces(image_path, size)
    except Exception as e:
        print(f"  WARNING: Could not load {image_path}: {e}")
        return None

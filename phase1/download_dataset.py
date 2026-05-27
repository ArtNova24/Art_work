"""
Historic Image Restoration Phase 1 — Step 1: Dataset Download & Preparation (Streaming version)
Downloads WikiArt-10 from HuggingFace using fast sharded streaming from Artificio/WikiArt.
Handles Surrealism and other styles to reach 500 images per class.
Indexes the Indian Art-8 dataset.
Organises images into data/wikiart/<style>/ folders.
Generates style_mapping.json (18 classes: 10 WikiArt + 8 Indian).
"""

import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add parent to path so config is importable
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, INDIAN_DIR, FEATURES_DIR, WIKIART_STYLE_MAP,
    WIKIART_CLASSES, INDIAN_FOLDER_MAP, INDIAN_CLASSES,
    ALL_CLASSES, CLASS_TO_IDX, MAX_WIKIART_PER_CLASS, RANDOM_SEED
)

random.seed(RANDOM_SEED)

# Mapping from Artificio/WikiArt styles to our clean classes
ARTIFICIO_MAP = {
    "Abstract Expressionism":    "abstract_expressionism",
    "Cubism":                   "cubism",
    "Baroque":                  "baroque",
    "Impressionism":            "impressionism",
    "Romanticism":              "romanticism",
    "Art Nouveau (Modern)":     "art_nouveau",
    "Art Nouveau":              "art_nouveau",
    "Minimalism":               "minimalism",
    "Pop Art":                  "pop_art",
    "Surrealism":               "surrealism",
    "Early Renaissance":        "renaissance",
    "High Renaissance":         "renaissance",
    "Northern Renaissance":     "renaissance",
}

def download_wikiart():
    """Download missing WikiArt images from HuggingFace Artificio/WikiArt and save to data/wikiart/."""
    print("\n" + "="*60)
    print("  Downloading WikiArt-10 from HuggingFace (Artificio Streaming)")
    print("="*60)

    # Create output folders
    for cls in WIKIART_CLASSES:
        os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

    # Initialise current saved counts
    saved_counts = {}
    for cls in WIKIART_CLASSES:
        saved_counts[cls] = len(list(Path(DATA_DIR, cls).glob("*.jpg")))

    print(f"\n  Images already on disk: {sum(saved_counts.values())}")
    for cls in WIKIART_CLASSES:
        print(f"    {cls:30s}: {saved_counts[cls]} images")

    # Check if we need more images
    needed_classes = {cls for cls in WIKIART_CLASSES if saved_counts[cls] < MAX_WIKIART_PER_CLASS}
    if not needed_classes:
        print("  All WikiArt classes already have enough images. Skipping download.")
        return saved_counts

    total_needed = sum(max(0, MAX_WIKIART_PER_CLASS - saved_counts[cls]) for cls in WIKIART_CLASSES)
    print(f"\n  Downloading {total_needed} more images for classes: {needed_classes}")

    # Load dataset in streaming mode
    from datasets import load_dataset
    print("  Connecting to Artificio/WikiArt dataset on HF...")
    ds = load_dataset("Artificio/WikiArt", split="train", streaming=True)

    # Stream and download
    with tqdm(total=total_needed, desc="  Saving WikiArt images", unit="img") as pbar:
        for example in ds:
            # Check if all classes are full
            if all(saved_counts[cls] >= MAX_WIKIART_PER_CLASS for cls in WIKIART_CLASSES):
                print("\n  ✓ All target classes reached their caps!")
                break

            hf_style = example.get("style")
            if not hf_style or hf_style not in ARTIFICIO_MAP:
                continue

            target_cls = ARTIFICIO_MAP[hf_style]
            count = saved_counts[target_cls]

            if count >= MAX_WIKIART_PER_CLASS:
                continue

            # Load and save image
            try:
                img = example.get("image")
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    # Try to open from bytes if it's a dict
                    import io
                    if isinstance(img, dict) and "bytes" in img:
                        img = Image.open(io.BytesIO(img["bytes"]))
                    else:
                        continue

                img = img.convert("RGB")
                save_path = os.path.join(DATA_DIR, target_cls, f"{target_cls}_{count:05d}.jpg")
                img.save(save_path, "JPEG", quality=95)

                saved_counts[target_cls] += 1
                pbar.update(1)
            except Exception as e:
                # Silent fail to keep progress bar clean
                pass

    print("\n  WikiArt-10 download complete:")
    for cls in WIKIART_CLASSES:
        print(f"    {cls:30s}: {saved_counts[cls]} images")

    return saved_counts


def verify_indian_art():
    """Verify and index the Indian Art-8 dataset from the local folder."""
    print("\n" + "="*60)
    print("  Verifying Indian Art-8 Dataset")
    print("="*60)

    counts = {}
    for folder_name, class_name in INDIAN_FOLDER_MAP.items():
        folder_path = os.path.join(INDIAN_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"  WARNING: Folder not found: {folder_path}")
            counts[class_name] = 0
            continue
        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        counts[class_name] = len(images)
        print(f"  {class_name:25s}: {len(images)} images  ← {folder_path}")

    print(f"\n  Total Indian art images: {sum(counts.values())}")
    return counts


def generate_style_mapping():
    """Generate style_mapping.json mapping integer index → class name."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    mapping = {str(i): cls for i, cls in enumerate(ALL_CLASSES)}
    out_path = os.path.join(FEATURES_DIR, "style_mapping.json")
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n  Saved style_mapping.json → {out_path}")
    print(f"  18 classes: {ALL_CLASSES}")
    return mapping


def build_image_index():
    """
    Build a master list of all (image_path, class_idx) pairs.
    Saves as image_index.json for use by all feature extractors.
    """
    print("\n" + "="*60)
    print("  Building Master Image Index")
    print("="*60)

    index = []

    # WikiArt-10
    for cls in WIKIART_CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"  WARNING: {cls_dir} not found")
            continue
        imgs = sorted([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for fname in imgs:
            index.append({
                "path": os.path.join(cls_dir, fname),
                "class": cls,
                "class_idx": CLASS_TO_IDX[cls],
                "source": "wikiart"
            })

    # Indian Art-8
    for folder_name, class_name in INDIAN_FOLDER_MAP.items():
        folder_path = os.path.join(INDIAN_DIR, folder_name)
        if not os.path.exists(folder_path):
            continue
        imgs = sorted([f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
        for fname in imgs:
            index.append({
                "path": os.path.join(folder_path, fname),
                "class": class_name,
                "class_idx": CLASS_TO_IDX[class_name],
                "source": "indian"
            })

    # Shuffle for diversity
    random.shuffle(index)

    out_path = os.path.join(FEATURES_DIR, "image_index.json")
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Total images indexed: {len(index)}")
    per_class = {}
    for item in index:
        per_class[item['class']] = per_class.get(item['class'], 0) + 1
    print("\n  Per-class image counts:")
    for cls in ALL_CLASSES:
        src = "WikiArt" if cls in WIKIART_CLASSES else "Indian"
        print(f"    [{src:7s}] {cls:30s}: {per_class.get(cls, 0)}")

    return index


if __name__ == "__main__":
    print("\n" + "*"*60)
    print("  Historic Image Restoration Phase 1 — Step 1: Dataset Preparation (Artificio Stream)")
    print("*"*60)

    # Disable Xet download to avoid potential hangs/stalls
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    # Step 1a: Download WikiArt from HuggingFace
    wikiart_counts = download_wikiart()

    # Step 1b: Verify Indian art dataset
    indian_counts = verify_indian_art()

    # Step 1c: Generate style mapping
    generate_style_mapping()

    # Step 1d: Build master image index
    build_image_index()

    print("\n  ✓ Step 1 complete. Ready for feature extraction.\n")

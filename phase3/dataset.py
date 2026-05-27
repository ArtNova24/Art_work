"""
ANTIGRAVITY Phase 3 — Style JEPA Dataset.
Loads preprocessed raw image tensors alongside pre-saved,
optimized hybrid 989-dimensional style vectors.
Uses deterministic split matching to guarantee alignment.
All prints and comments are kept strictly in ASCII.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Import path parameters
from phase3.config import FEATURES_DIR, IMG_SIZE, RANDOM_SEED
from phase1.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO

class StyleJEPAImageDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        split: 'train', 'val', or 'test'
        """
        self.split = split
        
        # 1. Load pre-saved pre-split features and labels
        features_path = os.path.join(FEATURES_DIR, f"features_{split}.npy")
        labels_path = os.path.join(FEATURES_DIR, f"labels_{split}.npy")
        
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Pre-split feature files not found for split {split}! "
                f"Please ensure Phase 2/2.5 was executed completely."
            )
            
        self.features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)
        
        # 2. Load unified image index to map indices back to raw image file paths
        index_path = os.path.join(FEATURES_DIR, "image_index.json")
        with open(index_path, 'r') as f:
            self.image_index = json.load(f)
            
        # 3. Deterministically reconstruct the original train/val/test index splits
        indices = np.arange(len(self.image_index))
        all_labels = np.array([item['class_idx'] for item in self.image_index], dtype=np.int32)
        
        # Perform exact same stratified split used in Phase 1 & 2
        temp_idx, test_idx = train_test_split(
            indices, test_size=TEST_RATIO, stratify=all_labels, random_state=RANDOM_SEED
        )
        val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        train_idx, val_idx = train_test_split(
            temp_idx, test_size=val_ratio_adjusted, stratify=all_labels[temp_idx], random_state=RANDOM_SEED
        )
        
        if split == 'train':
            self.split_indices = train_idx
        elif split == 'val':
            self.split_indices = val_idx
        elif split == 'test':
            self.split_indices = test_idx
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # Verify sizes match
        assert len(self.split_indices) == len(self.features), \
            f"Shape mismatch: {len(self.split_indices)} file indices vs {len(self.features)} loaded features."
            
        # 4. Standard image preprocessing transforms
        if transform is not None:
            self.transform = transform
        else:
            # Reconstruct and normalize images strictly to range [-1, 1] using Tanh-friendly scaling
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Retrieve the unified index for the target item in the current split
        real_idx = self.split_indices[idx]
        item = self.image_index[real_idx]
        
        img_path = item['path']
        label = self.labels[idx]
        style_vector = self.features[idx]
        
        # Load raw image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback if image file is corrupted or missing
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            
        # Transform image
        img_tensor = self.transform(img)
        
        return img_tensor, torch.tensor(style_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

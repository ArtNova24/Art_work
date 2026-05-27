"""
ANTIGRAVITY Phase 3 — Block-Wise Masking Pipeline.
Generates contiguous block masks over the 14x14 patch grid,
simulating realistic physical damage to paintings while maintaining
a fixed 50% mask ratio (exactly 98 context and 98 target patches)
for clean, high-speed PyTorch batched tensor processing.
All prints and comments are kept strictly in ASCII.
"""
import numpy as np
import torch

class BlockMaskGenerator:
    def __init__(self, grid_size=14, target_masked=98):
        self.grid_size = grid_size
        self.total_patches = grid_size * grid_size
        self.target_masked = target_masked  # Exactly 98 out of 196 (50%)

    def generate_mask(self):
        """
        Generates a block-wise random mask with exactly target_masked patches.
        Returns:
            mask: 1D torch.BoolTensor of shape (total_patches,) where True is MASKED (target)
                  and False is INTACT (context).
        """
        # Start with all zero mask (False = intact / context)
        mask_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        current_masked_count = 0
        attempts = 0
        
        # Phase 1: Contiguous Block Masking
        while current_masked_count < self.target_masked and attempts < 100:
            attempts += 1
            # Random block dimensions (width/height from 2 to 5 patches)
            block_h = np.random.randint(2, 6)
            block_w = np.random.randint(2, 6)
            
            # Random starting coordinate
            y = np.random.randint(0, self.grid_size)
            x = np.random.randint(0, self.grid_size)
            
            y_end = min(y + block_h, self.grid_size)
            x_end = min(x + block_w, self.grid_size)
            
            # Mask the region
            mask_grid[y:y_end, x:x_end] = True
            current_masked_count = np.sum(mask_grid)
            
        # Flatten mask
        mask_flat = mask_grid.flatten()
        
        # Phase 2: Force exact count correction
        masked_indices = np.where(mask_flat == True)[0]
        unmasked_indices = np.where(mask_flat == False)[0]
        
        if len(masked_indices) > self.target_masked:
            # Too many masked: randomly unmask some
            excess = len(masked_indices) - self.target_masked
            unmask_choice = np.random.choice(masked_indices, size=excess, replace=False)
            mask_flat[unmask_choice] = False
        elif len(masked_indices) < self.target_masked:
            # Too few masked: randomly mask some
            deficit = self.target_masked - len(masked_indices)
            mask_choice = np.random.choice(unmasked_indices, size=deficit, replace=False)
            mask_flat[mask_choice] = True
            
        return torch.from_numpy(mask_flat)

    def collate_masks(self, batch_size):
        """
        Generates a batch of masks.
        Returns:
            masks: BoolTensor of shape (batch_size, total_patches)
        """
        batch_masks = []
        for _ in range(batch_size):
            batch_masks.append(self.generate_mask())
        return torch.stack(batch_masks)

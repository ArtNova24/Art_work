"""
ANTIGRAVITY Phase 3 — Neural Network Models.
Implements:
  1. StyleProjector: Projects 989-dim hybrid style vector to 256-dim.
  2. Attention & TransformerBlock: Pure PyTorch implementation of Vit blocks.
  3. ViTContextEncoder: Context encoder (processes unmasked patches).
  4. StyleConditionedPredictor: Predictor decoder (conditioned on style embedding).
  5. PixelDecoder: Lightweight convolutional / MLP reconstruction decoder.
All prints and comments are kept strictly in ASCII.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Style Embedding Projector
class StyleProjector(nn.Module):
    def __init__(self, input_dim=989, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.net(x)


# -- Self-Attention block
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -- Transformer Block (Encoder/Decoder layer)
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -- Vision Transformer Context / Target Encoder
class ViTContextEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256, depth=6, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch projection
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None, get_all=False):
        """
        x: (B, 3, 224, 224)
        mask: (B, 196) Boolean tensor where True is masked (target) and False is intact (context).
        get_all: if True, skips masking and processes the entire image (used by target encoder).
        """
        B = x.shape[0]
        # Patch projection
        patches = self.patch_embed(x)  # (B, embed_dim, 14, 14)
        patches = patches.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        
        # Add position embeddings
        patches = patches + self.pos_embed
        
        if get_all or mask is None:
            # Process entire sequence
            for block in self.blocks:
                patches = block(patches)
            return self.norm(patches)
        
        # Masked selection: Gather context patches (False in mask)
        # Sort mask: False (0) elements sort before True (1) elements
        sorted_indices = torch.argsort(mask.to(torch.int32), dim=1)
        num_ctx = self.num_patches - mask.sum(dim=1)[0].item() # 196 - 98 = 98
        
        context_indices = sorted_indices[:, :num_ctx]
        
        # Gather patches
        ctx_patches = torch.gather(patches, 1, context_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        
        # Process context sequence
        for block in self.blocks:
            ctx_patches = block(ctx_patches)
            
        return self.norm(ctx_patches), context_indices


# -- Style-Conditioned Predictor (Transformer Decoder)
class StyleConditionedPredictor(nn.Module):
    def __init__(self, embed_dim=256, depth=4, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = 196
        
        # Learnable masked-patch token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Target Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_ctx, s_emb, mask):
        """
        z_ctx: (B, N_ctx, embed_dim) Context latents from encoder.
        s_emb: (B, embed_dim) Projected style embedding vector.
        mask: (B, 196) Boolean mask grid.
        """
        B = z_ctx.shape[0]
        
        # Determine target (masked) indices (True in mask)
        sorted_indices = torch.argsort(mask.to(torch.int32), dim=1)
        num_ctx = self.num_patches - mask.sum(dim=1)[0].item()
        
        target_indices = sorted_indices[:, num_ctx:] # Select target indices
        
        # Gather target positional encodings
        target_pos = torch.gather(self.pos_embed.expand(B, -1, -1), 1, target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        
        # Initialize target patches as Mask Token + Target Positional Encoding
        target_tokens = self.mask_token.expand(B, target_indices.shape[1], -1) + target_pos
        
        # Prepare style token (B, 1, embed_dim)
        style_token = s_emb.unsqueeze(1)
        
        # Concatenate: [Style Token; Context Latents; Target Tokens]
        # Shape: (B, 1 + N_ctx + N_tgt, embed_dim) -> (B, 1 + 98 + 98, 256) -> (B, 197, 256)
        x = torch.cat([style_token, z_ctx, target_tokens], dim=1)
        
        # Process predictor sequence
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Extract target predictions (the last N_tgt tokens in the sequence)
        z_tgt_pred = x[:, -target_indices.shape[1]:]
        
        return z_tgt_pred, target_indices


# -- Lightweight Pixel Decoder
class PixelDecoder(nn.Module):
    def __init__(self, embed_dim=256, patch_size=16, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, channels * patch_size * patch_size),
            nn.Tanh()  # Outputs pixels in range [-1, 1]
        )

    def forward(self, z_tgt_pred):
        """
        z_tgt_pred: (B, N_tgt, embed_dim)
        Returns: Reconstructed target patches of shape (B, N_tgt, 3, 16, 16)
        """
        B, N_tgt, _ = z_tgt_pred.shape
        pixels = self.net(z_tgt_pred)  # (B, N_tgt, 3 * 16 * 16)
        pixels = pixels.reshape(B, N_tgt, self.channels, self.patch_size, self.patch_size)
        return pixels

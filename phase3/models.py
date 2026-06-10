"""
Historic Image Restoration Phase 3 — Neural Network Models.
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


# -- Vision Transformer Context / Target Encoder (Pretrained ViT MAE)
class ViTContextEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=6, heads=8, mlp_ratio=4.0, dropout=0.1, pretrained=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2  # 196
        
        import timm
        # Load pretrained ViT-B/16 (MAE pretrained)
        self.vit = timm.create_model(
            'vit_base_patch16_224.mae',   # MAE pretrained weights
            pretrained=pretrained,
            num_classes=0,                # Remove classification head
            global_pool='',               # Return all patch tokens
        )

        # Projection head to match downstream embed_dim if needed
        self.proj = nn.Linear(768, embed_dim) if embed_dim != 768 else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None, get_all=False):
        """
        x: (B, 3, 224, 224)
        mask: (B, 196) Boolean tensor where True is masked (target) and False is intact (context).
        get_all: if True, skips masking and processes the entire image (used by target encoder).
        """
        B = x.shape[0]

        # Extract all patch tokens from pretrained ViT
        # Returns (B, 197, 768) — 196 patches + 1 CLS token
        features = self.vit.forward_features(x)
        patches = features[:, 1:, :]  # Remove CLS → (B, 196, 768)
        patches = self.proj(patches)  # → (B, 196, embed_dim)
        patches = self.norm(patches)

        if get_all or mask is None:
            return patches

        # Apply masking: keep only context (unmasked) patches (False in mask)
        sorted_indices = torch.argsort(mask.to(torch.int32), dim=1)
        num_ctx = self.num_patches - int(mask.sum(dim=1)[0].item())
        context_indices = sorted_indices[:, :num_ctx]

        ctx_patches = torch.gather(
            patches, 1,
            context_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )
        return ctx_patches, context_indices


# -- FiLM Conditioning Layer
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Given a style vector s of shape (B, style_dim),
    learns per-feature scale and shift for a feature of shape (B, N, feat_dim).
    """
    def __init__(self, style_dim, feat_dim):
        super().__init__()
        self.gamma_net = nn.Linear(style_dim, feat_dim)
        self.beta_net  = nn.Linear(style_dim, feat_dim)
        # Initialize to identity (no modulation at start)
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, x, style):
        """
        x     : (B, N, feat_dim)
        style : (B, style_dim)
        """
        gamma = self.gamma_net(style).unsqueeze(1)  # (B, 1, feat_dim)
        beta  = self.beta_net(style).unsqueeze(1)   # (B, 1, feat_dim)
        return gamma * x + beta                     # Broadcast over N


# -- Transformer Block with FiLM
class FiLMTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1, style_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        # FiLM layer applied AFTER each sub-layer
        self.film1 = FiLMLayer(style_dim, dim)
        self.film2 = FiLMLayer(style_dim, dim)

    def forward(self, x, style=None):
        # Self-attention with FiLM modulation
        attn_out = self.attn(self.norm1(x))
        if style is not None:
            attn_out = self.film1(attn_out, style)
        x = x + attn_out

        # MLP with FiLM modulation
        mlp_out = self.mlp(self.norm2(x))
        if style is not None:
            mlp_out = self.film2(mlp_out, style)
        x = x + mlp_out
        return x


# -- Style-Conditioned Predictor (Transformer Decoder with FiLM)
class StyleConditionedPredictor(nn.Module):
    def __init__(self, embed_dim=768, depth=4, heads=8, mlp_ratio=4.0, dropout=0.1, style_dim=256):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_patches = 196

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Use FiLM-conditioned transformer blocks instead of plain blocks
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(
                dim=embed_dim, heads=heads,
                mlp_ratio=mlp_ratio, dropout=dropout,
                style_dim=style_dim
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_ctx, s_emb, mask):
        """
        z_ctx: (B, N_ctx, embed_dim) Context latents from encoder.
        s_emb: (B, style_dim) Projected style embedding vector.
        mask: (B, 196) Boolean mask grid.
        """
        B = z_ctx.shape[0]

        sorted_indices = torch.argsort(mask.to(torch.int32), dim=1)
        num_ctx = self.num_patches - int(mask.sum(dim=1)[0].item())
        target_indices = sorted_indices[:, num_ctx:]

        target_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), 1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )
        target_tokens = self.mask_token.expand(
            B, target_indices.shape[1], -1
        ) + target_pos

        # Concatenate context + target (NO style prepend token anymore)
        x = torch.cat([z_ctx, target_tokens], dim=1)

        # Pass style through FiLM at EVERY layer — cannot be ignored
        for block in self.blocks:
            x = block(x, style=s_emb)   # <-- style conditions every block

        x = self.norm(x)
        z_tgt_pred = x[:, -target_indices.shape[1]:]
        return z_tgt_pred, target_indices


# -- Convolutional Pixel Decoder
class PixelDecoder(nn.Module):
    """
    Convolutional pixel decoder.
    Input : (B, N_tgt, embed_dim) latent predictions
    Output: (B, N_tgt, 3, 16, 16) pixel patches in [-1, 1]
    """
    def __init__(self, embed_dim=768, patch_size=16, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.channels   = channels
        self.embed_dim  = embed_dim

        # Project latent to spatial feature map seed
        self.fc = nn.Linear(embed_dim, 512 * 2 * 2)  # 512 channels, 2x2 spatial

        # Convolutional upsampling: 2x2 → 4x4 → 8x8 → 16x16
        self.conv_decoder = nn.Sequential(
            # 2x2 → 4x4
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 4x4 → 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 8x8 → 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Final: map to RGB channels
            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z_tgt_pred):
        """
        z_tgt_pred: (B, N_tgt, embed_dim)
        Returns   : (B, N_tgt, 3, 16, 16)
        """
        B, N_tgt, _ = z_tgt_pred.shape

        # Flatten patches into batch dimension for conv processing
        z_flat = z_tgt_pred.reshape(B * N_tgt, self.embed_dim)

        # Linear projection to spatial seed
        x = self.fc(z_flat)                          # (B*N_tgt, 512*4)
        x = x.reshape(B * N_tgt, 512, 2, 2)          # (B*N_tgt, 512, 2, 2)

        # Convolutional upsampling to 16x16
        x = self.conv_decoder(x)                     # (B*N_tgt, 3, 16, 16)

        # Reshape back to patch sequence
        x = x.reshape(B, N_tgt, self.channels,
                       self.patch_size, self.patch_size)
        return x

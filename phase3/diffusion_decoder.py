"""
Historic Image Restoration Phase 3 — Conditional Diffusion Decoder.
Implements DDPM noise scheduling and small conditional ResNet-based U-Net
for patch-level visual restoration (16x16 pixels).
All prints and comments are kept strictly in ASCII.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal time embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock(nn.Module):
    """Residual block conditioned on time + style embeddings."""
    def __init__(self, in_ch, out_ch, time_emb_dim, style_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Time conditioning
        self.time_mlp = nn.Linear(time_emb_dim, out_ch * 2)
        # Style conditioning (FiLM)
        self.style_mlp = nn.Linear(style_dim, out_ch * 2)

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb, style_emb):
        h = self.conv1(F.gelu(self.norm1(x)))

        # Time modulation
        t_scale, t_shift = self.time_mlp(time_emb).chunk(2, dim=-1)
        h = h * (t_scale[:, :, None, None] + 1) + t_shift[:, :, None, None]

        # Style modulation (FiLM)
        s_scale, s_shift = self.style_mlp(style_emb).chunk(2, dim=-1)
        h = h * (s_scale[:, :, None, None] + 1) + s_shift[:, :, None, None]

        h = self.conv2(F.gelu(self.norm2(h)))
        return h + self.res_conv(x)


class DiffusionPatchDecoder(nn.Module):
    """
    Small U-Net style denoising network for 16x16 patches.
    Conditioned on:
      - Diffusion timestep t
      - Predicted latent z_tgt (from I-JEPA predictor)
      - Style embedding s_emb (from StyleProjector)
    """
    def __init__(self, latent_dim=768, style_dim=256,
                 channels=3, patch_size=16, time_emb_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Project latent to image space for input conditioning
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, 64 * patch_size * patch_size),
            nn.GELU()
        )

        # Noisy patch: 3 channels + latent conditioning: 64 channels = 67 input channels
        self.down1 = ConditionalResBlock(3 + 64, 64,  time_emb_dim, style_dim)
        self.down2 = ConditionalResBlock(64,    128, time_emb_dim, style_dim)
        self.mid   = ConditionalResBlock(128,   128, time_emb_dim, style_dim)
        self.up1   = ConditionalResBlock(256,   64,  time_emb_dim, style_dim)
        self.up2   = ConditionalResBlock(128,   64,  time_emb_dim, style_dim)
        self.out   = nn.Conv2d(64, channels, 1)

    def forward(self, x_noisy, t, z_tgt, s_emb):
        """
        x_noisy : (B*N_tgt, 3, 16, 16)   noisy patch
        t       : (B*N_tgt,)              diffusion timestep
        z_tgt   : (B*N_tgt, latent_dim)  predicted latent
        s_emb   : (B*N_tgt, style_dim)   style embedding (broadcast)
        """
        BN = x_noisy.shape[0]

        # Time embedding
        t_emb = self.time_emb(t)

        # Project latent to spatial conditioning
        lat = self.latent_proj(z_tgt).reshape(BN, 64, 16, 16)

        # Concatenate noisy patch with latent conditioning
        x = torch.cat([x_noisy, lat], dim=1)   # (BN, 67, 16, 16)

        # Encoder path
        h1 = self.down1(x, t_emb, s_emb)       # (BN, 64, 16, 16)
        h2 = self.down2(h1, t_emb, s_emb)      # (BN, 128, 16, 16)
        h  = self.mid(h2, t_emb, s_emb)        # (BN, 128, 16, 16)

        # Decoder path with skip connections
        h = self.up1(torch.cat([h, h2], dim=1), t_emb, s_emb)  # (BN, 64, 16, 16)
        h = self.up2(torch.cat([h, h1], dim=1), t_emb, s_emb)  # (BN, 64, 16, 16)

        return self.out(h)   # (BN, 3, 16, 16) — predicted noise


class DDPM:
    """DDPM noise scheduler for patch-level diffusion."""
    def __init__(self, num_timesteps=1000, device='cpu'):
        self.T = num_timesteps
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    def add_noise(self, x0, t):
        """Add noise to clean patches x0 at timestep t."""
        alpha_t = self.alphas_cumprod[t][:, None, None, None]
        noise = torch.randn_like(x0)
        return (alpha_t.sqrt() * x0 + (1 - alpha_t).sqrt() * noise), noise

    @torch.no_grad()
    def sample(self, model, z_tgt, s_emb, shape, device, num_steps=50):
        """DDIM-style fast sampling from the diffusion decoder."""
        BN = z_tgt.shape[0]
        x = torch.randn(shape, device=device)  # Start from pure noise

        timesteps = torch.linspace(self.T - 1, 0, num_steps, dtype=torch.long, device=device)
        for t_val in timesteps:
            t_batch = t_val.expand(BN)
            pred_noise = model(x, t_batch, z_tgt, s_emb)
            alpha_t = self.alphas_cumprod[t_val]
            x = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            if t_val > 0:
                x += (1 - self.alphas_cumprod[max(t_val - 1, 0)]).sqrt() * 0.1 * torch.randn_like(x)
        return x.clamp(-1, 1)

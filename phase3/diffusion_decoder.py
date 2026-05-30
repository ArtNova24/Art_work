"""
Historic Image Restoration Phase 3 - Conditional Diffusion Decoder with GuidePaint.

Implements (in this single file):
  1. SinusoidalPosEmb       : sinusoidal timestep embedding
  2. ConditionalResBlock    : FiLM-conditioned residual block (time + style)
  3. DiffusionPatchDecoder  : U-Net denoising network for 16x16 patches
                              + latent_to_pixel reference head (for GuidePaint y)
  4. DDPM                   : Noise scheduler with THREE sampling modes:
       a. sample()           : Fast DDIM (no guidance, speed-critical path)
       b. sample_guided()    : GuidePaint lossless image-guided sampling [Algorithm 1]
       c. (interrupted)      : Interrupted sampling via interrupt_at_t kwarg

Key equations from GuidePaint (Yu et al., 2025, npj Heritage Science,
  DOI: 10.1038/s40494-025-01693-z):
  DDPM forward [Eq.2]: q(x_t|x_0) ~ N(sqrt(abar_t)*x_0, (1-abar_t)*I)
  Training loss [Eq.6]: L = E[||eps - eps_theta(x_t, t)||^2]   (standard MSE)
  Reverse mean [Eq.4]: mu_theta = (1/sqrt(a_t))*(x_t - (beta_t/sqrt(1-abar_t))*eps)
  x0 estimate [Eq.8]:  x0_est = (1/sqrt(abar_t))*(x_t - sqrt(1-abar_t)*eps_theta)
  Similarity   [Eq.9]:  F(x_t,y,t) = -||(1-m)*x0_est - (1-m)*y||^2
  Guided step  [Eq.7]:  x_{t-1} = mu_theta + gamma*Sigma*grad_xt_F + Sigma*noise
  Lossless end [Line9]: x = (1-m)*y + m*x0(x_1,t)

In our patch-level system:
  - x0      : clean target patches (B*N_tgt, 3, 16, 16)
  - y_ref   : pixel-space reference derived from z_tgt via latent_to_pixel head
  - m       : 1 everywhere per patch (all target patches are fully unknown)
              => (1-m) = 1, so F = -||x0_est - y_ref||^2 (full-patch guidance)
  - gamma   : configurable (GUIDEPAINT_GAMMA in config.py, paper used 0.001)
  - interrupt_at_t : for removing unmarkable degradations (e.g. 249 in paper)
  - seed    : different seeds -> diverse restoration outputs (GuidePaint Sec.3)

All prints and comments are kept strictly in ASCII.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sinusoidal Timestep Embedding
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal time embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device   = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (N, dim)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Conditional Residual Block
# ─────────────────────────────────────────────────────────────────────────────
class ConditionalResBlock(nn.Module):
    """
    Residual block conditioned on time + style embeddings via FiLM.
    scale and shift are produced from both time_emb and style_emb separately,
    giving the network independent control of temporal and stylistic conditioning.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, style_dim):
        super().__init__()
        # Use a valid group count for every channel count. The U-Net combines
        # 3 noisy RGB channels with a 64-channel latent map, so in_ch = 67
        # is not divisible by 8; GroupNorm(8, 67) would crash at runtime.
        self.norm1 = nn.GroupNorm(1 if in_ch % 8 != 0 else 8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(1 if out_ch % 8 != 0 else 8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Time conditioning: produces per-channel scale and shift
        self.time_mlp  = nn.Linear(time_emb_dim, out_ch * 2)
        # Style conditioning (FiLM): produces per-channel scale and shift
        self.style_mlp = nn.Linear(style_dim, out_ch * 2)

        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb, style_emb):
        h = self.conv1(F.gelu(self.norm1(x)))

        # Time FiLM modulation
        t_scale, t_shift = self.time_mlp(time_emb).chunk(2, dim=-1)
        h = h * (t_scale[:, :, None, None] + 1) + t_shift[:, :, None, None]

        # Style FiLM modulation
        s_scale, s_shift = self.style_mlp(style_emb).chunk(2, dim=-1)
        h = h * (s_scale[:, :, None, None] + 1) + s_shift[:, :, None, None]

        h = self.conv2(F.gelu(self.norm2(h)))
        return h + self.res_conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Diffusion Patch Decoder (U-Net denoiser + GuidePaint reference head)
# ─────────────────────────────────────────────────────────────────────────────
class DiffusionPatchDecoder(nn.Module):
    """
    Small U-Net style denoising network for 16x16 patches.
    Conditioned on:
      - Diffusion timestep t
      - Predicted latent z_tgt  (from I-JEPA predictor)
      - Style embedding s_emb   (from StyleProjector)

    GuidePaint extension:
      - latent_to_pixel head: z_tgt -> pixel-space reference y  [GuidePaint Eq.9]
        This head produces the reference signal used during guided sampling to
        steer x0_est toward the latent-consistent pixel content, without any
        lossy feature compression (the GuidePaint key contribution).
    """
    def __init__(self, latent_dim=768, style_dim=256,
                 channels=3, patch_size=16, time_emb_dim=128):
        super().__init__()
        self.channels   = channels
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        # ── Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # ── Latent-to-spatial conditioning (shared for U-Net input)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, 64 * patch_size * patch_size),
            nn.GELU()
        )

        # ── GuidePaint reference head: z_tgt -> pixel-space estimate y [Eq.9]
        # Provides a lossless pixel-level reference for gradient-guided sampling.
        # Uses a shallow MLP: lossy compression (e.g. VGG features) is deliberately
        # avoided — this is the "lossless" in GuidePaint's name.
        self.latent_to_pixel = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, channels * patch_size * patch_size),
            nn.Tanh()   # output in [-1, 1] matching clean patch range
        )

        # ── U-Net blocks: input = noisy patch (3ch) + latent conditioning (64ch) = 67ch
        self.down1 = ConditionalResBlock(3 + 64, 64,  time_emb_dim, style_dim)
        self.down2 = ConditionalResBlock(64,    128,  time_emb_dim, style_dim)
        self.mid   = ConditionalResBlock(128,   128,  time_emb_dim, style_dim)
        self.up1   = ConditionalResBlock(256,   64,   time_emb_dim, style_dim)
        self.up2   = ConditionalResBlock(128,   64,   time_emb_dim, style_dim)
        self.out   = nn.Conv2d(64, channels, 1)

    def get_pixel_reference(self, z_tgt):
        """
        Produces the pixel-space reference y for GuidePaint Eq.9.
        This is the lossless equivalent of the conditional image y in the paper.

        In our patch-level system, we use the I-JEPA latent z_tgt as the source
        of structural knowledge (analogous to the intact known-region pixels in the
        original GuidePaint for full-image restoration).

        z_tgt : (BN, latent_dim)
        Returns: (BN, 3, 16, 16) reference in [-1, 1]
        """
        BN  = z_tgt.shape[0]
        ref = self.latent_to_pixel(z_tgt)   # (BN, 3*16*16)
        return ref.reshape(BN, self.channels, self.patch_size, self.patch_size)

    def forward(self, x_noisy, t, z_tgt, s_emb):
        """
        Predicts the noise epsilon added at timestep t to patch x_noisy.

        x_noisy : (BN, 3, 16, 16)   noisy patch at timestep t
        t       : (BN,)              integer diffusion timestep
        z_tgt   : (BN, latent_dim)  I-JEPA predicted latent
        s_emb   : (BN, style_dim)   projected style embedding

        Returns : (BN, 3, 16, 16)   predicted noise epsilon_theta
        """
        BN    = x_noisy.shape[0]
        t_emb = self.time_emb(t)   # (BN, time_emb_dim)

        # Project latent to spatial conditioning feature map
        lat = self.latent_proj(z_tgt).reshape(BN, 64, self.patch_size, self.patch_size)

        # Concatenate noisy patch with latent conditioning channel-wise
        x = torch.cat([x_noisy, lat], dim=1)   # (BN, 67, 16, 16)

        # U-Net encoder path
        h1 = self.down1(x,  t_emb, s_emb)      # (BN, 64,  16, 16)
        h2 = self.down2(h1, t_emb, s_emb)      # (BN, 128, 16, 16)
        h  = self.mid(h2,   t_emb, s_emb)      # (BN, 128, 16, 16)

        # U-Net decoder path with skip connections
        h = self.up1(torch.cat([h, h2], dim=1), t_emb, s_emb)  # (BN, 64, 16, 16)
        h = self.up2(torch.cat([h, h1], dim=1), t_emb, s_emb)  # (BN, 64, 16, 16)

        return self.out(h)   # (BN, 3, 16, 16)  — predicted noise


# ─────────────────────────────────────────────────────────────────────────────
# 4. DDPM Noise Scheduler + GuidePaint Samplers
# ─────────────────────────────────────────────────────────────────────────────
class DDPM:
    """
    DDPM noise scheduler implementing:
      A. Standard forward process q(x_t|x_0)                          [Eq.2]
      B. Fast DDIM-style reverse sampling (no guidance)
      C. GuidePaint lossless image-guided reverse sampling             [Algorithm 1]
         with optional interrupted sampling strategy                   [Sec.4]

    All three variants share the same trained denoising network (DiffusionPatchDecoder).
    The guidance is applied ONLY during sampling — training uses the standard
    noise-prediction MSE objective (Eq.6), which remains unchanged.
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T      = num_timesteps
        self.device = device

        betas              = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas             = 1.0 - betas
        self.betas         = betas.to(device)
        self.alphas        = alphas.to(device)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    # ── Forward process ──────────────────────────────────────────────────────
    def add_noise(self, x0, t):
        """
        Forward diffusion: q(x_t|x_0) ~ N(sqrt(abar_t)*x_0, (1-abar_t)*I)  [Eq.2]
        Used during training to produce noisy patches.
        Returns: (x_t, noise)
        """
        a_bar = self.alphas_cumprod[t][:, None, None, None]
        noise = torch.randn_like(x0)
        return (a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise), noise

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _estimate_x0(self, x_t, t_scalar, eps_pred):
        """
        Estimates the clean x_0 from current x_t and predicted noise.  [Eq.8]
        x0_est = (1/sqrt(abar_t)) * (x_t - sqrt(1-abar_t) * eps_theta)
        This is computed at every reverse step in GuidePaint Algorithm 1.
        """
        a_bar = self.alphas_cumprod[int(t_scalar)]
        return (x_t - (1.0 - a_bar).sqrt() * eps_pred) / a_bar.sqrt()

    def _ddpm_mean(self, x_t, t_scalar, eps_pred):
        """
        Standard DDPM reverse mean mu_theta(x_t, t).  [Eq.4]
        mu = (1/sqrt(a_t)) * (x_t - (beta_t/sqrt(1-abar_t)) * eps)
        """
        ts    = int(t_scalar)
        a_t   = self.alphas[ts]
        a_bar = self.alphas_cumprod[ts]
        beta  = self.betas[ts]
        return (1.0 / a_t.sqrt()) * (x_t - (beta / (1.0 - a_bar).sqrt()) * eps_pred)

    # ── Mode A: Fast DDIM sampling (no guidance) ─────────────────────────────
    @torch.no_grad()
    def sample(self, model, z_tgt, s_emb, shape, device, num_steps=50):
        """
        Standard DDIM-style fast sampling. No gradient computation.
        Used for speed-critical inference when GuidePaint guidance is not needed.
        """
        BN = z_tgt.shape[0]
        x  = torch.randn(shape, device=device)

        timesteps = torch.linspace(self.T - 1, 0, num_steps,
                                   dtype=torch.long, device=device)
        for t_val in timesteps:
            t_batch    = t_val.expand(BN)
            pred_noise = model(x, t_batch, z_tgt, s_emb)
            a_bar      = self.alphas_cumprod[t_val]
            x = (x - (1.0 - a_bar).sqrt() * pred_noise) / a_bar.sqrt()
            if t_val > 0:
                # Add small stochastic noise for diversity (DDPM-style)
                x += self.betas[t_val].sqrt() * 0.1 * torch.randn_like(x)
        return x.clamp(-1, 1)

    # ── Mode B: GuidePaint Lossless Image-Guided Sampling (Algorithm 1) ──────
    def sample_guided(self, model, z_tgt, s_emb, shape, device,
                      num_steps=50, gamma=0.1,
                      interrupt_at_t=None, seed=None):
        """
        GuidePaint Lossless Image-Guided Sampling (Algorithm 1, Yu et al. 2025).

        At each reverse step, instead of blindly denoising, we:
          1. Predict noise: eps_theta(x_t, t)
          2. Estimate x_0 from x_t and eps_theta            [GuidePaint Eq.8]
          3. Compute pixel-level similarity F to reference y [GuidePaint Eq.9]
          4. Use grad_{x_t}F to steer the reverse mean       [GuidePaint Eq.7]

        Key advantage over naive DDIM:
          The gradient is computed directly in pixel space from x0_est vs y_ref,
          with NO lossy feature compression — this is the 'lossless' property.
          The intact/known patches (via y_ref from z_tgt) are kept exactly aligned
          while the unknown/damaged patches are freely generated by the model.

        Args:
            model         : DiffusionPatchDecoder (denoising U-Net)
            z_tgt         : (BN, latent_dim)  I-JEPA predictor latents
            s_emb         : (BN, style_dim)   projected style embeddings
            shape         : tuple (BN, 3, 16, 16)
            device        : torch.device
            num_steps     : int    number of reverse diffusion steps
            gamma         : float  GuidePaint guidance weight (paper default: 0.001)
                                   Higher = stronger alignment with reference y.
                                   Recommended: 0.05-0.2 for patch-level system.
            interrupt_at_t: int or None
                                   Interrupted sampling strategy (GuidePaint Sec.4).
                                   If set, stop at this timestep and return x0_est
                                   directly — removes subtle unmarkable degradations.
                                   Example: interrupt_at_t=249 (as used in paper).
            seed          : int or None
                                   Random seed for diverse outputs (GuidePaint Sec.3).
                                   Different seeds -> different but valid restorations.

        Algorithm (GuidePaint Algorithm 1):
          x_T ~ N(0, I)
          for t = T..1:
            [Line 3] x0_est = (1/sqrt(abar_t))*(x_t - sqrt(1-abar_t)*eps_theta)
            [Line 4] grad_F = d/dx_t [-||(1-m)*x0_est - (1-m)*y||^2]
            [Line 5] eps ~ N(0,I)
            [Line 6] mu_theta = DDPM reverse mean
            [Line 7] x_{t-1} = mu_theta + gamma*beta_t*grad_F + sqrt(beta_t)*eps
            -- Interrupted sampling: if t <= interrupt_at_t, return x0_est --
          [Line 9] x = (1-m)*y + m*x0(x_1, t)
          return x

        Returns:
            x : (BN, 3, 16, 16) restored patches in [-1, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)
            if device != torch.device('cpu') and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        BN = z_tgt.shape[0]
        x  = torch.randn(shape, device=device)

        # Pre-compute the lossless pixel-space reference y_ref from z_tgt.
        # This is done ONCE outside the loop (no per-step feature compression).
        # In GuidePaint: y = the known-region pixels of the damaged image.
        # In our system: y = latent-derived pixel estimate from z_tgt.
        # The gradient of F w.r.t. x_t is computed directly in pixel space
        # (no VGG or encoder intermediaries) — this is the 'lossless' guarantee.
        with torch.no_grad():
            y_ref = model.get_pixel_reference(z_tgt)   # (BN, 3, 16, 16)

        timesteps = torch.linspace(self.T - 1, 0, num_steps,
                                   dtype=torch.long, device=device)

        x0_est_last = None   # track final x0_est for lossless line-9 step

        for t_val in timesteps:
            ts = int(t_val.item())
            t_batch = t_val.expand(BN)

            # ── Gradient-enable x_t for GuidePaint gradient computation
            x = x.detach().requires_grad_(True)

            with torch.enable_grad():
                # [Line 3, Algo.1] Predict noise: eps_theta(x_t, t)
                eps_pred = model(x, t_batch, z_tgt.detach(), s_emb.detach())

                # [Line 3, Algo.1 / Eq.8] Estimate x_0 from current x_t
                # x0_est = (1/sqrt(abar_t)) * (x_t - sqrt(1-abar_t) * eps_theta)
                a_bar  = self.alphas_cumprod[ts]
                x0_est = (x - (1.0 - a_bar).sqrt() * eps_pred) / a_bar.sqrt()

                # ── Interrupted sampling strategy [GuidePaint Sec.4]
                # At intermediate step t_stop, return x0_est directly.
                # The outline/structure is recovered first (large t), and local
                # degradation details are discarded by stopping before full recovery.
                # Example: stop at t=249 to remove unmarkable dense degradations.
                if interrupt_at_t is not None and ts <= interrupt_at_t:
                    return x0_est.detach().clamp(-1, 1)

                # [Line 4, Algo.1 / Eq.9] Compute similarity F and gradient
                # F(x_t, y, t) = -||(1-m) * x0_est - (1-m) * y||^2
                # In our system m=1 everywhere per patch (all target pixels unknown)
                # => (1-m) = 1 => F = -||x0_est - y_ref||^2
                # Gradient: d/dx_t F steers x toward higher similarity with y_ref.
                F_val = -((x0_est - y_ref.detach()) ** 2).sum()
                grad_F = torch.autograd.grad(F_val, x, create_graph=False)[0]

            # Detach all tensors after gradient computation
            x        = x.detach()
            eps_pred = eps_pred.detach()
            grad_F   = grad_F.detach()
            x0_est_last = x0_est.detach()

            with torch.no_grad():
                # [Line 6, Algo.1 / Eq.4] Standard DDPM reverse mean mu_theta
                beta_t = self.betas[ts]
                a_t    = self.alphas[ts]
                a_bar  = self.alphas_cumprod[ts]
                mu_theta = (1.0 / a_t.sqrt()) * (
                    x - (beta_t / (1.0 - a_bar).sqrt()) * eps_pred
                )

                # [Line 7, Algo.1 / Eq.7] Gradient-corrected reverse step:
                # x_{t-1} = mu_theta + gamma * Sigma_theta * grad_F + Sigma_theta * eps
                # Sigma_theta = beta_t (fixed variance schedule)
                # The gamma term pulls x_{t-1} toward higher F values (lower pixel MSE).
                sigma = beta_t.sqrt()
                noise = torch.randn_like(x) if ts > 0 else torch.zeros_like(x)
                x = mu_theta + gamma * beta_t * grad_F + sigma * noise

        # [Line 9, Algo.1] Lossless final step: x = (1-m)*y + m*x0(x_1, t)
        # Since m=1 for all target patches (fully unknown), this simplifies to:
        # x = x0_est_last (the estimated clean patch)
        # This ensures the final output is in pixel space, not accumulating noise.
        if x0_est_last is not None:
            return x0_est_last.clamp(-1, 1)
        return x.clamp(-1, 1)

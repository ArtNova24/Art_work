"""
Historic Image Restoration Phase 3 — Style-Conditioned I-JEPA Training Suite.
Implements:
  1. Loss orchestrations (Latent L2 + Diffusion noise-prediction joint optimization).
  2. Momentum-based target encoder tracking (EMA).
  3. Patch folding/unfolding diagnostic saves.
  4. Side-by-side restoration visualization generation.
All prints and comments are kept strictly in ASCII.
"""
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision.utils import save_image

# Central configs
from phase3.config import (
    DEVICE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    EMA_MOMENTUM_BASE, EMA_MOMENTUM_MAX, LATENT_LOSS_WEIGHT, PIXEL_LOSS_WEIGHT,
    DIFFUSION_LOSS_WEIGHT, DECODER_TYPE,
    DIFFUSION_TIMESTEPS, DIFFUSION_BETA_START, DIFFUSION_BETA_END, DDIM_SAMPLING_STEPS,
    GUIDEPAINT_GAMMA, GUIDEPAINT_SAMPLING_STEPS, GUIDEPAINT_INTERRUPT_T,
    PROJECTOR_PATH, ENCODER_PATH, PREDICTOR_PATH,
    DECODER_PATH, DECODER_DIFFUSION_PATH, TARGET_ENCODER_PATH,
    RECON_DIR, VIT_EMBED_DIM
)
from phase3.masking import BlockMaskGenerator
from phase3.models import (
    StyleProjector, ViTContextEncoder, StyleConditionedPredictor,
    PixelDecoder, DiffusionPixelDecoder, get_pixel_decoder
)
from phase3.dataset import StyleJEPAImageDataset

# -- Perceptual Loss Module
class PerceptualStyleLoss(nn.Module):
    """
    Computes VGG-based perceptual loss + Gram matrix style loss.
    Uses VGG-16 features from multiple layers.
    """
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features
        # Layers: relu1_2, relu2_2, relu3_3, relu4_3
        self.slice1 = nn.Sequential(*list(vgg)[:4]).to(device).eval()
        self.slice2 = nn.Sequential(*list(vgg)[4:9]).to(device).eval()
        self.slice3 = nn.Sequential(*list(vgg)[9:16]).to(device).eval()
        self.slice4 = nn.Sequential(*list(vgg)[16:23]).to(device).eval()

        for param in self.parameters():
            param.requires_grad = False  # Frozen VGG

        # ImageNet normalization for VGG input
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def normalize_for_vgg(self, x):
        # x is in [-1, 1], convert to [0, 1] then ImageNet normalize
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def gram_matrix(self, feat):
        B, C, H, W = feat.shape
        feat_flat = feat.reshape(B, C, H * W)
        gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2))
        return gram / (C * H * W)

    def forward(self, pred, target):
        """
        pred, target: (B, 3, H, W) in [-1, 1]
        Returns: perceptual_loss, style_loss (both scalars)
        """
        pred_n   = self.normalize_for_vgg(pred)
        target_n = self.normalize_for_vgg(target)

        # Extract VGG features
        p1 = self.slice1(pred_n);  t1 = self.slice1(target_n)
        p2 = self.slice2(p1);      t2 = self.slice2(t1)
        p3 = self.slice3(p2);      t3 = self.slice3(t2)
        p4 = self.slice4(p3);      t4 = self.slice4(t3)

        # Perceptual loss: feature L2 distance
        perc_loss = (F.mse_loss(p1, t1) + F.mse_loss(p2, t2) +
                     F.mse_loss(p3, t3) + F.mse_loss(p4, t4))

        # Style loss: Gram matrix distance
        style_loss = (F.mse_loss(self.gram_matrix(p1), self.gram_matrix(t1)) +
                      F.mse_loss(self.gram_matrix(p2), self.gram_matrix(t2)) +
                      F.mse_loss(self.gram_matrix(p3), self.gram_matrix(t3)) +
                      F.mse_loss(self.gram_matrix(p4), self.gram_matrix(t4)))

        return perc_loss, style_loss


def extract_patches(img_tensor, patch_size=16):
    """
    Slices a batch of images of shape (B, 3, 224, 224)
    into non-overlapping patches of shape (B, 196, 3, 16, 16).
    """
    B, C, H, W = img_tensor.shape
    num_patches_side = H // patch_size
    
    # Extract patches using unfold
    patches = img_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Shape: (B, C, 14, 14, 16, 16)
    patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(1, 2)
    # Shape: (B, 196, C, 16, 16)
    return patches

def reconstruct_image(patches, patch_size=16):
    """
    Folds a batch of patches of shape (B, 196, 3, 16, 16)
    back into a batch of full images of shape (B, 3, 224, 224).
    """
    B, N, C, P, _ = patches.shape
    num_patches_side = int(math.sqrt(N))  # 14
    
    recon_grid = patches.reshape(B, num_patches_side, num_patches_side, C, P, P)
    recon_grid = recon_grid.permute(0, 3, 1, 4, 2, 5)
    # Shape: (B, C, 14, 16, 14, 16)
    recon_img = recon_grid.reshape(B, C, num_patches_side * P, num_patches_side * P)
    # Shape: (B, 3, 224, 224)
    return recon_img

def run_training(dry_run=False, num_epochs=None, decoder_type=None):
    """
    Executes the self-supervised style-conditioned I-JEPA training run.
    decoder_type: 'diffusion' | 'mlp' | None (uses DECODER_TYPE from config)
    """
    # Resolve decoder type
    _decoder_type = decoder_type if decoder_type is not None else DECODER_TYPE
    _use_diffusion = (_decoder_type == 'diffusion')
    t_start = time.time()
    
    # 1. Datasets & Dataloaders
    print("  Loading datasets...", flush=True)
    train_ds = StyleJEPAImageDataset(split='train')
    val_ds = StyleJEPAImageDataset(split='val')
    
    if dry_run:
        # Mini loader for verification dry-runs
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
        epochs_to_run = 2
        print(f"    [DRY RUN] Configured 2 epochs over {len(train_ds)} samples.", flush=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        epochs_to_run = num_epochs if num_epochs is not None else EPOCHS
        print(f"    Configured {epochs_to_run} epochs over {len(train_ds)} training samples.", flush=True)

    # 2. Instantiate Models
    print("  Initializing Style-Conditioned I-JEPA components...")
    projector = StyleProjector().to(DEVICE)
    context_encoder = ViTContextEncoder().to(DEVICE)
    predictor = StyleConditionedPredictor().to(DEVICE)
    pixel_decoder = get_pixel_decoder(
        decoder_type=_decoder_type,
        latent_dim=VIT_EMBED_DIM,
        style_dim=256,
        num_timesteps=DIFFUSION_TIMESTEPS,
        beta_start=DIFFUSION_BETA_START,
        beta_end=DIFFUSION_BETA_END,
        device=str(DEVICE),
    ).to(DEVICE)
    print(f"    Pixel decoder: {_decoder_type.upper()} ({type(pixel_decoder).__name__})", flush=True)
    
    # Initialize EMA Target Encoder
    target_encoder = ViTContextEncoder().to(DEVICE)
    target_encoder.load_state_dict(context_encoder.state_dict())
    
    # Freeze Target Encoder weights (updated only via momentum EMA)
    for param in target_encoder.parameters():
        param.requires_grad = False
        
    print(f"    Models loaded on device: {DEVICE}", flush=True)

    # Initialize perceptual style loss helper (used for MLP path only)
    perceptual_loss_fn = PerceptualStyleLoss(device=DEVICE) if not _use_diffusion else None

    # 3. Optimizer & Schedulers
    # Differential learning rates for pretrained vs new components
    vit_params = list(context_encoder.vit.parameters())
    vit_param_ids = set(map(id, vit_params))
    other_enc_params = [p for p in context_encoder.parameters() if id(p) not in vit_param_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': vit_params, 'lr': LR * 0.1},
        {'params': other_enc_params, 'lr': LR},
        {'params': projector.parameters(), 'lr': LR},
        {'params': predictor.parameters(), 'lr': LR},
        {'params': pixel_decoder.parameters(), 'lr': LR},
    ], lr=LR, weight_decay=WEIGHT_DECAY)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_run)
    mask_gen = BlockMaskGenerator(grid_size=14, target_masked=98)
    
    best_val_loss = float('inf')
    
    # 4. Master Training Loop
    print("\n  Starting I-JEPA Training Loop...", flush=True)
    for epoch in range(epochs_to_run):
        t_epoch_start = time.time()
        
        # Base training phase
        projector.train()
        context_encoder.train()
        predictor.train()
        pixel_decoder.train()
        target_encoder.eval()  # EMA target remains in evaluation mode
        
        train_latent_loss_sum  = 0.0
        train_diffusion_loss_sum = 0.0   # diffusion path
        train_pixel_loss_sum   = 0.0    # mlp path
        train_loss_sum         = 0.0
        
        for batch_idx, (imgs, style_vecs, _) in enumerate(train_loader):
            B = imgs.shape[0]
            
            # Generate static block masks for this batch: exactly 50% masked (98 patches)
            masks = mask_gen.collate_masks(B).to(DEVICE) # (B, 196)
            imgs = imgs.to(DEVICE)
            style_vecs = style_vecs.to(DEVICE)
            
            # Vectorized slice representations
            img_patches = extract_patches(imgs)  # (B, 196, 3, 16, 16)
            
            # Reconstruct exact sorted indexing matching BlockMaskGenerator
            sorted_indices = torch.argsort(masks.to(torch.int32), dim=1)
            target_indices = sorted_indices[:, 98:] # (B, 98) Target (True / Masked)
            
            # --- 1. Forward pass Target Encoder (Stop-Grad EMA)
            with torch.no_grad():
                z_target_all = target_encoder(imgs, get_all=True) # (B, 196, VIT_EMBED_DIM)
                # Gather ground-truth target patch representations
                z_target_gt = torch.gather(
                    z_target_all, 1, 
                    target_indices.unsqueeze(-1).expand(-1, -1, VIT_EMBED_DIM)
                ) # (B, 98, VIT_EMBED_DIM)
            
            # --- 2. Forward pass Style Projector
            s_emb = projector(style_vecs) # (B, STYLE_EMBED_DIM)
            
            # --- 3. Forward pass Context Encoder (Only unmasked patches)
            z_ctx, _ = context_encoder(imgs, mask=masks) # (B, 98, VIT_EMBED_DIM)
            
            # --- 4. Forward pass Predictor (Predict Target latents conditioned on style)
            z_tgt_pred, _ = predictor(z_ctx, s_emb, mask=masks) # (B, 98, VIT_EMBED_DIM)
            
            # --- 5. Gather ground-truth pixel patches (always needed)
            pixel_gt = torch.gather(
                img_patches, 1,
                target_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 16, 16)
            ) # (B, 98, 3, 16, 16)

            # --- 6. Loss calculation (decoder-type branching)
            latent_loss = F.mse_loss(z_tgt_pred, z_target_gt.detach())

            if _use_diffusion:
                # DIFFUSION PATH: compute DDPM noise-prediction loss
                # compute_loss() internally samples a random timestep t
                diffusion_loss = pixel_decoder.compute_loss(z_tgt_pred, pixel_gt, s_emb)
                loss = LATENT_LOSS_WEIGHT * latent_loss + DIFFUSION_LOSS_WEIGHT * diffusion_loss
                pixel_loss = diffusion_loss  # for unified logging
            else:
                # MLP PATH: simple pixel reconstruction + perceptual loss
                pixel_pred = pixel_decoder(z_tgt_pred)  # (B, 98, 3, 16, 16)
                recon_patches = img_patches.clone()
                recon_patches[torch.arange(B).unsqueeze(-1), target_indices] = pixel_pred
                pred_img = reconstruct_image(recon_patches)  # (B, 3, 224, 224)
                mse_loss = F.mse_loss(pixel_pred, pixel_gt)
                perc_loss, style_gram_loss = perceptual_loss_fn(pred_img, imgs)
                pixel_loss = 0.5 * mse_loss + 0.3 * perc_loss + 0.2 * style_gram_loss
                loss = LATENT_LOSS_WEIGHT * latent_loss + PIXEL_LOSS_WEIGHT * pixel_loss
            
            # --- 7. Backprop & step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- 8. EMA update for Target Encoder weights
            # Compute epoch-scaling cosine momentum value
            epoch_ratio = epoch / epochs_to_run
            ema_momentum = EMA_MOMENTUM_MAX - (EMA_MOMENTUM_MAX - EMA_MOMENTUM_BASE) * (math.cos(math.pi * epoch_ratio) + 1.0) / 2.0
            
            with torch.no_grad():
                for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
                    param_k.data = ema_momentum * param_k.data + (1.0 - ema_momentum) * param_q.data
            
            # Record losses
            train_latent_loss_sum  += latent_loss.item() * B
            train_pixel_loss_sum   += pixel_loss.item() * B
            if _use_diffusion:
                train_diffusion_loss_sum += pixel_loss.item() * B
            train_loss_sum += loss.item() * B
            
            if dry_run and batch_idx >= 2:
                break
                
        # Epoch metrics
        epoch_samples = len(train_loader.dataset) if not dry_run else (3 * 8)
        epoch_latent_loss = train_latent_loss_sum / epoch_samples
        epoch_pixel_loss = train_pixel_loss_sum / epoch_samples
        epoch_loss = train_loss_sum / epoch_samples
        
        lr_scheduler.step()
        
        # --- Validation & Visual diagnostics
        projector.eval()
        context_encoder.eval()
        predictor.eval()
        pixel_decoder.eval()
        
        val_loss_sum = 0.0
        val_samples = 0
        
        # Visual reconstruction pipeline: render first validation batch
        vis_saved = False
        
        with torch.no_grad():
            for val_imgs, val_style_vecs, _ in val_loader:
                B_val = val_imgs.shape[0]
                val_masks = mask_gen.collate_masks(B_val).to(DEVICE)
                val_imgs = val_imgs.to(DEVICE)
                val_style_vecs = val_style_vecs.to(DEVICE)
                
                # Unfold
                val_img_patches = extract_patches(val_imgs)
                val_sorted = torch.argsort(val_masks.to(torch.int32), dim=1)
                val_target_idx = val_sorted[:, 98:]
                val_ctx_idx = val_sorted[:, :98]
                
                # Forward
                val_z_tgt_all = target_encoder(val_imgs, get_all=True)
                val_z_tgt_gt = torch.gather(val_z_tgt_all, 1, val_target_idx.unsqueeze(-1).expand(-1, -1, VIT_EMBED_DIM))
                
                val_s_emb = projector(val_style_vecs)
                val_z_ctx, _ = context_encoder(val_imgs, mask=val_masks)
                val_z_tgt_pred, _ = predictor(val_z_ctx, val_s_emb, mask=val_masks)

                val_pixel_gt = torch.gather(
                    val_img_patches, 1,
                    val_target_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 16, 16)
                )

                val_latent_l = F.mse_loss(val_z_tgt_pred, val_z_tgt_gt)

                if _use_diffusion:
                    # GuidePaint guided sampling for pixel reconstruction visualization
                    # Uses sample_guided() to match inference behaviour exactly.
                    # compute_loss() gives the val noise-prediction MSE (no grad needed).
                    val_pixel_pred = pixel_decoder.sample_with_mode(
                        val_z_tgt_pred, val_s_emb,
                        ddim_steps=GUIDEPAINT_SAMPLING_STEPS,
                        mode='guided',
                        gamma=GUIDEPAINT_GAMMA,
                        interrupt_at_t=GUIDEPAINT_INTERRUPT_T,
                    )
                    val_diff_l = pixel_decoder.compute_loss(val_z_tgt_pred, val_pixel_gt, val_s_emb)
                    val_pixel_l = val_diff_l
                    val_loss = LATENT_LOSS_WEIGHT * val_latent_l + DIFFUSION_LOSS_WEIGHT * val_diff_l
                else:
                    val_pixel_pred = pixel_decoder(val_z_tgt_pred)
                    val_recon_patches = val_img_patches.clone()
                    val_recon_patches[torch.arange(B_val).unsqueeze(-1), val_target_idx] = val_pixel_pred
                    val_pred_img = reconstruct_image(val_recon_patches)
                    val_mse_l = F.mse_loss(val_pixel_pred, val_pixel_gt)
                    val_perc_l, val_style_l = perceptual_loss_fn(val_pred_img, val_imgs)
                    val_pixel_l = 0.5 * val_mse_l + 0.3 * val_perc_l + 0.2 * val_style_l
                    val_loss = LATENT_LOSS_WEIGHT * val_latent_l + PIXEL_LOSS_WEIGHT * val_pixel_l
                
                val_loss_sum += val_loss.item() * B_val
                val_samples += B_val
                
                # Generate intermediate visual reconstruction cards (Only for first batch of the epoch)
                if not vis_saved:
                    vis_saved = True
                    # Reconstruct and save first 3 samples of validation batch side-by-side
                    vis_panels = []
                    num_vis = min(3, B_val)
                    
                    for i in range(num_vis):
                        # 1. Original image (Un-normalized to [0,1])
                        orig_img = val_imgs[i] * 0.5 + 0.5
                        
                        # 2. Corrupted Image (Set masked target patches to dark gray)
                        corr_patches = val_img_patches[i].clone()
                        corr_patches[val_target_idx[i]] = -0.8  # dark gray
                        corr_img = reconstruct_image(corr_patches.unsqueeze(0))[0] * 0.5 + 0.5
                        
                        # 3. Reconstructed Image (Set target patches to decoded prediction)
                        recon_patches = val_img_patches[i].clone()
                        recon_patches[val_target_idx[i]] = val_pixel_pred[i]
                        recon_img = reconstruct_image(recon_patches.unsqueeze(0))[0] * 0.5 + 0.5
                        
                        # Strip: Original | Corrupted | Reconstructed
                        strip = torch.cat([orig_img, corr_img, recon_img], dim=2)  # Concatenate horizontally
                        vis_panels.append(strip)
                        
                    # Stack panels vertically and save
                    stacked_panel = torch.cat(vis_panels, dim=1)
                    vis_path = os.path.join(RECON_DIR, f"epoch_{epoch+1}.png")
                    save_image(stacked_panel, vis_path)
                    
                if dry_run:
                    break
                    
        epoch_val_loss = val_loss_sum / val_samples if val_samples > 0 else 0.0
        
        # Log epoch summary
        print(f"    Epoch {epoch+1:02d}/{epochs_to_run:02d} | "
              f"Train Loss: {epoch_loss:.5f} (Latent: {epoch_latent_loss:.5f}, Pixel: {epoch_pixel_loss:.5f}) | "
              f"Val Loss: {epoch_val_loss:.5f} | "
              f"LR: {lr_scheduler.get_last_lr()[0]:.2e} | "
              f"EMA: {ema_momentum:.4f} | "
              f"Time: {time.time()-t_epoch_start:.1f}s", flush=True)
              
        # Save checkpoints if Val loss improves
        if not dry_run and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            decoder_ckpt_path = DECODER_DIFFUSION_PATH if _use_diffusion else DECODER_PATH
            torch.save(projector.state_dict(), PROJECTOR_PATH)
            torch.save(context_encoder.state_dict(), ENCODER_PATH)
            torch.save(predictor.state_dict(), PREDICTOR_PATH)
            torch.save(pixel_decoder.state_dict(), decoder_ckpt_path)
            torch.save(target_encoder.state_dict(), TARGET_ENCODER_PATH)
            print(f"      [CHECKPOINT] Saved best models (decoder -> {decoder_ckpt_path}).", flush=True)

    print(f"\n  Style-Conditioned I-JEPA Model Training successfully concluded in {(time.time()-t_start)/60.0:.1f} minutes.")
    if not dry_run:
        decoder_ckpt_path = DECODER_DIFFUSION_PATH if _use_diffusion else DECODER_PATH
        print(f"    Best Validation Loss: {best_val_loss:.5f}")
        print(f"    Decoder type: {_decoder_type.upper()}")
        print(f"    Saved models in features/:")
        print(f"      - {PROJECTOR_PATH}")
        print(f"      - {ENCODER_PATH}")
        print(f"      - {PREDICTOR_PATH}")
        print(f"      - {decoder_ckpt_path}")
        print(f"      - {TARGET_ENCODER_PATH}")

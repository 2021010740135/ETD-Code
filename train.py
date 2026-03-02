# train.py
import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from data_loader import get_dataloaders
from model import PhysicsAwareUNet1D, GaussianDiffusion1D
import torch.nn.functional as F

# 【极客优化】：开启 TF32 矩阵乘法加速 (Ampere及以上架构 GPU 速度提升 30%+)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ==============================================================================
# 1. 训练配置 (ICPC 竞赛级调参)
# ==============================================================================
CONFIG = {
    'epochs': 100,
    'max_lr': 3e-4,          
    'weight_decay': 1e-2,    
    'grad_clip': 1.0,
    'patience': 20,          
    'ema_decay': 0.9999,     

    'lambda_phys': 0.5,
    'cond_drop_prob': 0.1,

    'save_dir': './checkpoints',
    'log_dir': './logs',
    'experiment_name': 'phase3_sota_diffusion',

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 2026
}

# ==============================================================================
# 2. 稳健的 EMA 管理器 (防 OOM 崩溃版)
# ==============================================================================
class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.data))

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ==============================================================================
# 3. 核心计算模块
# ==============================================================================
def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'training.log')
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calc_mask_aware_loss(noise_pred, noise_real, mask):
    """【数学防线】：只在 Mask == 1 的观测点上计算梯度"""
    raw_mse = F.mse_loss(noise_pred, noise_real, reduction='none')
    masked_mse = (raw_mse * mask).sum() / (mask.sum() + 1e-8)
    return masked_mse

def save_checkpoint_diffusion(diffusion_model, optimizer, epoch, loss, path, extra=None, is_ema=False):
    """【修复】：保存完整的 diffusion_model，保住 null_phys_emb 参数"""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': diffusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'is_ema': is_ema
    }
    if extra: ckpt.update(extra)
    torch.save(ckpt, path)

# ==============================================================================
# 4. 训练与验证流 (极速异步版)
# ==============================================================================
def train_one_epoch(diffusion, loader, optimizer, scheduler, grad_scaler, device, epoch, logger, ema):
    diffusion.train()
    phys_criterion = nn.MSELoss()
    total_loss, total_diff_loss, total_phys_loss = 0.0, 0.0, 0.0
    use_amp = (device.type == "cuda")
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{CONFIG['epochs']:03d} [Train]", leave=False)

    for residuals, patched, masks, phys_feats, labels, cons_nos in pbar:
        residuals = residuals.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        phys_feats = phys_feats.to(device, non_blocking=True)

        t = torch.randint(0, diffusion.timesteps, (residuals.shape[0],), device=device).long()
        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            noise_pred, noise_real, pred_phys0 = diffusion(residuals, masks, t, phys_feats)

            loss_diff = calc_mask_aware_loss(noise_pred, noise_real, masks)
            real_phys0 = phys_feats[:, 0].unsqueeze(1) 
            loss_phys = phys_criterion(pred_phys0, real_phys0)

            loss = loss_diff + CONFIG['lambda_phys'] * loss_phys

        grad_scaler.scale(loss).backward()
        
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), CONFIG['grad_clip'])
        
        grad_scaler.step(optimizer)
        grad_scaler.update()
        
        scheduler.step()
        ema.update()

        total_loss += float(loss.item())
        total_diff_loss += float(loss_diff.item())
        total_phys_loss += float(loss_phys.item())

        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})

    avg_loss = total_loss / max(1, len(loader))
    logger.info(
        f"[Train] Epoch {epoch:03d}: Total={avg_loss:.5f} | "
        f"Diff={total_diff_loss / max(1, len(loader)):.5f} | "
        f"Phys={total_phys_loss / max(1, len(loader)):.5f}"
    )
    return avg_loss

@torch.no_grad()
def validate(diffusion, loader, device, epoch, logger, ema=None):
    if loader is None: return float('inf')
    
    # 强制固定验证集上的时间步加噪，保证 Loss 具有绝对可比性
    torch.manual_seed(CONFIG['seed'])

    if ema is not None: ema.apply_shadow()
    diffusion.eval()
    
    phys_criterion = nn.MSELoss()
    total_loss = 0.0

    try: 
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{CONFIG['epochs']:03d} [Valid]", leave=False)
        for residuals, patched, masks, phys_feats, labels, cons_nos in pbar:
            residuals = residuals.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            phys_feats = phys_feats.to(device, non_blocking=True)

            t = torch.randint(0, diffusion.timesteps, (residuals.shape[0],), device=device).long()
            
            original_drop_prob = diffusion.cond_drop_prob
            diffusion.cond_drop_prob = 0.0 
            
            noise_pred, noise_real, pred_phys0 = diffusion(residuals, masks, t, phys_feats)
            diffusion.cond_drop_prob = original_drop_prob

            loss_diff = calc_mask_aware_loss(noise_pred, noise_real, masks)
            real_phys0 = phys_feats[:, 0].unsqueeze(1)
            loss_phys = phys_criterion(pred_phys0, real_phys0)

            loss = loss_diff + CONFIG['lambda_phys'] * loss_phys
            total_loss += float(loss.item())

    finally:
        if ema is not None: ema.restore()
        
    avg_loss = total_loss / max(1, len(loader))
    # 恢复系统的随机性
    torch.manual_seed(int(time.time()))
    logger.info(f"[Val]   Epoch {epoch:03d}: Loss={avg_loss:.5f} {'(EMA)' if ema else ''}")
    return avg_loss

# ==============================================================================
# 5. 主程序入口
# ==============================================================================
def main():
    set_seed(CONFIG['seed'])
    logger = setup_logger(CONFIG['log_dir'])
    device = torch.device(CONFIG['device'])
    logger.info(f"🚀 Initializing High-Performance Training Pipeline on {device}")

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    summary_txt = "epoch_results.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("🚀 SGCC Phase 3: Next-Gen U-Net Training Summary\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Status':<20}\n")
        f.write("-" * 65 + "\n")

    # 【对齐管线】：接入双轨数据，严格提取纯净的扩散专属 loader
    diff_train_dl, diff_val_dl, clf_train_dl, test_dl = get_dataloaders()

    unet = PhysicsAwareUNet1D(in_channels=2, out_channels=1, base_dim=64, phys_dim=6).to(device)
    diffusion = GaussianDiffusion1D(
        unet, seq_length=256, timesteps=1000, 
        cond_drop_prob=CONFIG['cond_drop_prob'], phys_dim=6
    ).to(device)

    ema = EMA(diffusion, CONFIG['ema_decay'])
    optimizer = optim.AdamW(diffusion.parameters(), lr=CONFIG['max_lr'], weight_decay=CONFIG['weight_decay'])

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG['max_lr'], 
        epochs=CONFIG['epochs'], 
        steps_per_epoch=len(diff_train_dl),
        pct_start=0.1,         
        anneal_strategy='cos', 
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    grad_scaler = GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("🔥 Engine ignited. Starting AdaLN-Zero + Mask-Aware Diffusion Loop (Pure Normal Track)...")
    start_time = time.time()

    for epoch in range(1, CONFIG['epochs'] + 1):
        # 【核心修正】：喂入 diff_train_dl 和 diff_val_dl
        train_loss = train_one_epoch(diffusion, diff_train_dl, optimizer, scheduler, grad_scaler, device, epoch, logger, ema)
        val_loss = validate(diffusion, diff_val_dl, device, epoch, logger, ema=ema)

        status_str = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            ema.apply_shadow()
            save_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
            # 保存整个 diffusion 容器
            save_checkpoint_diffusion(diffusion, optimizer, epoch, val_loss, save_path, extra={'experiment_name': CONFIG['experiment_name']}, is_ema=True)
            ema.restore()
            
            logger.info(f"✨ New SOTA Checkpoint! (Val Loss: {val_loss:.5f})")
            status_str = "✨ BEST MODEL"
        else:
            patience_counter += 1
            logger.info(f"⏳ Early Stopping Counter: {patience_counter}/{CONFIG['patience']}")
            status_str = f"⏳ Patience {patience_counter}/{CONFIG['patience']}"

        with open(summary_txt, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch:<2} | {train_loss:<12.5f} | {val_loss:<12.5f} | {status_str}\n")

        if patience_counter >= CONFIG['patience']:
            logger.info(f"🛑 Training halted due to Early Stopping at Epoch {epoch}.")
            with open(summary_txt, "a", encoding="utf-8") as f:
                f.write(f"\n🛑 Early stopping triggered at Epoch {epoch}!\n")
            break

    total_time = time.time() - start_time
    logger.info(f"✅ Mission Accomplished in {total_time / 3600:.2f} hours.")
    logger.info(f"🏆 Ultimate Validation Loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    main()
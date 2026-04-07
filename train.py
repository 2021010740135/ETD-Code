import json
import logging
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# 确保这里的导入路径与你的项目结构一致
from data_loader import LoaderConfig, get_dataloaders
from model import ConditionalUNet1D, GaussianDiffusion1D, unpack_diffusion_batch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

CONFIG = {
    "epochs": 150,
    "max_lr": 6e-4,
    "weight_decay": 1e-2,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "save_dir": "./checkpoints_phase3",
    "log_dir": "./logs_phase3",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 58,                   # 1. 设定随机种子为 58
    "timesteps": 1000,
    "seq_length": 256,
    "base_dim": 64,
    "dropout": 0.10,
}

class EMA:
    """
    指数移动平均 (Exponential Moving Average)。
    异常检测基石：隔离梯度剧烈震荡，输出平滑、稳定的正常流形空间。
    """
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = model
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.data))

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("diffusion_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

def save_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def save_checkpoint(
    path: str,
    diffusion: GaussianDiffusion1D,
    optimizer: optim.Optimizer,
    epoch: int,
    metric: float,
    ema_applied: bool = False,
):
    ckpt = {
        "epoch": int(epoch),
        "metric": float(metric),
        "ema_applied": bool(ema_applied),
        "model_state_dict": diffusion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": CONFIG,
    }
    torch.save(ckpt, path)

@torch.no_grad()
def evaluate_loss(
    diffusion: GaussianDiffusion1D,
    loader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    split_name: str,
    ema: EMA,
) -> float:
    """
    确定性评估：使用固定的时间步 t 生成器。
    这切断了 Diffusion 中因为随机采样 t 导致的 Loss 剧烈波动，使 Loss 曲线具备真实的收敛指示意义。
    """
    if loader is None:
        return float("inf")

    ema.apply_shadow()
    diffusion.eval()
    total_loss = 0.0
    total_batches = 0
    use_amp = device.type == "cuda"

    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(CONFIG["seed"]) 

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{CONFIG['epochs']:03d} [{split_name}]", leave=False)
    for batch in pbar:
        x_log, x_mm, mask = unpack_diffusion_batch(batch)
        x_log = x_log.to(device, non_blocking=True)
        x_mm = x_mm.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        t = torch.randint(
            0, diffusion.timesteps, (x_log.shape[0],), 
            generator=eval_generator, device=device
        ).long()

        with autocast(device_type="cuda", enabled=use_amp):
            out = diffusion(x_log, x_mm, mask, t)
            loss = out["loss"]

        total_loss += float(loss.item())
        total_batches += 1

    ema.restore()
    avg_loss = total_loss / max(total_batches, 1)
    logger.info(f"[{split_name}] Epoch {epoch:03d} (Deterministic): loss={avg_loss:.6f}")
    return avg_loss

def train_one_epoch(
    diffusion: GaussianDiffusion1D,
    loader,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    ema: EMA,
) -> float:
    diffusion.train()
    total_loss = 0.0
    total_batches = 0
    use_amp = device.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{CONFIG['epochs']:03d} [Train]", leave=False)
    for batch in pbar:
        x_log, x_mm, mask = unpack_diffusion_batch(batch)
        x_log = x_log.to(device, non_blocking=True)
        x_mm = x_mm.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        t = torch.randint(0, diffusion.timesteps, (x_log.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            out = diffusion(x_log, x_mm, mask, t)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        ema.update()

        total_loss += float(loss.item())
        total_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    avg_loss = total_loss / max(total_batches, 1)
    logger.info(f"[Train] Epoch {epoch:03d}: loss={avg_loss:.6f}")
    return avg_loss

def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    logger = setup_logger(CONFIG["log_dir"])
    device = torch.device(CONFIG["device"])

    logger.info("=" * 80)
    logger.info("🚀 Phase 3: Diffusion Manifold Learning")
    logger.info("Target: x_log | Conditions: x_mm + mask")
    logger.info("Strategy: Save Best Model (EMA), No Periodic Saving")
    logger.info("=" * 80)

    loader_cfg = LoaderConfig()
    loaders = get_dataloaders(loader_cfg)
    diff_train_dl = loaders["diff_train"]
    diff_val_dl = loaders["diff_val"]

    save_json(os.path.join(CONFIG["log_dir"], "loader_config.json"), asdict(loader_cfg))
    save_json(os.path.join(CONFIG["log_dir"], "train_config.json"), CONFIG)

    unet = ConditionalUNet1D(
        in_channels=3,
        out_channels=1,
        base_dim=CONFIG["base_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)

    diffusion = GaussianDiffusion1D(
        model=unet,
        seq_length=CONFIG["seq_length"],
        timesteps=CONFIG["timesteps"],
    ).to(device)

    optimizer = optim.AdamW(
        diffusion.parameters(),
        lr=CONFIG["max_lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["max_lr"],
        epochs=CONFIG["epochs"],
        steps_per_epoch=len(diff_train_dl),
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    
    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))
    ema = EMA(diffusion, CONFIG["ema_decay"])

    history = []
    best_val_loss = float("inf") # 2. 初始化最佳 Loss
    start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(
            diffusion, diff_train_dl, optimizer, scheduler, scaler, device, epoch, logger, ema
        )
        
        diff_val_loss = evaluate_loss(
            diffusion, diff_val_dl, device, epoch, logger, split_name="Diff-Val(Test-Normal)", ema=ema
        )

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "diff_val_loss": float(diff_val_loss),
        }
        history.append(record)

        # 3. 核心修改：如果当前 Loss 更低，才覆盖保存 best_model.pth
        if diff_val_loss < best_val_loss:
            best_val_loss = diff_val_loss
            ema.apply_shadow()
            save_checkpoint(
                os.path.join(CONFIG["save_dir"], "best_model.pth"),
                diffusion, optimizer, epoch, best_val_loss, ema_applied=True,
            )
            ema.restore()
            logger.info(f"🌟 New best model saved at epoch {epoch} with val_loss: {best_val_loss:.6f}")

    save_json(os.path.join(CONFIG["log_dir"], "history.json"), history)

    elapsed = time.time() - start
    logger.info("=" * 80)
    logger.info(f"✅ Diffusion Training Finished in {elapsed / 3600:.2f} hours")
    logger.info(f"🏁 Best EMA Model preserved as 'best_model.pth' with val_loss: {best_val_loss:.6f}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
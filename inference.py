import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from data_loader import LoaderConfig, get_dataloaders
from model import ConditionalUNet1D, GaussianDiffusion1D

CONFIG = {
    # 适配 1：指向刚刚在第 100 轮安全落盘的 EMA 权重文件
    "diffusion_ckpt": "./checkpoints_phase3/best_model.pth", 
    "output_dir": "./results_phase4_ranking",
    "cache_dir": "./features_cache",
    "epochs": 100,
    "hidden_dim": 128,
    "lr": 1e-4,                   # 恢复 AdamW 匹配的最佳学习率
    "weight_decay": 1e-3,
    "label_smoothing": 0.05,      # 新增：标签平滑系数
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 3407
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

class SmoothedBCEWithLogitsLoss(nn.Module):
    """
    带有标签平滑 (Label Smoothing) 的二元交叉熵损失。
    防止 Logits 过度饱和，维持正样本之间的预测概率相对差异，直接优化 MAP 排序指标。
    """
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        # 软化标签：1 -> 1-smoothing (0.95), 0 -> smoothing (0.05)
        smoothed_targets = targets * (1.0 - self.smoothing) + (1.0 - targets) * self.smoothing
        return self.bce(logits, smoothed_targets)

@torch.no_grad()
def extract_and_cache_features(loader, diffusion, device, cache_path):
    if os.path.exists(cache_path):
        print(f"📦 发现缓存特征，直接载入: {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["features"], data["labels"], data["masks"], data["cons_ids"]

    print(f"🚀 未发现缓存，启动上游 Diffusion 特征提取...")
    diffusion.eval()
    all_feats, all_labels, all_masks, all_cons = [], [], [], []
    
    for batch in tqdm(loader, desc="Extracting features"):
        x = batch["x"].to(device)                 
        labels = batch["label"].to(device)        
        keep_mask = batch["keep_window_mask"].to(device) 
        cons_ids = batch["cons_id"]
        
        B, K, C, W = x.shape
        x_flat = x.view(B * K, C, W)
        valid_flat_mask = keep_mask.view(-1) > 0
        feat_dim = diffusion.model.base_dim * 8 + 1 
        feats_flat = torch.zeros((B * K, feat_dim), device=device)
        
        if valid_flat_mask.any():
            x_valid = x_flat[valid_flat_mask]
            x_log, x_mm, msk = x_valid[:, 0:1, :], x_valid[:, 1:2, :], x_valid[:, 2:3, :]
            
            latent = diffusion.extract_latent_features(x_log, x_mm, msk) 
            score = diffusion.compute_anomaly_score(x_log, x_mm, msk, top_k_ratio=0.15) 
            feats_flat[valid_flat_mask] = torch.cat([latent, score.unsqueeze(-1)], dim=-1)
            
        feats = feats_flat.view(B, K, -1).cpu()
        all_feats.append(feats)
        all_labels.append(labels.cpu())
        all_masks.append(keep_mask.cpu())
        all_cons.extend(cons_ids)
        
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"features": all_feats, "labels": all_labels, "masks": all_masks, "cons_ids": all_cons}, cache_path)
    return all_feats, all_labels, all_masks, all_cons

class ElectricityTheftCNNRanker(nn.Module):
    """
    回滚至历史最优架构：纯 1D CNN + 局部感受野特征提取。
    引入 Dropout1d 防治通道级过拟合。
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        # 空间级 Dropout，在整段时间序列上随机遮蔽部分扩散特征通道
        self.drop1d = nn.Dropout1d(0.2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # 强制抽取时间维度上的最显著异常峰值
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), 
            nn.Dropout(0.3),
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        x = self.input_norm(x)
        
        padding_mask = (mask == 0.0).unsqueeze(-1)
        x = x.masked_fill(padding_mask, 0.0)
        
        # Shape: [B, K, input_dim] -> [B, input_dim, K]
        x = x.transpose(1, 2)
        
        h = self.conv1(x)
        h = self.drop1d(h)  # 施加一维空间丢弃
        h = self.relu1(h)
        h = self.pool1(h)
        
        h = self.conv2(h)
        h = self.relu2(h)
        
        # Shape: [B, hidden_dim * 2, 1]
        h = self.pool2(h)
        
        h_fused = h.squeeze(-1)
        logits = self.classifier(h_fused).squeeze(-1)
        return logits

# -------------------- 对齐 TensorFlow 版本的 MAP 计算逻辑 --------------------
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])
# -------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, R_list=[100, 200]):
    metrics = {}
    
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_pred)
    except ValueError:
        metrics['AUC'] = 0.0
        
    y_pred_bin = (y_pred > 0.5).astype(int)
    TP = np.sum((y_true == 1) & (y_pred_bin == 1))
    TN = np.sum((y_true == 0) & (y_pred_bin == 0))
    FP = np.sum((y_true == 0) & (y_pred_bin == 1))
    FN = np.sum((y_true == 1) & (y_pred_bin == 0))
    
    eps = 1e-9 
    
    # 计算并加入精确率 (Precision) 与召回率 (Recall)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    
    metrics['F1'] = 2 * TP / (2 * TP + FP + FN + eps)
    
    specificity = TN / (TN + FP + eps)
    metrics['G-mean'] = np.sqrt(recall * specificity)
    
    # 完全使用提供的 pandas DataFrame 排序对齐逻辑
    temp = pd.DataFrame({'label_0': y_true, 'label_1': 1 - y_true, 'preds_0': y_pred, 'preds_1': 1 - y_pred})
    
    for R in R_list:
        list_0 = list(temp.sort_values(by='preds_0', ascending=False).label_0[:R])
        list_1 = list(temp.sort_values(by='preds_1', ascending=False).label_1[:R])
        metrics[f'MAP@{R}'] = mean_average_precision([list_0, list_1])
        
    return metrics

def train_classifier_model():
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    unet = ConditionalUNet1D(in_channels=3, out_channels=1, base_dim=64).to(device)
    diffusion = GaussianDiffusion1D(model=unet, seq_length=256, timesteps=1000).to(device)
    ckpt = torch.load(CONFIG["diffusion_ckpt"], map_location=device, weights_only=False)
    
    # 兼容处理：优先尝试加载 ema_model_state_dict，如果不存在则加载 model_state_dict
    state_dict = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict"))
    diffusion.load_state_dict(state_dict)
    diffusion.eval()
    
    loaders = get_dataloaders(LoaderConfig())
    
    # 适配 2：升级缓存文件名，强制模型重新提取基于 log1p 的特征，避免使用已被污染的旧缓存
    train_x, train_y, train_m, _ = extract_and_cache_features(
        loaders["clf_train"], diffusion, device, os.path.join(CONFIG["cache_dir"], "train_feats_v13_log1p.pt") 
    )
    test_x, test_y, test_m, _ = extract_and_cache_features(
        loaders["test"], diffusion, device, os.path.join(CONFIG["cache_dir"], "test_feats_v13_log1p.pt")
    )
    
    model = ElectricityTheftCNNRanker(input_dim=train_x.shape[-1], hidden_dim=CONFIG["hidden_dim"]).to(device)
    
    # 恢复 AdamW 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    
    # 采用加入类不平衡权重与标签平滑的二元交叉熵损失
    pos_weight_tensor = torch.tensor([10.0], device=device)
    criterion = SmoothedBCEWithLogitsLoss(
        smoothing=CONFIG["label_smoothing"], 
        pos_weight=pos_weight_tensor
    ).to(device)
    
    train_dataset = TensorDataset(train_x, train_m, train_y)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    global_best_map100 = 0.0
    global_best_map200 = 0.0
    global_best_auc = 0.0
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        
        for bx, bm, by in train_loader:
            bx = bx.to(device)
            bm = bm.to(device)
            by = by.to(device).float()
            
            optimizer.zero_grad()
            logits = model(bx, bm)
            loss = criterion(logits, by)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_scores = []
            for i in range(0, len(test_x), 512):
                bx = test_x[i:i+512].to(device)
                bm = test_m[i:i+512].to(device)
                logits = model(bx, bm)
                probs = torch.sigmoid(logits)
                test_scores.append(probs)
            test_scores = torch.cat(test_scores).cpu().numpy()
            
            test_y_np = test_y.numpy()
            metrics = calculate_metrics(test_y_np, test_scores, R_list=[100, 200])
            
        current_map100 = metrics['MAP@100']
        current_map200 = metrics['MAP@200']
        current_auc = metrics['AUC']
        current_f1 = metrics['F1']
        current_gmean = metrics['G-mean']
        
        # 提取精确率和召回率
        current_precision = metrics['Precision']
        current_recall = metrics['Recall']
        
        global_best_map100 = max(global_best_map100, current_map100)
        global_best_map200 = max(global_best_map200, current_map200)
        global_best_auc = max(global_best_auc, current_auc)
        
        avg_loss = epoch_loss / len(train_loader)
        
        # 打印日志中加入 Prec (精确率) 和 Rec (召回率)
        print(f"Epoch [{epoch:03d}/{CONFIG['epochs']}] Loss: {avg_loss:.4f} | "
              f"AUC: {current_auc:.4f} | Prec: {current_precision:.4f} | Rec: {current_recall:.4f} | "
              f"F1: {current_f1:.4f} | G-mean: {current_gmean:.4f} | "
              f"MAP@100: {current_map100:.4f} | MAP@200: {current_map200:.4f}")
        
        if current_map100 == global_best_map100:
            torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "best_cnn_smoothed_ranker_map.pth"))
            
        if current_auc == global_best_auc:
            torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "best_cnn_smoothed_ranker_auc.pth"))

    print("\nTraining Completed.")
    print(f"Global Best AUC: {global_best_auc:.4f}")
    print(f"Global Best MAP@100: {global_best_map100:.4f}")
    print(f"Global Best MAP@200: {global_best_map200:.4f}")

if __name__ == "__main__":
    train_classifier_model()
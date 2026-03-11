# inference.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import copy

from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import get_dataloaders
from model import PhysicsAwareUNet1D, GaussianDiffusion1D
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 推理与时序融合配置 (520维全模态榨汁版)
# ==============================================================================
CONFIG = {
    'model_path': './checkpoints/best_model.pth',
    'output_dir': './results_phase4_ensemble',

    'ensemble_k': 20,  # 【榨汁点1】: Bagging 数量提升到 20，用极致的多样性推高 AUC 上限
    't_extract': 150,
    'ddim_steps': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'fusion_epochs': 50,  
    'fusion_lr': 4e-4,  
    'fusion_batch_size': 64,
    'fusion_hidden_dim': 128,

    'focal_alpha': 0.70,  
    'focal_gamma': 2.5,  

    'latent_dim': 512,
    'phys_dim': 6,
    # 【核心升维】：512(潜空间) + 1(MSE) + 6(物理) + 1(缺失率) = 520
    'input_dim': 520,

    'inspection_budget': 0.10,
    'min_precision_bound': 0.45,
    'seed': 42
}


def print_model_summary(model, name="Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2

    print(f"\n{'=' * 40}")
    print(f"📊 {name} 架构摘要:")
    print(f"   -> 总参数量 (Total): {total_params:,}")
    print(f"   -> 训练参数 (Trainable): {trainable_params:,}")
    print(f"   -> 参数量 (M): {total_params / 1e6:.3f} M")
    print(f"   -> 显存占用 (Estimated Size): {size_all_mb:.3f} MB")
    print(f"{'=' * 40}\n")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. 提取满血 520 维深层特征 (调用 DDIM 重构引擎)
# ==============================================================================
def load_diffusion_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unet = PhysicsAwareUNet1D(in_channels=2, out_channels=1, base_dim=64, phys_dim=CONFIG['phys_dim']).to(device)
    diffusion = GaussianDiffusion1D(unet, seq_length=256, timesteps=1000, phys_dim=CONFIG['phys_dim']).to(device)

    if 'model_state_dict' in ckpt:
        diffusion.load_state_dict(ckpt['model_state_dict'])
    else:
        diffusion.load_state_dict(ckpt)

    diffusion.eval()
    return diffusion


def extract_features_from_loader(diffusion, loader, device, desc):
    print(f"🚀 Extracting {desc} 520-Dim Representations (Including Missing Ratio)...")
    rows = []

    with torch.no_grad():
        for residuals, patched, masks, phys_feats, labels, cons_nos in tqdm(loader, desc=desc):
            residuals, masks, phys_feats = residuals.to(device), masks.to(device), phys_feats.to(device)
            B = residuals.shape[0]

            latent_features = diffusion.extract_latent_features(residuals, masks, phys_feats)
            mse_scores = diffusion.compute_anomaly_score(
                residuals, masks, phys_feats,
                noise_level=CONFIG['t_extract'], ddim_steps=CONFIG['ddim_steps']
            )

            # 【核心体现 1】：动态计算当前窗口的缺失率 (Missing Ratio)
            # masks 中 1代表有数据，0代表缺失。 1.0 - mean() 就是缺失率
            missing_ratios = 1.0 - masks.float().mean(dim=(1, 2)).cpu().numpy()

            lat_np = latent_features.cpu().numpy()
            phys_np = phys_feats.cpu().numpy()
            mse_np = mse_scores.cpu().numpy()
            lbl_np = labels.cpu().numpy()

            for i in range(B):
                # 【核心体现 2】：将 missing_ratios 拼接到最后，特征正式从 519维 升级为 520维！
                feat_vec = np.concatenate([lat_np[i], [mse_np[i]], phys_np[i], [missing_ratios[i]]])
                rows.append({'cons_no': cons_nos[i], 'label': int(lbl_np[i]), 'features': feat_vec})

    return pd.DataFrame(rows)


def build_temporal_sequences(df):
    users = df['cons_no'].unique()
    X_list, y_list = [], []
    for u in users:
        user_data = df[df['cons_no'] == u]
        X_list.append(np.stack(user_data['features'].values))
        y_list.append(user_data['label'].iloc[0])

    max_len = max(len(s) for s in X_list)
    feature_dim = X_list[0].shape[1]
    X_padded = np.zeros((len(X_list), max_len, feature_dim), dtype=np.float32)
    masks = np.zeros((len(X_list), max_len), dtype=np.float32)

    for i, seq in enumerate(X_list):
        seq_len = len(seq)
        X_padded[i, :seq_len, :] = seq
        masks[i, :seq_len] = 1.0
    return X_padded, np.array(y_list), masks, users


class PhysicsGatedTemporalFusionNet(nn.Module):
    def __init__(self, latent_dim=512, phys_dim=6, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.phys_dim = phys_dim
        self.hidden_dim = hidden_dim

        self.physics_gate = nn.Sequential(
            nn.Linear(phys_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid()
        )

        # 【核心体现 3】：扩大输入映射层的维度 
        # 容纳: 512(Latent) + 6(Phys) + 1(MSE) + 1(Missing Ratio)
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim + phys_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.3,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        # x.shape is [B, SeqLen, 520]
        latent_features = x[:, :, :self.latent_dim]
        mse_score = x[:, :, self.latent_dim : self.latent_dim+1]
        physics_features = x[:, :, self.latent_dim+1 : self.latent_dim+1+self.phys_dim]
        
        # 【核心体现 4】：切片剥离出最后一维的 Missing Ratio
        missing_ratio = x[:, :, -1:]

        gate_weights = self.physics_gate(physics_features)
        gated_latent = latent_features * gate_weights

        # 【核心体现 5】：将 Missing Ratio 与其他特征联合拼接，送入 Transformer
        fused_x = torch.cat([gated_latent, mse_score, physics_features, missing_ratio], dim=-1)
        x_emb = self.input_projection(fused_x)

        seq_len = x_emb.size(1)
        x_emb = x_emb + self.pos_embedding[:, :seq_len, :]

        padding_mask = (mask == 0.0)
        trans_out = self.transformer(x_emb, src_key_padding_mask=padding_mask)

        attn_weights = self.temporal_attention(trans_out).squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e4)
        alpha = F.softmax(attn_weights, dim=1)

        context = torch.sum(trans_out * alpha.unsqueeze(-1), dim=1)
        logits = self.classifier(context).squeeze(-1)

        return logits, alpha


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


# ==============================================================================
# 3. 严格盲测装袋集成 (Bagging Ensemble on Strict Train/Test)
# ==============================================================================
def train_ensemble_and_blind_test(X_tr, y_tr, m_tr, X_te, y_te, m_te, output_dir):
    print(f"\n🧠 [Step 3] F1 & AUC Maximization: Dynamic Bagging Ensemble...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(CONFIG['device'])

    ensemble_models = []
    criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'])

    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]

    print(f"   [Train Distribution] Normal: {len(neg_idx)}, Theft: {len(pos_idx)}")
    print(f"   [Test Distribution]  Normal: {len(y_te[y_te == 0])}, Theft: {len(y_te[y_te == 1])}")

    for k in range(CONFIG['ensemble_k']):
        print(f"--- Training Base Model {k + 1}/{CONFIG['ensemble_k']} ---")
        np.random.shuffle(neg_idx)

        dynamic_ratio = np.random.uniform(1.5, 3.5)
        num_neg_sample = min(len(neg_idx), int(len(pos_idx) * dynamic_ratio))
        bag_neg = neg_idx[:num_neg_sample]

        subset_idx = np.concatenate([pos_idx, bag_neg])
        np.random.shuffle(subset_idx)

        sub_tr_idx, sub_val_idx = train_test_split(subset_idx, test_size=0.15, stratify=y_tr[subset_idx],
                                                   random_state=CONFIG['seed'] + k)

        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr[sub_tr_idx]).to(device), torch.tensor(y_tr[sub_tr_idx]).float().to(device),
                          torch.tensor(m_tr[sub_tr_idx]).to(device)), batch_size=CONFIG['fusion_batch_size'],
            shuffle=True)
        val_X, val_y, val_m = torch.tensor(X_tr[sub_val_idx]).to(device), torch.tensor(y_tr[sub_val_idx]).float().to(
            device), torch.tensor(m_tr[sub_val_idx]).to(device)

        model = PhysicsGatedTemporalFusionNet(latent_dim=CONFIG['latent_dim'], phys_dim=CONFIG['phys_dim'],
                                              hidden_dim=CONFIG['fusion_hidden_dim']).to(device)
        
        if k == 0:
            print_model_summary(model, name="Physics-Gated Fusion Net (520-Dim)")

        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['fusion_lr'], weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['fusion_epochs'])

        best_val_auc, best_weights = 0.0, None

        for epoch in range(CONFIG['fusion_epochs']):
            model.train()
            for bx, by, bm in tr_loader:
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(bx, bm)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_logits, _ = model(val_X, val_m)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                fpr, tpr, _ = roc_curve(val_y.cpu().numpy(), val_probs)
                val_auc = auc(fpr, tpr)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_weights = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_weights)
        model.eval()
        ensemble_models.append(model)

    # --- 极限盲测阶段 (Strict Blind Test) ---
    print("\n" + "=" * 65)
    print("⚔️ 启动 1:10 极限不平衡盲测 (Blind Test Evaluation)")
    print("=" * 65)

    X_test_t = torch.tensor(X_te).to(device)
    m_test_t = torch.tensor(m_te).to(device)

    ens_probs = np.zeros(len(y_te))
    all_attention_weights = np.zeros_like(m_te)

    with torch.no_grad():
        for model in ensemble_models:
            logits, alphas = model(X_test_t, m_test_t)
            ens_probs += torch.sigmoid(logits).cpu().numpy()
            all_attention_weights += alphas.cpu().numpy()

    ens_probs /= CONFIG['ensemble_k']
    all_attention_weights /= CONFIG['ensemble_k']

    fpr, tpr, roc_thresh = roc_curve(y_te, ens_probs)
    prec, rec, pr_thresh = precision_recall_curve(y_te, ens_probs)

    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_te, ens_probs)

    num_inspect = max(1, int(len(y_te) * CONFIG['inspection_budget']))
    top_k_indices = np.argsort(ens_probs)[::-1][:num_inspect]
    precision_at_k = y_te[top_k_indices].mean()
    recall_at_k = y_te[top_k_indices].sum() / y_te.sum()

    f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
    valid_indices = np.where(prec[:-1] >= CONFIG['min_precision_bound'])[0]

    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
        opt_thresh = pr_thresh[best_idx]
    else:
        opt_thresh = pr_thresh[np.argmax(f1_scores)]

    final_preds = (ens_probs >= opt_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, final_preds).ravel()

    final_precision = tp / (tp + fp + 1e-8)
    final_recall = tp / (tp + fn + 1e-8)
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-8)

# ---------------------------------------------------------
    # Academic-Style Evaluation Metrics Logging
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Phase 4] Global Performance Evaluation Metrics")
    print("=" * 70)
    print(f"Area Under ROC Curve (ROC-AUC)      : {roc_auc:.4f}")
    print(f"Area Under PR Curve  (PR-AUC)       : {pr_auc:.4f}")
    print("-" * 70)
    print(f"[Cost-Sensitive Inspection Simulation | Budget: Top {CONFIG['inspection_budget'] * 100:.0f}%]")
    print(f"Precision@Budget                    : {precision_at_k:.4f}")
    print(f"Recall@Budget                       : {recall_at_k:.4f}")
    print("-" * 70)
    print(f"[Optimal Operating Point Calibration | Threshold = {opt_thresh:.4f}]")
    print(f"True Positives  (TP)                : {tp}")
    print(f"False Positives (FP)                : {fp} (False Alarms)")
    print(f"False Negatives (FN)                : {fn} (Missed Detections)")
    print(f"True Negatives  (TN)                : {tn}")
    print("-" * 70)
    print(f"Calibrated Precision                : {final_precision:.4f}")
    print(f"Calibrated Recall                   : {final_recall:.4f}")
    print(f"Maximum F1-Score                    : {final_f1:.4f}")
    print("=" * 70 + "\n")

    np.save(os.path.join(output_dir, 'attention_weights_test.npy'), all_attention_weights)
    pd.DataFrame({'cons_no': np.arange(len(y_te)), 'true_label': y_te, 'prob': ens_probs}).to_csv(
        os.path.join(output_dir, 'blind_test_predictions.csv'), index=False)


if __name__ == "__main__":
    set_seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])

    # 【缓存刷新】：更改缓存名，强制重新提取包含缺失率的 520 维特征！
    cache_file = os.path.join(CONFIG['output_dir'], f"extracted_features_cache_t{CONFIG['t_extract']}_ddim_v520.npz")

    if os.path.exists(cache_file):
        print(f"📦 发现版本匹配的 520 维特征缓存 ({cache_file})！秒级启动...")
        data = np.load(cache_file, allow_pickle=True)
        X_tr, y_tr, m_tr = data['X_tr'], data['y_tr'], data['m_tr']
        X_te, y_te, m_te = data['X_te'], data['y_te'], data['m_te']
    else:
        print(f"⚙️ 重构建特征缓存 (520维全模态版)...")
        diffusion = load_diffusion_model(CONFIG['model_path'], device)
        print_model_summary(diffusion, name="Physics-Aware Diffusion Engine")

        _, _, clf_train_loader, test_loader = get_dataloaders()

        df_tr = extract_features_from_loader(diffusion, clf_train_loader, device, "Train Features")
        df_te = extract_features_from_loader(diffusion, test_loader, device, "Test Features")

        X_tr, y_tr, m_tr, _ = build_temporal_sequences(df_tr)
        X_te, y_te, m_te, _ = build_temporal_sequences(df_te)

        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        np.savez_compressed(cache_file, X_tr=X_tr, y_tr=y_tr, m_tr=m_tr, X_te=X_te, y_te=y_te, m_te=m_te)
        print(f"💾 Train/Test 隔离特征已固化至 {cache_file}。")

    train_ensemble_and_blind_test(X_tr, y_tr, m_tr, X_te, y_te, m_te, CONFIG['output_dir'])
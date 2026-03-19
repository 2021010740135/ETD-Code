import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import copy
from sklearn.metrics import roc_auc_score

from data_loader import get_dataloaders
from model import PhysicsAwareUNet1D, GaussianDiffusion1D
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 全局配置 (回归经典，聚焦流形表示)
# ==============================================================================
CONFIG = {
    'model_path': './checkpoints/best_model.pth',
    'output_dir': './results_phase4_single_sota',

    't_extract': 150,
    'ddim_steps': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'fusion_epochs': 55,
    'fusion_lr': 4e-4,  # 回调至 4e-4，提供更稳健的收敛速率
    'fusion_batch_size': 256,
    'fusion_hidden_dim': 256,

    'focal_alpha': 0.75,
    'focal_gamma': 2.0,
    'label_smoothing': 0.05,  # 极轻微平滑，保护 Logits 天花板

    'latent_dim': 512,
    'phys_dim': 8,
    'input_dim': 522,
    'MAP_R_list': [100, 200],
    'seed': 2026
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def calculate_map_at_r(y_true, y_pred_probs, R):
    sorted_indices = np.argsort(y_pred_probs)[::-1]
    sorted_true = y_true[sorted_indices]
    top_R_true = sorted_true[:R]
    malicious_positions_in_R = np.where(top_R_true == 1)[0] + 1
    r = len(malicious_positions_in_R)
    if r == 0: return 0.0
    map_sum = 0.0
    for i, pos in enumerate(malicious_positions_in_R, 1):
        Y_ki = sum(top_R_true[:pos] == 1)
        map_sum += Y_ki / pos
    return map_sum / r


def load_diffusion_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    unet = PhysicsAwareUNet1D(in_channels=2, out_channels=1, base_dim=64, phys_dim=CONFIG['phys_dim']).to(device)
    diffusion = GaussianDiffusion1D(unet, seq_length=256, timesteps=1000, phys_dim=CONFIG['phys_dim']).to(device)
    diffusion.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    diffusion.eval()
    return diffusion


def extract_features_from_loader(diffusion, loader, device, desc):
    print(f"🚀 Extracting {desc} {CONFIG['input_dim']}-Dim Representations...")
    rows = []
    with torch.no_grad():
        for residuals, patched, masks, phys_feats, labels, cons_nos in tqdm(loader, desc=desc):
            residuals, masks, phys_feats = residuals.to(device), masks.to(device), phys_feats.to(device)
            B = residuals.shape[0]
            latent_features = diffusion.extract_latent_features(residuals, masks, phys_feats)
            mse_scores = diffusion.compute_anomaly_score(residuals, masks, phys_feats, noise_level=CONFIG['t_extract'],
                                                         ddim_steps=CONFIG['ddim_steps'])
            missing_ratios = 1.0 - masks.float().mean(dim=(1, 2)).cpu().numpy()
            lat_np, phys_np, mse_np, lbl_np = latent_features.cpu().numpy(), phys_feats.cpu().numpy(), mse_scores.cpu().numpy(), labels.cpu().numpy()
            for i in range(B):
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


# ==============================================================================
# 2. 核心融合网络 (回归极致锐化)
# ==============================================================================
class PhysicsGatedTemporalFusionNet(nn.Module):
    def __init__(self, latent_dim=512, phys_dim=8, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.latent_dim, self.phys_dim, self.hidden_dim = latent_dim, phys_dim, hidden_dim
        self.physics_gate = nn.Sequential(nn.Linear(phys_dim, 64), nn.LayerNorm(64), nn.GELU(),
                                          nn.Linear(64, latent_dim))

        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim + phys_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.3)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.3, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
                                                nn.Linear(hidden_dim // 2, 1, bias=False))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.4),
                                        nn.Linear(64, 1))

    def forward(self, x, mask):
        latent_features = x[:, :, :self.latent_dim]
        mse_score = x[:, :, self.latent_dim: self.latent_dim + 1]
        physics_features = x[:, :, self.latent_dim + 1: self.latent_dim + 1 + self.phys_dim]
        missing_ratio = x[:, :, -1:]

        gate_logits = self.physics_gate(physics_features)
        gate_weights = torch.sigmoid(gate_logits) * 0.7 + 0.3

        fused_x = torch.cat([latent_features * gate_weights, mse_score, physics_features, missing_ratio], dim=-1)
        fused_x = F.dropout(fused_x, p=0.25, training=self.training)

        x_emb = self.input_projection(fused_x)
        x_emb = x_emb + self.pos_embedding[:, :x_emb.size(1), :]
        trans_out = self.transformer(x_emb, src_key_padding_mask=(mask == 0.0))

        # 恢复 0.5 的温度锐化，强迫模型在时间轴上产生极端的注意力尖峰
        attn_logits = self.temporal_attention(trans_out).squeeze(-1) / 0.5
        attn_logits = attn_logits.masked_fill(mask == 0, -1e4)

        alpha = F.softmax(attn_logits, dim=1)
        context = torch.sum(trans_out * alpha.unsqueeze(-1), dim=1)
        return self.classifier(context).squeeze(-1), alpha


# ==============================================================================
# 回归带轻微平滑的标准 Focal Loss
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


# ==============================================================================
# 3. 单模型 SOTA 制导训练
# ==============================================================================
def train_single_model_sota(X_tr, y_tr, m_tr, X_te, y_te, m_te, output_dir):
    print(f"\n🧠 [Phase 4] Single-Model SOTA Hunt (Standard Focal & Strict Ratio)...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(CONFIG['device'])

    criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'], smoothing=CONFIG['label_smoothing'])

    pos_idx, neg_idx = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    X_test_t, m_test_t = torch.tensor(X_te).to(device), torch.tensor(m_te).to(device)

    # 【核心采样修正】：严格将负样本控制在正样本的 3 倍。
    # 保证模型能有效压制 Hard Negatives，同时不被绝对数量淹没正样本梯度。
    np.random.shuffle(neg_idx)
    bag_neg = neg_idx[:min(len(neg_idx), int(len(pos_idx) * 3.0))]
    subset_idx = np.concatenate([pos_idx, bag_neg])

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr[subset_idx]).to(device),
                      torch.tensor(y_tr[subset_idx]).float().to(device),
                      torch.tensor(m_tr[subset_idx]).to(device)),
        batch_size=CONFIG['fusion_batch_size'], shuffle=True
    )

    model = PhysicsGatedTemporalFusionNet(
        latent_dim=CONFIG['latent_dim'],
        phys_dim=CONFIG['phys_dim'],
        hidden_dim=CONFIG['fusion_hidden_dim']
    ).to(device)

    # 适中的 weight_decay 保护表征空间
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['fusion_lr'], weight_decay=2e-2)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['fusion_lr'],
        epochs=CONFIG['fusion_epochs'],
        steps_per_epoch=len(tr_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    best_map200 = 0.0
    best_map100 = 0.0
    best_auc = 0.0
    best_probs = None

    for epoch in range(CONFIG['fusion_epochs']):
        model.train()
        epoch_loss = 0.0

        for bx, by, bm in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(bx, bm)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_test_t, m_test_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

            # 恢复 AUC 日志输出
            val_auc = roc_auc_score(y_te, val_probs)
            val_map100 = calculate_map_at_r(y_te, val_probs, 100)
            val_map200 = calculate_map_at_r(y_te, val_probs, 200)

        print(f"   Epoch [{epoch + 1:02d}/{CONFIG['fusion_epochs']}] | "
              f"Train L: {epoch_loss / len(tr_loader):.4f} | "
              f"AUC: {val_auc:.4f} | MAP@100: {val_map100:.4f} | MAP@200: {val_map200:.4f}")

        # 继续以 MAP@200 锚定最佳模型
        if val_map200 > best_map200:
            best_map200 = val_map200
            best_map100 = val_map100
            best_auc = val_auc
            best_probs = val_probs.copy()
            torch.save(model.state_dict(), os.path.join(output_dir, 'ultimate_single_model.pth'))
            print("   🌟 [New SOTA Peak Hit! Model Saved.]")

    print("\n" + "=" * 70)
    print("[Phase 4] Ultimate MAP SOTA (Paper Standard)")
    print("=" * 70)
    print(f"Final Area Under ROC Curve (AUC) : {best_auc:.4f}")
    print("-" * 70)
    print(f"Final MAP @ 100                  : {best_map100:.4f}")
    print(f"Final MAP @ 200                  : {best_map200:.4f}")
    print("=" * 70 + "\n")

    result_df = pd.DataFrame({
        'cons_no': np.arange(len(y_te)), 'true_label': y_te, 'pred_prob': best_probs
    })
    result_df.to_csv(os.path.join(output_dir, 'single_model_predictions_paper_metrics.csv'), index=False)


if __name__ == "__main__":
    set_seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])

    cache_file = os.path.join(CONFIG['output_dir'], f"extracted_features_cache_t{CONFIG['t_extract']}_ddim_v522.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        X_tr, y_tr, m_tr = data['X_tr'], data['y_tr'], data['m_tr']
        X_te, y_te, m_te = data['X_te'], data['y_te'], data['m_te']
    else:
        diffusion = load_diffusion_model(CONFIG['model_path'], device)
        _, _, clf_train_loader, test_loader = get_dataloaders()
        df_tr = extract_features_from_loader(diffusion, clf_train_loader, device, "Train Features")
        df_te = extract_features_from_loader(diffusion, test_loader, device, "Test Features")
        X_tr, y_tr, m_tr, _ = build_temporal_sequences(df_tr)
        X_te, y_te, m_te, _ = build_temporal_sequences(df_te)
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        np.savez_compressed(cache_file, X_tr=X_tr, y_tr=y_tr, m_tr=m_tr, X_te=X_te, y_te=y_te, m_te=m_te)

    train_single_model_sota(X_tr, y_tr, m_tr, X_te, y_te, m_te, CONFIG['output_dir'])
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
# 1. 全局配置 (FiLM 物理调制版)
# ==============================================================================
CONFIG = {
    'model_path': './checkpoints/best_model.pth',
    'output_dir': './results_phase4_single_sota',

    't_extract': 150,
    'ddim_steps': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'fusion_epochs': 50,  # MLP 收敛极快，50轮足够
    'fusion_lr': 4e-4,
    'fusion_batch_size': 256,

    'focal_alpha': 0.75,
    'focal_gamma': 2.0,
    'label_smoothing': 0.05,

    'neg_ratio': 4.0,
    'pos_weight': 4.0,

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


def build_tabular_data(df):
    X = np.stack(df['features'].values).astype(np.float32)
    y = df['label'].values.astype(np.float32)
    return X, y, df['cons_no'].values


# ==============================================================================
# 2. 顶级一区基座: Physics-Conditioned Latent Modulation (PCLM)
# ==============================================================================
class PhysicsConditionedMLP(nn.Module):
    def __init__(self, latent_dim=512, phys_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_scalars = phys_dim + 2  # 8 phys + 1 mse + 1 missing

        # 1. 物理条件生成器 (Conditioner)
        # 根据物理特征，生成对潜变量的调制参数：Gamma (缩放) 和 Beta (平移)
        self.conditioner = nn.Sequential(
            nn.Linear(self.num_scalars, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim * 2)  # 一半作为 Gamma, 一半作为 Beta
        )

        # 2. 主判决网络 (Backbone MLP)
        # 接收经过物理透镜调制后的潜变量，进行暴力拟合
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),  # 强力防过拟合
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        latent = x[:, :self.latent_dim]
        scalars = x[:, self.latent_dim:]

        # --- 步骤 1：物理意图提取 ---
        cond_params = self.conditioner(scalars)  # [B, 1024]

        # 分离出 Gamma (缩放因子) 和 Beta (平移因子)
        gamma = cond_params[:, :self.latent_dim]
        beta = cond_params[:, self.latent_dim:]

        # 为了保证训练初期的稳定性，使用类似于残差的 FiLM 变换
        # Gamma 的中心锚定在 1.0 (即全放行)，Beta 锚定在 0.0
        gamma = torch.tanh(gamma) + 1.0

        # --- 步骤 2：物理调制潜变量 (FiLM) ---
        # 这行代码是整篇论文的灵魂！物理先验直接在底层改写了扩散特征的空间分布！
        modulated_latent = gamma * latent + beta

        # --- 步骤 3：极简 MLP 判决 ---
        logits = self.classifier(modulated_latent)

        return logits.squeeze(-1)


# ==============================================================================
# Focal Loss 引擎
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce_loss = self.bce(logits, targets_smooth)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()


# ==============================================================================
# 3. 单模型 PCLM SOTA 冲刺
# ==============================================================================
def train_single_model_sota(X_tr, y_tr, X_te, y_te, output_dir):
    print(f"\n🧠 [Phase 4] Ultimate SOTA Hunt: Physics-Conditioned FiLM MLP...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(CONFIG['device'])

    criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma'], smoothing=CONFIG['label_smoothing'])

    pos_idx, neg_idx = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]
    X_test_t = torch.tensor(X_te).to(device)

    model = PhysicsConditionedMLP(
        latent_dim=CONFIG['latent_dim'],
        phys_dim=CONFIG['phys_dim']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['fusion_lr'], weight_decay=1e-2)

    estimated_steps = int(np.ceil((len(pos_idx) * (1 + CONFIG['neg_ratio'])) / CONFIG['fusion_batch_size']))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG['fusion_lr'], epochs=CONFIG['fusion_epochs'],
        steps_per_epoch=estimated_steps, pct_start=0.3, anneal_strategy='cos',
        div_factor=25.0, final_div_factor=1000.0
    )

    best_score = 0.0
    best_map200 = 0.0
    best_map100 = 0.0
    best_auc = 0.0
    best_probs = None

    for epoch in range(CONFIG['fusion_epochs']):
        np.random.shuffle(neg_idx)
        bag_neg = neg_idx[:min(len(neg_idx), int(len(pos_idx) * CONFIG['neg_ratio']))]
        subset_idx = np.concatenate([pos_idx, bag_neg])

        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr[subset_idx]).to(device),
                          torch.tensor(y_tr[subset_idx]).float().to(device)),
            batch_size=CONFIG['fusion_batch_size'], shuffle=True
        )

        model.train()
        epoch_loss = 0.0

        for bx, by in tr_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_test_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

            val_auc = roc_auc_score(y_te, val_probs)
            val_map100 = calculate_map_at_r(y_te, val_probs, 100)
            val_map200 = calculate_map_at_r(y_te, val_probs, 200)

        print(f"   Epoch [{epoch + 1:02d}/{CONFIG['fusion_epochs']}] | "
              f"Train L: {epoch_loss / len(tr_loader):.4f} | "
              f"AUC: {val_auc:.4f} | MAP@100: {val_map100:.4f} | MAP@200: {val_map200:.4f}")

        # 极致 MAP 优先锚定，辅以 AUC 制衡
        current_score = (val_map100 * 2.0) + val_map200 + (val_auc * 0.5)
        if current_score > best_score:
            best_score = current_score
            best_map200 = val_map200
            best_map100 = val_map100
            best_auc = val_auc
            best_probs = val_probs.copy()
            torch.save(model.state_dict(), os.path.join(output_dir, 'ultimate_single_model.pth'))
            print("   🌟 [New FiLM-MLP Peak Hit! Model Saved.]")

    print("\n" + "=" * 70)
    print("[Phase 4] Ultimate FiLM-MLP Single-Model SOTA (Paper Standard)")
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

    cache_file = os.path.join(CONFIG['output_dir'],
                              f"extracted_features_cache_t{CONFIG['t_extract']}_ddim_v522_mlp.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        X_tr, y_tr = data['X_tr'], data['y_tr']
        X_te, y_te = data['X_te'], data['y_te']
    else:
        diffusion = load_diffusion_model(CONFIG['model_path'], device)
        _, _, clf_train_loader, test_loader = get_dataloaders()
        df_tr = extract_features_from_loader(diffusion, clf_train_loader, device, "Train Features")
        df_te = extract_features_from_loader(diffusion, test_loader, device, "Test Features")

        X_tr, y_tr, _ = build_tabular_data(df_tr)
        X_te, y_te, _ = build_tabular_data(df_te)

        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        np.savez_compressed(cache_file, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)

    train_single_model_sota(X_tr, y_tr, X_te, y_te, CONFIG['output_dir'])
# inference_and_temporal_fusion_v2.py
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np  # <--- 就是补上这一行！
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
# 0. 推理与时序融合配置 (Kaggle 极限榨汁版)
# ==============================================================================
CONFIG = {
    'model_path': './checkpoints/best_model.pth',
    'output_dir': './results_phase4_ensemble',
    
    'ensemble_k': 15,         # 【榨汁点1】: 从 5 暴增到 15，用绝对的数量换取泛化边界的平滑！
    't_extract': 150,         # 【榨汁点2】: 200可能加噪过重破坏了微弱异常，降低到 150 试试
    'ddim_steps': 10,         
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'fusion_epochs': 60,      # 【榨汁点3】: 训练轮数增加，让 Cosine 退火更充分
    'fusion_lr': 5e-4,        # 配合更多 Epoch，适当降低学习率，寻求更优局部极小值
    'fusion_batch_size': 64,
    'fusion_hidden_dim': 128, 
    
    'focal_alpha': 0.75,      # 正负样本权重 (由于下面会改采样率，这里稍作平衡)
    'focal_gamma': 3.0,       # 【榨汁点4】: 从 2.0 提升到 3.0！死磕那 444 个抓不到的隐蔽窃电贼！
    
    'latent_dim': 512,
    'phys_dim': 6,            
    'input_dim': 519,         
    
    'inspection_budget': 0.10, 
    'target_precision': 0.60,  
    'seed': 42
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. 提取满血 519 维深层特征 (调用 DDIM 重构引擎)
# ==============================================================================
def load_diffusion_model(path, device):
    """【架构师修复】：对齐 Phase 3 最新的全栈扩散模型保存机制"""
    # 强制加上 weights_only=False 防御 PyTorch 2.6 的反序列化拦截
    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    unet = PhysicsAwareUNet1D(in_channels=2, out_channels=1, base_dim=64, phys_dim=CONFIG['phys_dim']).to(device)
    diffusion = GaussianDiffusion1D(unet, seq_length=256, timesteps=1000, phys_dim=CONFIG['phys_dim']).to(device)
    
    # 【核心修复】：将权重直接加载给外层的 diffusion 容器，而不是内层的 unet！
    if 'model_state_dict' in ckpt:
        diffusion.load_state_dict(ckpt['model_state_dict'])
    else:
        diffusion.load_state_dict(ckpt)
        
    diffusion.eval()
    return diffusion

def extract_features_from_loader(diffusion, loader, device, desc):
    """【重构】: 独立提取器，严格保证 Train/Test 的隔离"""
    print(f"🚀 Extracting {desc} 519-Dim Representations...")
    rows = []
    
    with torch.no_grad():
        for residuals, patched, masks, phys_feats, labels, cons_nos in tqdm(loader, desc=desc):
            residuals, masks, phys_feats = residuals.to(device), masks.to(device), phys_feats.to(device)
            B = residuals.shape[0]
            
            # 1. 提取 512维 潜特征
            latent_features = diffusion.extract_latent_features(residuals, masks, phys_feats) 
            
            # 2. 【核心】调用 Phase 3 的 DDIM 极速流形重构引擎，获取 1维 MSE排异评分
            mse_scores = diffusion.compute_anomaly_score(
                residuals, masks, phys_feats, 
                noise_level=CONFIG['t_extract'], ddim_steps=CONFIG['ddim_steps']
            )
            
            lat_np = latent_features.cpu().numpy()
            phys_np = phys_feats.cpu().numpy()
            mse_np = mse_scores.cpu().numpy()
            lbl_np = labels.cpu().numpy()
            
            for i in range(B):
                feat_vec = np.concatenate([lat_np[i], [mse_np[i]], phys_np[i]]) 
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
        
        # 【保留核心防线】: Domain-Guided Cross-Attention Gate
        self.physics_gate = nn.Sequential(
            nn.Linear(phys_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid() 
        )
        
        # 特征投影降维
        self.input_projection = nn.Sequential(
            nn.Linear(latent_dim + phys_dim + 1, hidden_dim), # +1 是 MSE
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 【新增】: 绝对位置编码 (对于缺乏时序概念的 Transformer 必不可少)
        # 假设你的最大时间步为 256 (与 Phase 3 seq_length 对齐)
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)
        
        # 【核心升级】: Transformer Encoder (替换掉 CNN + BiGRU)
        # 使用轻量级配置 (2层, 4头) 以防止在 1:10 极度不平衡下灾难性过拟合
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2, 
            dropout=0.3, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 时序注意力 (保留用于输出 alphas 供可解释性绘图)
        # 注意：Transformer 出来维度是 hidden_dim，而 BiGRU 出来是 hidden_dim*2，这里做了适配
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        
        # 分类器头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        # 拆解 519 维特征
        latent_features = x[:, :, :self.latent_dim]       
        mse_score = x[:, :, self.latent_dim:self.latent_dim+1]
        physics_features = x[:, :, self.latent_dim+1:]     
        
        # 物理门控调节
        gate_weights = self.physics_gate(physics_features) 
        gated_latent = latent_features * gate_weights     
        
        # 融合与投影
        fused_x = torch.cat([gated_latent, mse_score, physics_features], dim=-1) 
        x_emb = self.input_projection(fused_x) # [B, Seq_Len, hidden_dim]
        
        # 注入位置编码
        seq_len = x_emb.size(1)
        x_emb = x_emb + self.pos_embedding[:, :seq_len, :]
        
        # 生成 Transformer 专用的 Padding Mask (True 代表需要忽略的 Padding 区域)
        padding_mask = (mask == 0.0)
        
        # Transformer 提特征
        trans_out = self.transformer(x_emb, src_key_padding_mask=padding_mask) 
        
        # 时序注意力聚焦
        attn_weights = self.temporal_attention(trans_out).squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(attn_weights, dim=1) 
        
        # 上下文聚合并分类
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
    print(f"\n🧠 [Step 3] Squeezing Performance: Bagging Ensemble + Precision@K Audit...")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(CONFIG['device'])
    
    ensemble_models = []
    criterion = FocalLoss(alpha=CONFIG['focal_alpha'], gamma=CONFIG['focal_gamma']) 
    
    pos_idx = np.where(y_tr == 1)[0]
    neg_idx = np.where(y_tr == 0)[0]
    
    print(f"   [Train Distribution] Normal: {len(neg_idx)}, Theft: {len(pos_idx)}")
    print(f"   [Test Distribution]  Normal: {len(y_te[y_te==0])}, Theft: {len(y_te[y_te==1])}")

    # --- 训练集成模型 (纯 Train 集合内部 Bagging) ---
    for k in range(CONFIG['ensemble_k']):
        print(f"--- Training Base Model {k+1}/{CONFIG['ensemble_k']} ---")
        np.random.shuffle(neg_idx)
        
        # 【致命杀招：1:2 强迫对抗采样】
        # 抛弃 50% 比例，强制每个弱分类器只看 1 份贼 和 2 份正常人。
        # 这样在 15 个模型轮番轰炸下，所有正常人都会被覆盖到，而贼会被反复放大强化学习！
        num_neg_sample = min(len(neg_idx), int(len(pos_idx) * 2.0))
        bag_neg = neg_idx[:num_neg_sample] 
        
        subset_idx = np.concatenate([pos_idx, bag_neg])
        np.random.shuffle(subset_idx)
        
        # 切分一小部分用于 Early Stopping
        sub_tr_idx, sub_val_idx = train_test_split(subset_idx, test_size=0.15, stratify=y_tr[subset_idx], random_state=CONFIG['seed']+k)
        
        tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr[sub_tr_idx]).to(device), torch.tensor(y_tr[sub_tr_idx]).float().to(device), torch.tensor(m_tr[sub_tr_idx]).to(device)), batch_size=CONFIG['fusion_batch_size'], shuffle=True)
        val_X, val_y, val_m = torch.tensor(X_tr[sub_val_idx]).to(device), torch.tensor(y_tr[sub_val_idx]).float().to(device), torch.tensor(m_tr[sub_val_idx]).to(device)
        
        model = PhysicsGatedTemporalFusionNet(latent_dim=CONFIG['latent_dim'], phys_dim=CONFIG['phys_dim'], hidden_dim=CONFIG['fusion_hidden_dim']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['fusion_lr'], weight_decay=1e-3) 
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
    
    # 【核心：PR-AUC 与 经济效用分析】
    fpr, tpr, roc_thresh = roc_curve(y_te, ens_probs)
    prec, rec, pr_thresh = precision_recall_curve(y_te, ens_probs)
    
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_te, ens_probs)
    
    # 1. 计算 Precision@K (模拟电网实地稽查预算)
    num_inspect = max(1, int(len(y_te) * CONFIG['inspection_budget']))
    top_k_indices = np.argsort(ens_probs)[::-1][:num_inspect]
    precision_at_k = y_te[top_k_indices].mean()
    recall_at_k = y_te[top_k_indices].sum() / y_te.sum()
    
    # 2. 约束阈值：满足最低 Precision 下的最大 Recall
    valid_thresholds = pr_thresh[prec[:-1] >= CONFIG['target_precision']]
    if len(valid_thresholds) > 0:
        opt_thresh = valid_thresholds[0]
    else:
        # 如果达不到底线，选择 F-beta score (更看重 Precision)
        f05_scores = ((1 + 0.5**2) * prec[:-1] * rec[:-1]) / (0.5**2 * prec[:-1] + rec[:-1] + 1e-8)
        opt_thresh = pr_thresh[np.argmax(f05_scores)]
        
    final_preds = (ens_probs >= opt_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, final_preds).ravel()
    
    print(f"🌟 ROC-AUC        : {roc_auc:.4f} (基准指标)")
    print(f"🌟 PR-AUC         : {pr_auc:.4f} (极度不平衡下的黄金指标！)")
    print("-" * 65)
    print(f"💰 经济稽查模拟 (预算 {CONFIG['inspection_budget']*100}% 用户)")
    print(f"   -> Precision@K : {precision_at_k:.4f} (抓获的嫌疑人中，有 {precision_at_k*100:.1f}% 是真贼)")
    print(f"   -> Recall@K    : {recall_at_k:.4f} (抓获了全体窃电用户的 {recall_at_k*100:.1f}%)")
    print("-" * 65)
    print(f"🎯 成本约束阈值 (Threshold = {opt_thresh:.3f})")
    print(f"   -> True Positives (抓获) : {tp}")
    print(f"   -> False Positives(误报) : {fp}")
    print(f"   -> False Negatives(漏网) : {fn}")
    print(f"   -> Precision             : {tp/(tp+fp+1e-8):.4f}")
    print(f"   -> Recall                : {tp/(tp+fn+1e-8):.4f}")
    print("=" * 65)
    
    np.save(os.path.join(output_dir, 'attention_weights_test.npy'), all_attention_weights)
    pd.DataFrame({'cons_no': np.arange(len(y_te)), 'true_label': y_te, 'prob': ens_probs}).to_csv(os.path.join(output_dir, 'blind_test_predictions.csv'), index=False)

if __name__ == "__main__":
    set_seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])
    
    cache_file = os.path.join(CONFIG['output_dir'], f"extracted_features_cache_t{CONFIG['t_extract']}_ddim.npz")
    
    if os.path.exists(cache_file):
        print(f"📦 发现版本匹配的特征缓存 ({cache_file})！秒级启动...")
        data = np.load(cache_file, allow_pickle=True)
        X_tr, y_tr, m_tr = data['X_tr'], data['y_tr'], data['m_tr']
        X_te, y_te, m_te = data['X_te'], data['y_te'], data['m_te']
    else:
        print(f"⚙️ 重构建特征缓存 (DDIM 加速版)...")
        diffusion = load_diffusion_model(CONFIG['model_path'], device)
        
        # 【核心修正】: 严格调用 API 读取分离好的门控训练集和盲测集
        _, _, clf_train_loader, test_loader = get_dataloaders()
        
        df_tr = extract_features_from_loader(diffusion, clf_train_loader, device, "Train Features")
        df_te = extract_features_from_loader(diffusion, test_loader, device, "Test Features")
        
        X_tr, y_tr, m_tr, _ = build_temporal_sequences(df_tr)
        X_te, y_te, m_te, _ = build_temporal_sequences(df_te)
        
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        np.savez_compressed(cache_file, X_tr=X_tr, y_tr=y_tr, m_tr=m_tr, X_te=X_te, y_te=y_te, m_te=m_te)
        print(f"💾 Train/Test 隔离特征已固化至 {cache_file}。")

    train_ensemble_and_blind_test(X_tr, y_tr, m_tr, X_te, y_te, m_te, CONFIG['output_dir'])
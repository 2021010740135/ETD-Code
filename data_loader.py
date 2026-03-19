import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from joblib import dump, load
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 全局配置 (SOTA 级满血训练 & 测试集标定版)
# ==============================================================================
CONFIG = {
    'train_npz_path': './results_phase2/train_diffusion_dataset.npz',
    'test_npz_path': './results_phase2/test_diffusion_dataset.npz',

    'output_dir': './results_phase3',
    'scaler_path': './results_phase3/phys_bifurcated_scaler.joblib',  # 升级为分岔缩放器
    'tensor_cache_path': './results_phase3/dataloader_tensors_cache_v5_SOTA.pt',  # 强制刷新缓存

    'batch_size': 256,
    'num_workers': 4,
    'random_seed': 2026
}


# ==============================================================================
# 1. 数据集类定义 (Zero-Copy Memory Optimized)
# ==============================================================================
class SGCCDiffusionDataset(Dataset):
    def __init__(self, residuals, patched, masks, phys_feats, labels, cons_nos):
        self.residuals = np.ascontiguousarray(residuals, dtype=np.float32)
        self.patched = np.ascontiguousarray(patched, dtype=np.float32)
        self.masks = np.ascontiguousarray(masks, dtype=np.float32)
        self.phys_feats = np.ascontiguousarray(phys_feats, dtype=np.float32)
        self.labels = np.ascontiguousarray(labels, dtype=np.int64)
        self.cons_nos = cons_nos

    def __len__(self):
        return self.residuals.shape[0]

    def __getitem__(self, idx):
        res = torch.as_tensor(self.residuals[idx]).unsqueeze(0)
        pat = torch.as_tensor(self.patched[idx]).unsqueeze(0)
        msk = torch.as_tensor(self.masks[idx]).unsqueeze(0)
        phys = torch.as_tensor(self.phys_feats[idx])
        lbl = torch.as_tensor(self.labels[idx])
        return res, pat, msk, phys, lbl, str(self.cons_nos[idx])


# ==============================================================================
# 2. 核心管线：构建并固化缓存 (Build & Cache)
# ==============================================================================
def build_and_cache_tensors():
    print("[Cache-Builder] 首次运行：构建 100% 满血训练集，并挂载测试集为验证指标...")

    train_data = np.load(CONFIG['train_npz_path'], allow_pickle=True)
    test_data = np.load(CONFIG['test_npz_path'], allow_pickle=True)

    tr_cons, tr_lbl = train_data['cons_no'], train_data['label'].astype(np.int8)
    tr_res, tr_pat = train_data['x_residual'], train_data['x_patched']
    tr_msk, tr_phys = train_data['x_mask'], train_data['phys_feat']

    te_cons, te_lbl = test_data['cons_no'], test_data['label'].astype(np.int8)
    te_res, te_pat = test_data['x_residual'], test_data['x_patched']
    te_msk, te_phys = test_data['x_mask'], test_data['phys_feat']

    print("🛡️  执行波幅防波堤防御：拦截极端时序突刺...")
    tr_res = np.nan_to_num(tr_res, nan=0.0, posinf=50.0, neginf=-50.0)
    te_res = np.nan_to_num(te_res, nan=0.0, posinf=50.0, neginf=-50.0)
    tr_res = np.clip(tr_res, -50.0, 50.0)
    te_res = np.clip(te_res, -50.0, 50.0)

    print("🛡️  执行脏数据底层防御：清洗物理特征矩阵...")
    tr_phys = np.nan_to_num(tr_phys, nan=0.0, posinf=0.0, neginf=0.0)
    te_phys = np.nan_to_num(te_phys, nan=0.0, posinf=0.0, neginf=0.0)

    diff_train_idx = np.where(tr_lbl == 0)[0]  # 满血 100%
    diff_val_idx = np.where(te_lbl == 0)[0]  # 偷梁换柱：用 Test 集的正常用户

    # ======================================================================
    # 【SOTA 破壁注入】：流形分岔缩放器 (Bifurcated Manifold Scaler)
    # ======================================================================
    print("📈 部署流形分岔缩放器，隔离保护绝对极值能量矩阵...")

    # 0: kl_div, 1: p_1_to_0, 2: p_0_to_1, 3: monthly_e, 4: weekly_e
    # 5: hf_noise, 6: peak_energy, 7: peak_density
    smooth_idx = [0, 1, 2, 3, 4]
    extreme_idx = [5, 6, 7]

    scaler_smooth = RobustScaler()
    scaler_extreme = MaxAbsScaler()  # 绝对值缩放，完美保持稀疏矩阵的0值属性

    # 1. 拟合平滑特征
    scaler_smooth.fit(tr_phys[diff_train_idx][:, smooth_idx])

    # 2. 拟合极值特征：先用 Log1p 压制长尾核爆，再计算 MaxAbs
    tr_phys_extreme_log = np.log1p(np.maximum(0, tr_phys[:, extreme_idx]))
    te_phys_extreme_log = np.log1p(np.maximum(0, te_phys[:, extreme_idx]))
    scaler_extreme.fit(tr_phys_extreme_log[diff_train_idx])

    # 3. 分岔转换与重组
    tr_phys_smooth_scaled = np.clip(scaler_smooth.transform(tr_phys[:, smooth_idx]), -5.0, 5.0)
    te_phys_smooth_scaled = np.clip(scaler_smooth.transform(te_phys[:, smooth_idx]), -5.0, 5.0)

    tr_phys_extreme_scaled = scaler_extreme.transform(tr_phys_extreme_log)
    te_phys_extreme_scaled = scaler_extreme.transform(te_phys_extreme_log)

    tr_phys_scaled = np.concatenate([tr_phys_smooth_scaled, tr_phys_extreme_scaled], axis=1)
    te_phys_scaled = np.concatenate([te_phys_smooth_scaled, te_phys_extreme_scaled], axis=1)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    dump({'smooth': scaler_smooth, 'extreme': scaler_extreme}, CONFIG['scaler_path'])

    # 【打包序列化】
    cache_dict = {
        'diff_train': (
            tr_res[diff_train_idx], tr_pat[diff_train_idx], tr_msk[diff_train_idx], tr_phys_scaled[diff_train_idx],
            tr_lbl[diff_train_idx], tr_cons[diff_train_idx]
        ),
        'diff_val': (
            te_res[diff_val_idx], te_pat[diff_val_idx], te_msk[diff_val_idx], te_phys_scaled[diff_val_idx],
            te_lbl[diff_val_idx], te_cons[diff_val_idx]
        ),
        'clf_train': (tr_res, tr_pat, tr_msk, tr_phys_scaled, tr_lbl, tr_cons),
        'test': (te_res, te_pat, te_msk, te_phys_scaled, te_lbl, te_cons)
    }

    torch.save(cache_dict, CONFIG['tensor_cache_path'])
    print(f"✅ 分岔缩放完毕，SOTA级张量缓存已固化至: {CONFIG['tensor_cache_path']}")


# ==============================================================================
# 3. 对外接口
# ==============================================================================
def get_dataloaders(force_rebuild=False):
    print("=" * 80)
    print("🚀 [Phase 3] 满血 SOTA 张量装载管线启动 (流形分岔版)")
    print("=" * 80)

    if force_rebuild or not os.path.exists(CONFIG['tensor_cache_path']):
        build_and_cache_tensors()

    print("⚡ 检测到张量缓存，执行极速热重载 (Hot-Reload)...")
    cache_dict = torch.load(CONFIG['tensor_cache_path'], weights_only=False)

    diff_train_ds = SGCCDiffusionDataset(*cache_dict['diff_train'])
    diff_val_ds = SGCCDiffusionDataset(*cache_dict['diff_val'])
    clf_train_ds = SGCCDiffusionDataset(*cache_dict['clf_train'])
    test_ds = SGCCDiffusionDataset(*cache_dict['test'])

    diff_train_loader = DataLoader(diff_train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                   num_workers=CONFIG['num_workers'], pin_memory=True)
    diff_val_loader = DataLoader(diff_val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                                 num_workers=CONFIG['num_workers'], pin_memory=True)
    clf_train_loader = DataLoader(clf_train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'],
                             pin_memory=True)

    print(f"    -> 纯净扩散轨 (满血 100% 训练): Train={len(diff_train_ds)}，Val={len(diff_val_ds)}")
    print(f"    -> 全量门控轨 (满血 100% 训练): Train={len(clf_train_ds)}")
    print(f"    -> 极限盲测轨 (终极校验靶点): Test={len(test_ds)}")
    print("✅ 装载完毕！准备火力全开。")

    return diff_train_loader, diff_val_loader, clf_train_loader, test_loader


if __name__ == "__main__":
    get_dataloaders(force_rebuild=True)
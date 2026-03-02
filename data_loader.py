# data_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 全局配置 (双轨隔离 & 缓存热重载版)
# ==============================================================================
CONFIG = {
    'train_npz_path': './results_phase2/train_diffusion_dataset.npz',
    'test_npz_path': './results_phase2/test_diffusion_dataset.npz',
    
    'output_dir': './results_phase3',
    'scaler_path': './results_phase3/phys_scaler.joblib',
    # 【新增】：张量缓存文件路径
    'tensor_cache_path': './results_phase3/dataloader_tensors_cache.pt', 
    
    'val_ratio_from_train': 0.15, 
    
    'batch_size': 128,  
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
    print("[Cache-Builder] 首次运行：读取原始 NPZ 并构建绝对纯净的张量切分...")
    
    train_data = np.load(CONFIG['train_npz_path'], allow_pickle=True)
    test_data = np.load(CONFIG['test_npz_path'], allow_pickle=True)

    tr_cons, tr_lbl = train_data['cons_no'], train_data['label'].astype(np.int8)
    tr_res, tr_pat = train_data['x_residual'], train_data['x_patched']
    tr_msk, tr_phys = train_data['x_mask'], train_data['phys_feat']

    te_cons, te_lbl = test_data['cons_no'], test_data['label'].astype(np.int8)
    te_res, te_pat = test_data['x_residual'], test_data['x_patched']
    te_msk, te_phys = test_data['x_mask'], test_data['phys_feat']

    pure_normal_idx = np.where(tr_lbl == 0)[0]
    np.random.seed(CONFIG['random_seed'])
    np.random.shuffle(pure_normal_idx)
    n_val = int(len(pure_normal_idx) * CONFIG['val_ratio_from_train'])
    
    diff_val_idx = pure_normal_idx[:n_val]
    diff_train_idx = pure_normal_idx[n_val:]

    scaler = StandardScaler()
    scaler.fit(tr_phys[diff_train_idx]) 
    tr_phys_scaled = np.clip(scaler.transform(tr_phys), -5.0, 5.0)
    te_phys_scaled = np.clip(scaler.transform(te_phys), -5.0, 5.0)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    dump(scaler, CONFIG['scaler_path'])

    # 【核心防御】：将处理好的张量字典打包序列化
    cache_dict = {
        'diff_train': (tr_res[diff_train_idx], tr_pat[diff_train_idx], tr_msk[diff_train_idx], tr_phys_scaled[diff_train_idx], tr_lbl[diff_train_idx], tr_cons[diff_train_idx]),
        'diff_val': (tr_res[diff_val_idx], tr_pat[diff_val_idx], tr_msk[diff_val_idx], tr_phys_scaled[diff_val_idx], tr_lbl[diff_val_idx], tr_cons[diff_val_idx]),
        'clf_train': (tr_res, tr_pat, tr_msk, tr_phys_scaled, tr_lbl, tr_cons),
        'test': (te_res, te_pat, te_msk, te_phys_scaled, te_lbl, te_cons)
    }
    
    torch.save(cache_dict, CONFIG['tensor_cache_path'])
    print(f"✅ 张量缓存已固化至: {CONFIG['tensor_cache_path']} (确保 Train/Inference 绝对对齐)")


# ==============================================================================
# 3. 对外接口 (热重载模式)
# ==============================================================================
def get_dataloaders(force_rebuild=False):
    """
    Train 和 Inference 脚本直接调用此接口。
    若缓存存在，0.1秒瞬间装载；若不存在，则构建缓存。
    """
    print("=" * 80)
    print("🚀 [Phase 3] 零污染张量装载管线启动 (Asymmetric Dual-Track Mode)")
    print("=" * 80)

    if force_rebuild or not os.path.exists(CONFIG['tensor_cache_path']):
        if not os.path.exists(CONFIG['train_npz_path']):
            raise FileNotFoundError("❌ 找不到 Phase2 的 Train 张量，请确认 Phase2 是否执行成功。")
        build_and_cache_tensors()

    print("⚡ 检测到张量缓存，执行极速热重载 (Hot-Reload)...")
    cache_dict = torch.load(CONFIG['tensor_cache_path'], weights_only=False)

    # 动态秒建 Dataset
    diff_train_ds = SGCCDiffusionDataset(*cache_dict['diff_train'])
    diff_val_ds = SGCCDiffusionDataset(*cache_dict['diff_val'])
    clf_train_ds = SGCCDiffusionDataset(*cache_dict['clf_train'])
    test_ds = SGCCDiffusionDataset(*cache_dict['test'])

    # 动态挂载 DataLoader (避开多进程 Pickle 陷阱)
    diff_train_loader = DataLoader(diff_train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    diff_val_loader = DataLoader(diff_val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    clf_train_loader = DataLoader(clf_train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    print(f"    -> 纯净扩散轨: Train={len(diff_train_ds)}，Val={len(diff_val_ds)}")
    print(f"    -> 全量门控轨: Train={len(clf_train_ds)}")
    print(f"    -> 极限盲测轨: Test={len(test_ds)}")
    print("✅ 装载完毕！准备火力全开。")
    
    return diff_train_loader, diff_val_loader, clf_train_loader, test_loader

if __name__ == "__main__":
    # 首次独立运行，生成缓存
    get_dataloaders(force_rebuild=True)
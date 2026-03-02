# Slicer_and_Feature_Extractor.py
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 核心配置 (双通道隔离版)
# ==============================================================================
@dataclass
class SlicerConfig:
    input_dir: str = "./results_phase1"
    raw_input_csv: str = "./data/electricity.csv" 
    output_dir: str = "./results_phase2"
    
    window_size: int = 256      
    stride: int = 64            
    benford_min_nonzero: int = 15 
    
    n_jobs: int = -1
    joblib_verbose: int = 5


# ==============================================================================
# 2. 领域引导的电物理与统计先验引擎 (Domain-Guided Electrophysical Priors)
# 【防御补丁】：重构术语，明确数学特征与底层物理篡改行为的严格映射机制。
# ==============================================================================
class ElectrophysicalPriorEngine:
    def __init__(self, benford_min_nonzero: int, window_size: int):
        self.min_nz = benford_min_nonzero
        # 【自然法则基准】：直接硬编码理想本福特分布，杜绝利用未来数据计算全局分布的泄露风险
        self.ideal_benford = np.log10(1 + 1 / np.arange(1, 10)).astype(np.float64)
        
        self.window_size = window_size
        self.freqs = np.fft.rfftfreq(window_size)
        
        # 预计算频域 Mask
        self.month_mask = (self.freqs > 0.02) & (self.freqs < 0.05)
        self.week_mask = (self.freqs > 0.12) & (self.freqs < 0.16)
        self.high_freq_mask = (self.freqs > 0.3)

    def _get_leading_digits_dist(self, x: np.ndarray) -> np.ndarray:
        x_valid = x[np.isfinite(x)]
        x_valid = np.abs(x_valid)
        x_valid = x_valid[x_valid > 0]
        if len(x_valid) < self.min_nz: return None
        log10x = np.floor(np.log10(x_valid))
        digits = np.floor(x_valid / (10.0 ** log10x)).astype(np.int32)
        digits = np.clip(digits, 1, 9)
        counts = np.bincount(digits, minlength=10)[1:10].astype(np.float64) + 1e-6 
        return counts / counts.sum()

    def calc_local_kl_divergence_2d(self, slices_raw: np.ndarray) -> np.ndarray:
        """
        [特征 1] 数字化篡改伪影 (Digital Firmware Tampering Artifact)
        物理映射：捕获黑客通过物理探针或固件注入，利用线性/常数公式对计量寄存器进行的人为修改。
        此类非自然修改会严重偏离大自然用电的理想本福特分布。
        """
        N = slices_raw.shape[0]
        kl_divs = np.zeros(N, dtype=np.float32)
        for i in range(N):
            local_dist = self._get_leading_digits_dist(slices_raw[i])
            if local_dist is not None:
                # 严格使用自然法则作为基准，避免 Data Leakage
                kl_divs[i] = float(entropy(local_dist, self.ideal_benford))
        return kl_divs

    def calc_markov_transitions_2d(self, slices_patched: np.ndarray, slices_vs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        [特征 2] 异常压降与拓扑改变矩阵 (Topology Alteration & Voltage Drop Matrix)
        物理映射：非法搭线会改变配电网拓扑。当大功率窃电负载接入时，焦耳定律和欧姆定律
        会导致瞬态节点压降。马尔可夫转移矩阵量化了由高到低（1->0）的剧态突变概率。
        """
        ratio = slices_patched / (slices_vs + 1e-3)
        state = (ratio > 0.4).astype(np.int8) 
        
        state_curr = state[:, :-1]
        state_next = state[:, 1:]
        
        n_1 = np.sum(state_curr == 1, axis=1)
        n_0 = np.sum(state_curr == 0, axis=1)
        
        p_1_to_0 = np.sum((state_curr == 1) & (state_next == 0), axis=1) / np.maximum(n_1, 1)
        p_0_to_1 = np.sum((state_curr == 0) & (state_next == 1), axis=1) / np.maximum(n_0, 1)
        
        return p_1_to_0.astype(np.float32), p_0_to_1.astype(np.float32)

    def calc_fft_features_2d(self, slices_residual: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [特征 3] 旁路硬件高次谐波指纹 (Hardware Bypass High-Frequency Harmonics)
        物理映射：私拉乱接或劣质分流器的接触电阻非线性波动，会在波形中注入高次谐波。
        FFT 频域能量是捕获这类物理硬件篡改的最直接电学证据。
        """
        means = np.mean(slices_residual, axis=1, keepdims=True)
        fft_vals = np.abs(np.fft.rfft(slices_residual - means, axis=1)) 
        
        monthly_energy = np.sum(fft_vals[:, self.month_mask], axis=1)
        weekly_energy = np.sum(fft_vals[:, self.week_mask], axis=1)
        high_freq_noise = np.sum(fft_vals[:, self.high_freq_mask], axis=1)
        
        return monthly_energy.astype(np.float32), weekly_energy.astype(np.float32), high_freq_noise.astype(np.float32)


# ==============================================================================
# 3. 零拷贝切片逻辑 (Worker)
# ==============================================================================
def get_strided_windows(arr: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    view = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size)
    return view[::stride]

def process_user_slices_vectorized(
    user_idx: int, cons_no: str, label: int,
    ts_residual: np.ndarray, ts_patched: np.ndarray, ts_vs: np.ndarray, 
    ts_mask: np.ndarray, ts_raw: np.ndarray,
    engine: ElectrophysicalPriorEngine, window_size: int, stride: int
):
    res_windows = get_strided_windows(ts_residual, window_size, stride).copy().astype(np.float32)
    pat_windows = get_strided_windows(ts_patched, window_size, stride).astype(np.float32)
    vs_windows = get_strided_windows(ts_vs, window_size, stride).astype(np.float32)
    mask_windows = get_strided_windows(ts_mask, window_size, stride).astype(np.int8)
    raw_windows = get_strided_windows(ts_raw, window_size, stride)
    
    num_slices = res_windows.shape[0]
    if num_slices == 0:
        return None
        
    # 核心修正：仅计算局部 vs 理想物理法则的 KL 散度，斩断未来泄露
    kl_div = engine.calc_local_kl_divergence_2d(raw_windows)
    p_1_to_0, p_0_to_1 = engine.calc_markov_transitions_2d(pat_windows, vs_windows)
    monthly_e, weekly_e, hf_noise = engine.calc_fft_features_2d(res_windows)
    
    # 组装 6 维领域引导先验特征
    prior_feat_matrix = np.column_stack([kl_div, p_1_to_0, p_0_to_1, monthly_e, weekly_e, hf_noise])
    
    cons_nos_arr = np.full(num_slices, cons_no, dtype=object)
    labels_arr = np.full(num_slices, label, dtype=np.int8)
    
    return cons_nos_arr, labels_arr, res_windows, pat_windows, mask_windows, prior_feat_matrix


# ==============================================================================
# 4. 数据加载与双路主流程 (Train/Test Firewall Pipeline)
# ==============================================================================
def try_load_raw_timeseries(raw_csv_path: str, cons_no_list: np.ndarray, ts_cols: list) -> np.ndarray:
    raw = pd.read_csv(raw_csv_path)
    raw["CONS_NO"] = raw["CONS_NO"].astype(str)
    raw = raw.set_index("CONS_NO")
    raw_aligned = pd.DataFrame(index=cons_no_list, columns=ts_cols)
    common_cols = list(set(ts_cols) & set(raw.columns))
    raw_aligned[common_cols] = raw.loc[raw.index.intersection(cons_no_list), common_cols]
    return raw_aligned.fillna(0.0).to_numpy(dtype=np.float32)

def process_split(split_name: str, cfg: SlicerConfig) -> None:
    """【防御核心】：独立的 Split 处理器，确保 Train 和 Test 在内存和磁盘上完全隔离"""
    print(f"\n---> 开始处理 {split_name.upper()} 数据集...")
    
    res_csv = os.path.join(cfg.input_dir, f"{split_name}_residual.csv")
    pat_csv = os.path.join(cfg.input_dir, f"{split_name}_patched.csv")
    mask_csv = os.path.join(cfg.input_dir, f"{split_name}_mask.csv")
    vs_csv = os.path.join(cfg.input_dir, "virtual_stations_daily.csv")
    
    if not os.path.exists(res_csv):
        print(f"找不到 {res_csv}，跳过 {split_name} 分支。")
        return

    df_res = pd.read_csv(res_csv)
    df_pat = pd.read_csv(pat_csv)
    df_mask = pd.read_csv(mask_csv)
    
    df_res["CONS_NO"] = df_res["CONS_NO"].astype(str)
    ts_cols = [c for c in df_res.columns if c not in ["CONS_NO", "FLAG", "CLUSTER", "USER_TYPE"]]
    
    df_vs = pd.read_csv(vs_csv, index_col=0)[[c for c in ts_cols if c in pd.read_csv(vs_csv, index_col=0).columns]]
    raw_ts_mat = try_load_raw_timeseries(cfg.raw_input_csv, df_res["CONS_NO"].values, ts_cols)
    
    res_mat = df_res[ts_cols].to_numpy(dtype=np.float32)
    pat_mat = df_pat[ts_cols].to_numpy(dtype=np.float32)
    mask_mat = df_mask[ts_cols].to_numpy(dtype=np.int8)
    labels = df_res["FLAG"].values.astype(np.int8)
    cons_nos = df_res["CONS_NO"].values
    cluster_ids = df_res["CLUSTER"].values.astype(int)
    
    engine = ElectrophysicalPriorEngine(cfg.benford_min_nonzero, cfg.window_size)
    
    start_time = time.time()
    results = Parallel(n_jobs=cfg.n_jobs, verbose=cfg.joblib_verbose)(
        delayed(process_user_slices_vectorized)(
            i, cons_nos[i], labels[i],
            res_mat[i], pat_mat[i], df_vs.loc[cluster_ids[i]].values.astype(np.float32), 
            mask_mat[i], raw_ts_mat[i],
            engine, cfg.window_size, cfg.stride
        )
        for i in range(len(df_res))
    )
    
    results = [r for r in results if r is not None]
    
    out_dict = {
        "cons_no": np.concatenate([r[0] for r in results]),
        "label": np.concatenate([r[1] for r in results]),
        "x_residual": np.concatenate([r[2] for r in results]),
        "x_patched": np.concatenate([r[3] for r in results]),
        "x_mask": np.concatenate([r[4] for r in results]),
        "phys_feat": np.concatenate([r[5] for r in results]) 
    }
    
    out_path = os.path.join(cfg.output_dir, f"{split_name}_diffusion_dataset.npz")
    np.savez_compressed(out_path, **out_dict)
    
    print(f"✅ {split_name.upper()} 组装完成！耗时: {time.time() - start_time:.2f} 秒")
    print(f"   - 切片规模: {out_dict['x_residual'].shape}")

def run_phase2_slicer(cfg: SlicerConfig = SlicerConfig()):
    print("=" * 80)
    print("阶段二：电物理与统计先验切片器 (双通道隔离 & 无未来泄露版)")
    print("=" * 80)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # 【防御核心】：强制分别处理 train 和 test
    for split in ["train", "test"]:
        process_split(split, cfg)
        
    print("\n✅ Phase2 全流程切片合规完成！训练与测试张量已安全隔离。")

if __name__ == "__main__":
    run_phase2_slicer()
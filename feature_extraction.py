from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ==============================================================================
# 1. 核心配置
# ==============================================================================
@dataclass
class SlicerConfig:
    input_dir: str = "./results_phase1"
    raw_input_csv: str = "./data/sgcc_raw.csv"
    output_dir: str = "./results_phase2"

    window_size: int = 256
    stride: int = 64
    benford_min_nonzero: int = 15
    batch_size: int = 2048


# ==============================================================================
# 2. 领域引导的电物理先验引擎
# ==============================================================================
class ElectrophysicalPriorEngineTensor:
    def __init__(self, benford_min_nonzero: int, window_size: int, device: torch.device):
        self.device = device
        self.min_nz = benford_min_nonzero
        self.eps = 1e-8

        # 理想本福特分布基准，常驻显存
        ideal_benford_np = np.log10(1 + 1 / np.arange(1, 10)).astype(np.float32)
        self.ideal_benford = torch.tensor(ideal_benford_np, device=device)

        self.window_size = window_size
        self.freqs = torch.fft.rfftfreq(window_size).to(device)

        # 预计算频域 Mask
        self.month_mask = (self.freqs > 0.02) & (self.freqs < 0.05)
        self.week_mask = (self.freqs > 0.12) & (self.freqs < 0.16)
        self.high_freq_mask = (self.freqs > 0.3)

    def calc_benford_kl_divergence_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        全量 GPU 并行计算 Batch 内所有切片的 KL 散度。
        x shape: [Batch, Window_Size]
        """
        B, W = x.shape
        kl_divs = torch.zeros(B, dtype=torch.float32, device=self.device)

        # 提取有效数字 (GPU 并行)
        x_abs = torch.abs(x)
        valid_mask = torch.isfinite(x_abs) & (x_abs > 0)

        for i in range(B):
            row_valid = x_abs[i][valid_mask[i]]
            if len(row_valid) < self.min_nz:
                continue

            log10x = torch.floor(torch.log10(row_valid))
            digits = torch.floor(row_valid / (10.0 ** log10x)).to(torch.int64)
            digits = torch.clamp(digits, 1, 9)

            counts = torch.bincount(digits, minlength=10)[1:10].float() + self.eps
            local_dist = counts / counts.sum()

            # 加入 eps 防止极大异常值导致的极窄分布产生 log(0) 崩溃
            kl_divs[i] = torch.sum(local_dist * torch.log((local_dist + self.eps) / (self.ideal_benford + self.eps)))

        return kl_divs

    def calc_markov_transitions_batch(self, slices_patched: torch.Tensor, slices_vs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        ratio = slices_patched / (slices_vs + self.eps)
        state = (ratio > 0.4).to(torch.int8)

        state_curr = state[:, :-1]
        state_next = state[:, 1:]

        n_1 = torch.sum(state_curr == 1, dim=1)
        n_0 = torch.sum(state_curr == 0, dim=1)

        p_1_to_0 = torch.sum((state_curr == 1) & (state_next == 0), dim=1).float() / torch.clamp(n_1.float(), min=1.0)
        p_0_to_1 = torch.sum((state_curr == 0) & (state_next == 1), dim=1).float() / torch.clamp(n_0.float(), min=1.0)

        return p_1_to_0, p_0_to_1

    def calc_fft_features_batch(self, slices_residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = torch.mean(slices_residual, dim=1, keepdim=True)
        # GPU 上满血并发傅里叶变换
        fft_vals = torch.abs(torch.fft.rfft(slices_residual - means, dim=1))

        monthly_energy = torch.sum(fft_vals[:, self.month_mask], dim=1)
        weekly_energy = torch.sum(fft_vals[:, self.week_mask], dim=1)
        high_freq_noise = torch.sum(fft_vals[:, self.high_freq_mask], dim=1)

        return monthly_energy, weekly_energy, high_freq_noise

    # 【SOTA 破壁注入】：新增极值脉冲能量与密度计算
    def calc_peak_features_batch(self, slices_peak: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取绝对极值残差的时域脉冲能量 (L2 Norm) 与 高频突刺密度。
        这是区分合法大户与窃电贼的决定性物理先验。
        """
        # 1. 脉冲爆发总能量：对极大异常值更敏感
        peak_energy = torch.norm(slices_peak, p=2, dim=1)

        # 2. 突刺密度：在窗口内发生异常脉冲的频率，eps 设为 1e-3 滤除浮点误差
        peak_density = torch.sum(slices_peak > 1e-3, dim=1).float() / self.window_size

        return peak_energy, peak_density


# ==============================================================================
# 3. 数据加载与主流程 (纯血 PyTorch 批处理版)
# ==============================================================================
def try_load_raw_timeseries(raw_csv_path: str, cons_no_list: np.ndarray, ts_cols: list) -> np.ndarray:
    raw = pd.read_csv(raw_csv_path)
    raw["CONS_NO"] = raw["CONS_NO"].astype(str)
    raw = raw.set_index("CONS_NO")
    raw_aligned = pd.DataFrame(index=cons_no_list, columns=ts_cols)
    common_cols = list(set(ts_cols) & set(raw.columns))
    raw_aligned[common_cols] = raw.loc[raw.index.intersection(cons_no_list), common_cols]
    return raw_aligned.fillna(0.0).to_numpy(dtype=np.float32)


def generate_tensor_windows(tensor: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """使用 PyTorch unfold 在 GPU 显存内瞬间完成零拷贝滑动窗口切片"""
    return tensor.unfold(dimension=1, size=window_size, step=stride)


def process_split(split_name: str, cfg: SlicerConfig, device: torch.device) -> None:
    print(f"\n---> 开始处理 {split_name.upper()} 数据集...")

    res_csv = os.path.join(cfg.input_dir, f"{split_name}_residual.csv")
    pat_csv = os.path.join(cfg.input_dir, f"{split_name}_patched.csv")
    mask_csv = os.path.join(cfg.input_dir, f"{split_name}_mask.csv")
    peak_csv = os.path.join(cfg.input_dir, f"{split_name}_peak.csv")  # 新增极值加载路径
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

    # 加载常规矩阵
    res_mat = df_res[ts_cols].to_numpy(dtype=np.float32)
    pat_mat = df_pat[ts_cols].to_numpy(dtype=np.float32)
    mask_mat = df_mask[ts_cols].to_numpy(dtype=np.float32)

    # 动态加载极值矩阵（兼容 Phase 1 是否输出）
    if os.path.exists(peak_csv):
        df_peak = pd.read_csv(peak_csv)
        peak_mat = df_peak[ts_cols].to_numpy(dtype=np.float32)
    else:
        print(f"⚠️ [警告] 找不到 {peak_csv}，正在构造全零矩阵以维持张量正交维度...")
        peak_mat = np.zeros_like(res_mat, dtype=np.float32)

    labels = df_res["FLAG"].values.astype(np.int8)
    cons_nos = df_res["CONS_NO"].values

    # 映射出虚拟气象站矩阵
    cluster_ids = df_res["CLUSTER"].values.astype(int)
    vs_mat = df_vs.loc[cluster_ids].values.astype(np.float32)

    num_users = len(df_res)
    engine = ElectrophysicalPriorEngineTensor(cfg.benford_min_nonzero, cfg.window_size, device)

    # 结果收集器
    all_cons_nos, all_labels = [], []
    all_res_win, all_pat_win, all_mask_win, all_phys_feat = [], [], [], []

    start_time = time.time()

    # 【GPU 批处理核心】：以 batch_size 为单位，将用户推入显存
    for i in tqdm(range(0, num_users, cfg.batch_size), desc="GPU Tensor Slicing"):
        end_idx = min(i + cfg.batch_size, num_users)

        # PUSH to GPU
        b_res = torch.tensor(res_mat[i:end_idx], device=device)
        b_pat = torch.tensor(pat_mat[i:end_idx], device=device)
        b_mask = torch.tensor(mask_mat[i:end_idx], device=device)
        b_raw = torch.tensor(raw_ts_mat[i:end_idx], device=device)
        b_vs = torch.tensor(vs_mat[i:end_idx], device=device)
        b_peak = torch.tensor(peak_mat[i:end_idx], device=device)  # 极值入列

        # GPU 内部瞬间切片 [Batch, Num_Windows, Window_Size]
        w_res = generate_tensor_windows(b_res, cfg.window_size, cfg.stride)
        w_pat = generate_tensor_windows(b_pat, cfg.window_size, cfg.stride)
        w_mask = generate_tensor_windows(b_mask, cfg.window_size, cfg.stride)
        w_raw = generate_tensor_windows(b_raw, cfg.window_size, cfg.stride)
        w_vs = generate_tensor_windows(b_vs, cfg.window_size, cfg.stride)
        w_peak = generate_tensor_windows(b_peak, cfg.window_size, cfg.stride)  # 极值切片

        B_users, num_win, W_size = w_res.shape
        total_slices = B_users * num_win

        # 将矩阵展平为 [Total_Slices, Window_Size] 以便送入引擎运算
        flat_res = w_res.reshape(total_slices, W_size)
        flat_pat = w_pat.reshape(total_slices, W_size)
        flat_mask = w_mask.reshape(total_slices, W_size)
        flat_raw = w_raw.reshape(total_slices, W_size)
        flat_vs = w_vs.reshape(total_slices, W_size)
        flat_peak = w_peak.reshape(total_slices, W_size)  # 极值展平

        # 执行显存内加速特征提取
        with torch.no_grad():
            kl_div = engine.calc_benford_kl_divergence_batch(flat_raw)
            p_1_to_0, p_0_to_1 = engine.calc_markov_transitions_batch(flat_pat, flat_vs)
            monthly_e, weekly_e, hf_noise = engine.calc_fft_features_batch(flat_res)

            # 【SOTA 破壁注入】：提取极值先验
            peak_energy, peak_density = engine.calc_peak_features_batch(flat_peak)

            # 组装物理特征张量 [Total_Slices, 8]  <-- 升维锁定
            phys_tensor = torch.stack([
                kl_div, p_1_to_0, p_0_to_1,
                monthly_e, weekly_e, hf_noise,
                peak_energy, peak_density
            ], dim=1)

        # 构造对应的标签和用户 ID 数组
        batch_cons = cons_nos[i:end_idx]
        batch_labels = labels[i:end_idx]

        # 广播 ID 和 Label 匹配切片数量
        expanded_cons = np.repeat(batch_cons, num_win)
        expanded_labels = np.repeat(batch_labels, num_win)

        # PULL back to CPU memory (防 OOM)
        all_cons_nos.append(expanded_cons)
        all_labels.append(expanded_labels)
        all_res_win.append(flat_res.cpu().numpy())
        all_pat_win.append(flat_pat.cpu().numpy())
        all_mask_win.append(flat_mask.cpu().numpy().astype(np.int8))
        all_phys_feat.append(phys_tensor.cpu().numpy())

    # 最终合并
    out_dict = {
        "cons_no": np.concatenate(all_cons_nos),
        "label": np.concatenate(all_labels),
        "x_residual": np.concatenate(all_res_win),
        "x_patched": np.concatenate(all_pat_win),
        "x_mask": np.concatenate(all_mask_win),
        "phys_feat": np.concatenate(all_phys_feat)
    }

    out_path = os.path.join(cfg.output_dir, f"{split_name}_diffusion_dataset.npz")
    np.savez_compressed(out_path, **out_dict)

    print(f"✅ {split_name.upper()} 组装完成！耗时: {time.time() - start_time:.2f} 秒")
    print(f"成功提取切片规模: {out_dict['x_residual'].shape}，物理特征升维至 8。")


def run_phase2_slicer(cfg: SlicerConfig = SlicerConfig()):
    print("=" * 80)
    print("🚀 阶段二：电物理与统计先验切片器 (极值正交增强版)")
    print("=" * 80)

    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for split in ["train", "test"]:
        process_split(split, cfg, device)

    print("\n✅ Phase2 完成！流形底座已构建完毕。")


if __name__ == "__main__":
    run_phase2_slicer()
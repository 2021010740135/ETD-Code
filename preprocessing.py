# preprocessing.py
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import json
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# ==============================================================================
# 1. 统一配置类 (Configuration)
# ==============================================================================
@dataclass
class GlobalConfig:
    """包含清洗参数 + 聚类解耦参数"""
    # 删除了导致样本丢弃的阈值参数，仅保留基础处理参数
    pad_days: int = 2
    pad_mode: str = "zero"

    robust_scale_epsilon: float = 10.0
    cluster_num: int = 5
    random_state: int = 42

    # 【防御补丁】：显式设定测试集比例，切断泄露源头
    test_size: float = 0.2


# ==============================================================================
# 2. 清洗引擎 (SGCCCleaner - 零丢弃版)
# 说明：不再执行任何样本级的删除操作，将所有“缺失”、“全0”、“极值”作为物理篡改特征保留。
# ==============================================================================
class SGCCCleaner:
    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self.data: pd.DataFrame | None = None
        self.mask_df: pd.DataFrame | None = None
        self.labels: pd.Series | None = None
        self.cons_ids: pd.Series | None = None
        self.ts_cols: List[str] = []
        self.report: Dict[str, Any] = {}

    def load_data(self, file_path: str) -> None:
        print(f"[1/Clean] 加载数据 (Offline Auditing Mode): {file_path}")
        raw = pd.read_csv(file_path)
        
        # 【核心修复】：防止 FLAG 列中存在 NaN 导致新版 Pandas 转换 np.int8 时报错崩溃
        # 如果标签缺失，默认将其视作正常用户 (0)
        self.labels = raw["FLAG"].fillna(0).astype(np.int8)
        
        self.cons_ids = raw["CONS_NO"].astype(str)

        self.ts_cols = [c for c in raw.columns if c not in ("FLAG", "CONS_NO")]
        try:
            self.ts_cols = sorted(self.ts_cols, key=lambda x: pd.to_datetime(x))
        except Exception:
            pass

        self.data = raw[self.ts_cols].copy()
        self.report["input_shape"] = raw.shape

    def _impute_missing_values(self) -> None:
        # 1. 绝对优先：在任何数据插值之前，先精准记录原始缺失情况！
        # 缺失=0，观测=1。这个 Mask 矩阵将成为捕获“暴力断电”窃电手法的核心证据。
        self.mask_df = (~self.data.isna()).astype(np.int8)

        # 2. 基础数值填充（仅为了让下游 KMeans 和 NN 计算不报 NaN 错误）
        # 扩散模型在计算 Loss 时，会通过 Mask 屏蔽掉这些插值出来的虚假数据。
        df = self.data.interpolate(method="linear", axis=1, limit=2, limit_direction="both")
        self.data = df.fillna(0.0).astype(np.float32)

    def _pad_sequence(self) -> None:
        if self.cfg.pad_days <= 0: return
        last_dt = pd.to_datetime(self.data.columns[-1])
        for i in range(1, self.cfg.pad_days + 1):
            new_col = (last_dt + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            self.data[new_col] = 0.0
            self.mask_df[new_col] = 0  # 填充的数据依然标记为不可信 (0)
        self.ts_cols = list(self.data.columns)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        print("[2/Clean] 执行零丢弃策略 (Zero-Drop Strategy)... 保留所有极值与断网缺失特征。")
        self._impute_missing_values()
        self._pad_sequence()
        return self.data, self.mask_df, self.labels, self.cons_ids, self.report


# ==============================================================================
# 3. 聚类引擎与环境解耦 (SGCCClusterer)
# ==============================================================================
class SGCCClusterer:
    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self.scaler: StandardScaler | None = None
        self.kmeans: KMeans | None = None
        self.virtual_stations: pd.DataFrame | None = None
        self.report: Dict[str, Any] = {}

    def _get_shape_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        X = df.to_numpy()
        p95_vals = np.percentile(X, 95, axis=1, keepdims=True)
        scales = np.maximum(p95_vals, self.cfg.robust_scale_epsilon)
        X_norm = np.clip(X / scales, 0.0, 2.0).astype(np.float32)
        norm_df = pd.DataFrame(X_norm, columns=df.columns)

        features = pd.DataFrame({
            'mean': norm_df.mean(axis=1),
            'std': norm_df.std(axis=1),
            'q90': norm_df.quantile(0.9, axis=1),
            'q10': norm_df.quantile(0.1, axis=1)
        }).fillna(0)

        return features.to_numpy(), norm_df, scales

    def fit(self, train_data: pd.DataFrame) -> None:
        print("[3/Cluster-FIT] 仅在训练集上锁定全局统计流形特征，杜绝测试集穿越...")

        train_features, train_norm_df, _ = self._get_shape_features(train_data)

        self.scaler = StandardScaler()
        train_feat_scaled = self.scaler.fit_transform(train_features)

        self.kmeans = KMeans(n_clusters=self.cfg.cluster_num, random_state=self.cfg.random_state, n_init=10)
        train_cluster_labels = self.kmeans.fit_predict(train_feat_scaled)

        print("[4/Cluster-FIT] 构建训练集专属的归一化中位数虚拟气象站...")
        df_with_cluster = train_norm_df.copy()
        df_with_cluster["cluster"] = train_cluster_labels
        self.virtual_stations = df_with_cluster.groupby("cluster").median()

        self.report["is_fitted"] = True
        self.report["train_scaler_mean"] = self.scaler.mean_.tolist()

    def transform(self, data: pd.DataFrame, mask_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        if self.kmeans is None or self.virtual_stations is None:
            raise ValueError("Clusterer must be fitted with train data before calling transform!")

        print("[5/Cluster-TRANSFORM] 应用已锁定的物理基准线进行向量化环境剥离...")

        features, norm_df, scales = self._get_shape_features(data)
        feat_scaled = self.scaler.transform(features)

        cluster_labels = self.kmeans.predict(feat_scaled)

        X_raw_np = data.to_numpy()
        mask_np = mask_df.to_numpy()

        base_curves_norm = self.virtual_stations.loc[cluster_labels].to_numpy()
        base_curves_physical = (base_curves_norm * scales).astype(np.float32)

        X_patched_np = np.where(mask_np == 1, X_raw_np, base_curves_physical).astype(np.float32)
        X_residual_np = (X_patched_np - base_curves_physical).astype(np.float32)

        df_patched = pd.DataFrame(X_patched_np, columns=data.columns)
        df_residual = pd.DataFrame(X_residual_np, columns=data.columns)

        return df_patched, df_residual, cluster_labels


# ==============================================================================
# 4. 主流程入口 (Main)
# ==============================================================================
if __name__ == "__main__":
    INPUT_FILE = "./data/electricity.csv"
    OUTPUT_DIR = "./results_phase1"

    if not os.path.exists(INPUT_FILE):
        print("Creating dummy data for testing...")
        os.makedirs("./data", exist_ok=True)
        dates = pd.date_range("2014-01-01", periods=1035, freq="D")
        n_users = 200
        data = np.random.rand(n_users, 1035) * 100
        data[np.random.rand(*data.shape) < 0.25] = np.nan
        df = pd.DataFrame(data, columns=dates.strftime("%Y-%m-%d"))
        df.insert(0, "CONS_NO", [f"U{i}" for i in range(n_users)])
        labels = np.zeros(n_users)
        labels[:int(n_users * 0.085)] = 1
        df.insert(0, "FLAG", labels.astype(np.int8))
        df.to_csv(INPUT_FILE, index=False)

    cfg = GlobalConfig()

    cleaner = SGCCCleaner(cfg)
    cleaner.load_data(INPUT_FILE)
    clean_data, mask_data, labels, cons_ids, report = cleaner.run()

    print("\n[Firewall] 执行严格的 Train/Test 隔离 (Stratified Split)...")
    idx_train, idx_test = train_test_split(
        np.arange(len(clean_data)),
        test_size=cfg.test_size,
        stratify=labels,
        random_state=cfg.random_state
    )

    X_train = clean_data.iloc[idx_train].reset_index(drop=True)
    mask_train = mask_data.iloc[idx_train].reset_index(drop=True)
    y_train = labels.iloc[idx_train].reset_index(drop=True)
    cons_train = cons_ids.iloc[idx_train].reset_index(drop=True)

    X_test = clean_data.iloc[idx_test].reset_index(drop=True)
    mask_test = mask_data.iloc[idx_test].reset_index(drop=True)
    y_test = labels.iloc[idx_test].reset_index(drop=True)
    cons_test = cons_ids.iloc[idx_test].reset_index(drop=True)

    clusterer = SGCCClusterer(cfg)
    clusterer.fit(X_train)

    patched_train, residual_train, cluster_train = clusterer.transform(X_train, mask_train)
    patched_test, residual_test, cluster_test = clusterer.transform(X_test, mask_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[Save] 序列化输出，严格区分 Train/Test 存储至 {OUTPUT_DIR}...")

    def save_split(split_name, cons, y, c_labels, patched, residual, mask):
        meta = pd.DataFrame({"CONS_NO": cons, "FLAG": y, "CLUSTER": c_labels})
        pd.concat([meta, patched], axis=1).to_csv(os.path.join(OUTPUT_DIR, f"{split_name}_patched.csv"), index=False)
        pd.concat([meta, residual], axis=1).to_csv(os.path.join(OUTPUT_DIR, f"{split_name}_residual.csv"), index=False)
        pd.concat([meta[["CONS_NO"]], mask], axis=1).to_csv(os.path.join(OUTPUT_DIR, f"{split_name}_mask.csv"), index=False)

    save_split("train", cons_train, y_train, cluster_train, patched_train, residual_train, mask_train)
    save_split("test", cons_test, y_test, cluster_test, patched_test, residual_test, mask_test)
    clusterer.virtual_stations.to_csv(os.path.join(OUTPUT_DIR, "virtual_stations_daily.csv"))

    print("✅ Phase 1 绝对物理流形保护级重构完成！极值与缺失特征已全量保留。")
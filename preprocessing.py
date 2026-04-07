# preprocessing.py
from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Configuration
# ==============================================================================
@dataclass
class GlobalConfig:
    input_file: str = "./data/electricity.csv"
    output_dir: str = "./results_phase1"

    # NaN and zero-value thresholds
    min_valid_days: int = 30
    max_zero_ratio: float = 0.95 

    # Train-test split configuration
    test_size: float = 0.20
    random_state: int = 58

# ==============================================================================
# 2. Preprocessing Engine
# ==============================================================================
class SGCCMinimalPreprocessor:
    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self.ts_cols = []
        self.report: Dict[str, Any] = {}

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print(f"Loading data: {self.cfg.input_file}")
        raw = pd.read_csv(self.cfg.input_file)

        if not {"CONS_NO", "FLAG"}.issubset(raw.columns):
            raise ValueError("Columns CONS_NO and FLAG are required.")

        raw = raw.dropna(subset=["FLAG"]).copy()
        raw["FLAG"] = raw["FLAG"].astype(int)
        raw["CONS_NO"] = raw["CONS_NO"].astype(str)

        self.ts_cols = [c for c in raw.columns if c not in ("CONS_NO", "FLAG")]
        try:
            self.ts_cols = sorted(self.ts_cols, key=lambda x: pd.to_datetime(x))
        except Exception:
            pass

        df_ts = raw[self.ts_cols].copy()
        initial_count = len(df_ts)

        # Filtering logic: Shape (N,)
        valid_counts = df_ts.notna().sum(axis=1)
        keep_mask_1 = valid_counts >= self.cfg.min_valid_days

        zero_counts = (df_ts < 0.01).sum(axis=1)
        keep_mask_2 = (zero_counts / len(self.ts_cols)) <= self.cfg.max_zero_ratio

        final_keep = keep_mask_1 & keep_mask_2
        df_ts = df_ts[final_keep].reset_index(drop=True)
        
        # Labels and IDs: Shape (N_filtered,)
        labels = raw.loc[final_keep, "FLAG"].reset_index(drop=True).to_numpy(dtype=np.int8)
        cons_ids = raw.loc[final_keep, "CONS_NO"].reset_index(drop=True).to_numpy()

        dropped_count = initial_count - len(df_ts)
        print(f"Dropped samples: {dropped_count}")

        # Mask extraction: Shape (N_filtered, T_steps), 1.0 for valid, 0.0 for NaN
        mask_np = (~df_ts.isna()).to_numpy(dtype=np.float32)

        # Imputation: x_{t} = x_{t-7}
        df_shifted = df_ts.shift(7, axis=1)
        df_imputed = df_ts.fillna(df_shifted)
        df_imputed = df_imputed.fillna(0.0)
        raw_np = df_imputed.to_numpy(dtype=np.float32)

        # Log1p transformation: Shape (N_filtered, T_steps)
        x_log = np.log1p(raw_np)

        self.report = {
            "initial_samples": initial_count,
            "dropped_samples": int(dropped_count),
            "final_samples": len(df_ts),
            "num_time_steps": len(self.ts_cols),
            "num_positive": int(np.sum(labels == 1)),
            "num_negative": int(np.sum(labels == 0)),
            "mean_missing_ratio": float(1.0 - mask_np.mean())
        }

        return x_log, mask_np, labels, cons_ids

# ==============================================================================
# 3. Stratified Split and I/O
# ==============================================================================
def save_stratified_splits(
    x_log: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    cons_ids: np.ndarray,
    ts_cols: list,
    cfg: GlobalConfig,
    report: dict
):
    print("Executing stratified Train/Test split and saving tensors.")
    os.makedirs(cfg.output_dir, exist_ok=True)

    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=cfg.test_size, stratify=labels, random_state=cfg.random_state
    )

    splits = {"train": idx_train, "test": idx_test}
    report["splits"] = {}

    for name, indices in splits.items():
        s_x_log = x_log[indices]
        s_mask = mask[indices]
        s_labels = labels[indices]
        s_cons = cons_ids[indices]

        npz_path = os.path.join(cfg.output_dir, f"{name}.npz")
        np.savez_compressed(
            npz_path,
            x_log=s_x_log,
            mask=s_mask,
            labels=s_labels,
            cons_ids=s_cons,
            time_cols=np.array(ts_cols, dtype=object)
        )

        report["splits"][name] = {
            "total": len(indices),
            "positives": int(np.sum(s_labels == 1)),
            "negatives": int(np.sum(s_labels == 0))
        }

    report["config"] = asdict(cfg)
    with open(os.path.join(cfg.output_dir, "audit_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("Pipeline execution complete. Summary:")
    print(f"Total samples: {len(labels)} (Class 0: {report['num_negative']} | Class 1: {report['num_positive']})")
    print(f"Train split: {report['splits']['train']['total']} | Test split: {report['splits']['test']['total']}")


if __name__ == "__main__":
    if not os.path.exists(GlobalConfig.input_file):
        print("Input file not found. Generating dummy dataset for testing.")
        os.makedirs(os.path.dirname(GlobalConfig.input_file), exist_ok=True)
        
        dates = pd.date_range("2014-01-01", periods=1035, freq="D")
        n_users = 500
        
        data = np.random.rand(n_users, 1035) * 100
        data[0:50, :] = np.nan 
        data[50:100, :] = 0.001 
        data[100:500, np.random.rand(400, 1035) < 0.1] = np.nan 
        
        df = pd.DataFrame(data, columns=dates.strftime("%Y-%m-%d"))
        df.insert(0, "CONS_NO", [f"U{i}" for i in range(n_users)])
        
        labels = np.zeros(n_users)
        labels[100:150] = 1
        df.insert(0, "FLAG", labels.astype(np.int8))
        df.to_csv(GlobalConfig.input_file, index=False)

    cfg = GlobalConfig()
    cleaner = SGCCMinimalPreprocessor(cfg)
    x_log, mask, labels, cons_ids = cleaner.run()
    save_stratified_splits(x_log, mask, labels, cons_ids, cleaner.ts_cols, cfg, cleaner.report)
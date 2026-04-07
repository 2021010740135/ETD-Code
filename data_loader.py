# data_pipeline.py
from __future__ import annotations

import os
import json
import warnings
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Configurations
# ==============================================================================
@dataclass
class WindowBuilderConfig:
    input_dir: str = "./results_phase1"
    output_dir: str = "./results_phase2"
    split_names: tuple[str, ...] = ("train", "test")
    window_size: int = 256
    stride: int = 28
    add_tail_window: bool = True
    min_valid_ratio: float = 0.0

@dataclass
class LoaderConfig:
    input_dir: str = "./results_phase2"
    train_file: str = "train_windows.npz"
    test_file: str = "test_windows.npz"
    train_normal_file: str = "train_normal_windows.npz"
    test_normal_file: str = "test_normal_windows.npz"
    
    batch_size_diffusion: int = 256
    batch_size_classifier: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    
    diffusion_min_valid_ratio: float = 0.0
    classifier_min_valid_ratio: float = 0.0
    seed: int = 58


# ==============================================================================
# 2. Window Construction Logic (Phase 2)
# ==============================================================================
def compute_window_starts(length: int, window_size: int, stride: int, add_tail_window: bool = True) -> np.ndarray:
    if length < window_size:
        raise ValueError(f"Sequence length {length} < window size {window_size}.")
    starts = list(range(0, length - window_size + 1, stride))
    tail_start = length - window_size
    if add_tail_window and (not starts or starts[-1] != tail_start):
        starts.append(tail_start)
    return np.array(sorted(set(starts)), dtype=np.int32)

def slice_2d_to_windows(x: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    """Shape transformation: [N, T] -> [N, K, W]"""
    idx = starts[:, None] + np.arange(window_size, dtype=np.int32)[None, :]
    return x[:, idx]

def build_split_windows(payload: Dict[str, np.ndarray], cfg: WindowBuilderConfig) -> Dict[str, np.ndarray]:
    x_log = payload["x_log"].astype(np.float32)
    mask = payload["mask"].astype(np.float32)
    labels = payload["labels"].astype(np.int8)
    cons_ids = payload["cons_ids"].astype(str)
    
    n_users, seq_len = x_log.shape
    starts = compute_window_starts(seq_len, cfg.window_size, cfg.stride, cfg.add_tail_window)
    
    # Global Min-Max Normalization (Incorporates future information across T)
    # Target shape: [N, seq_len]
    g_min = x_log.min(axis=-1, keepdims=True)
    g_max = x_log.max(axis=-1, keepdims=True)
    denom = np.maximum(g_max - g_min, 1e-6)
    x_mm_global = (x_log - g_min) / denom
    
    # Slicing operations
    x_log_w = slice_2d_to_windows(x_log, starts, cfg.window_size)
    x_mm_w = slice_2d_to_windows(x_mm_global, starts, cfg.window_size)
    mask_w = slice_2d_to_windows(mask, starts, cfg.window_size)
    
    valid_ratio = mask_w.mean(axis=-1).astype(np.float32)
    keep_window_mask = (valid_ratio >= cfg.min_valid_ratio).astype(np.int8)
    
    # Channel stacking. Shape: [N, K, 3, W]
    x = np.stack([x_log_w, x_mm_w, mask_w], axis=2).astype(np.float32)
    window_ends = (starts + cfg.window_size - 1).astype(np.int32)
    
    return {
        "x": x,
        "labels": labels,
        "cons_ids": cons_ids,
        "window_starts": starts,
        "window_ends": window_ends,
        "valid_ratio": valid_ratio,
        "keep_window_mask": keep_window_mask,
        "time_cols": payload["time_cols"],
        "channel_names": np.array(["x_log", "x_mm", "mask"], dtype=object)
    }

def subset_normal_users(payload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normal_mask = payload["labels"] == 0
    idx = np.where(normal_mask)[0]
    out = {}
    for k, v in payload.items():
        if k in {"window_starts", "window_ends", "time_cols", "channel_names"}:
            out[k] = v
        elif isinstance(v, np.ndarray) and v.shape[0] == len(normal_mask):
            out[k] = v[idx]
        else:
            out[k] = v
    return out

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def run_window_builder():
    cfg = WindowBuilderConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    report = {"config": asdict(cfg), "splits": {}}
    
    for split_name in cfg.split_names:
        input_path = os.path.join(cfg.input_dir, f"{split_name}.npz")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing file: {input_path}")
            
        data = np.load(input_path, allow_pickle=True)
        raw_payload = {k: data[k] for k in data.files}
        
        win_payload = build_split_windows(raw_payload, cfg)
        np.savez_compressed(os.path.join(cfg.output_dir, f"{split_name}_windows.npz"), **win_payload)
        
        normal_payload = subset_normal_users(win_payload)
        np.savez_compressed(os.path.join(cfg.output_dir, f"{split_name}_normal_windows.npz"), **normal_payload)
        
        report["splits"][split_name] = {"users": win_payload["x"].shape[0], "windows_per_user": win_payload["x"].shape[1]}
        report["splits"][f"{split_name}_normal"] = {"users": normal_payload["x"].shape[0], "windows_per_user": normal_payload["x"].shape[1]}

    report_path = os.path.join(cfg.output_dir, "window_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder, ensure_ascii=False)


# ==============================================================================
# 3. Dataset Definitions
# ==============================================================================
class WindowSplit:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing window asset file: {path}")

        raw = np.load(path, allow_pickle=True)
        payload = {k: raw[k] for k in raw.files}

        self.path = path
        self.x = payload["x"].astype(np.float32)                  # [N, K, C, W]
        self.labels = payload["labels"].astype(np.int64)          # [N]
        self.cons_ids = payload["cons_ids"].astype(str)           # [N]
        self.valid_ratio = payload["valid_ratio"].astype(np.float32)  # [N, K]
        
        channel_names = [str(c) for c in payload["channel_names"].tolist()]
        expected_channels = ["x_log", "x_mm", "mask"]
        if channel_names != expected_channels:
            raise ValueError(f"Channel mismatch. Expected {expected_channels}, got {channel_names}")

    @property
    def n_users(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_windows(self) -> int:
        return int(self.x.shape[1])


class DiffusionWindowDataset(Dataset):
    """
    Output Shape: [C, W]
    Flattens [N, K, C, W] across N and K dimensions.
    """
    def __init__(self, split: WindowSplit, min_valid_ratio: float = 0.0):
        self.split = split
        valid = split.valid_ratio >= min_valid_ratio
        
        self.index: List[Tuple[int, int]] = [
            (u, k) for u in range(split.n_users) for k in range(split.n_windows) if valid[u, k]
        ]
        if not self.index:
            raise ValueError(f"No valid windows in {split.path} after filtering by valid_ratio >= {min_valid_ratio}.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u, k = self.index[idx]
        return {
            "x": torch.from_numpy(self.split.x[u, k]),
            "valid_ratio": torch.tensor(self.split.valid_ratio[u, k], dtype=torch.float32),
        }


class UserBagDataset(Dataset):
    """
    Output Shape: [K, C, W]
    """
    def __init__(self, split: WindowSplit, min_valid_ratio: float = 0.0):
        self.split = split
        self.min_valid_ratio = min_valid_ratio

    def __len__(self) -> int:
        return self.split.n_users

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        valid_ratio = torch.from_numpy(self.split.valid_ratio[idx].copy())
        keep_mask = (valid_ratio >= self.min_valid_ratio).float()
        return {
            "x": torch.from_numpy(self.split.x[idx].copy()),
            "label": torch.tensor(self.split.labels[idx], dtype=torch.float32),
            "keep_window_mask": keep_mask,
            "cons_id": str(self.split.cons_ids[idx])
        }

# ==============================================================================
# 4. DataLoader Assembly
# ==============================================================================
def get_dataloaders(cfg: LoaderConfig = None) -> Dict[str, DataLoader]:
    if cfg is None:
        cfg = LoaderConfig()
        
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_split = WindowSplit(os.path.join(cfg.input_dir, cfg.train_file))
    test_split = WindowSplit(os.path.join(cfg.input_dir, cfg.test_file))
    train_normal_split = WindowSplit(os.path.join(cfg.input_dir, cfg.train_normal_file))
    test_normal_split = WindowSplit(os.path.join(cfg.input_dir, cfg.test_normal_file))

    datasets = {
        "diff_train": DiffusionWindowDataset(train_normal_split, cfg.diffusion_min_valid_ratio),
        "diff_val": DiffusionWindowDataset(test_normal_split, cfg.diffusion_min_valid_ratio),
        "clf_train": UserBagDataset(train_split, cfg.classifier_min_valid_ratio),
        "test": UserBagDataset(test_split, cfg.classifier_min_valid_ratio),
    }

    common_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers if cfg.num_workers > 0 else False,
    }
    
    return {
        "diff_train": DataLoader(datasets["diff_train"], batch_size=cfg.batch_size_diffusion, shuffle=True, drop_last=False, **common_kwargs),
        "diff_val": DataLoader(datasets["diff_val"], batch_size=cfg.batch_size_diffusion, shuffle=False, drop_last=False, **common_kwargs),
        "clf_train": DataLoader(datasets["clf_train"], batch_size=cfg.batch_size_classifier, shuffle=True, drop_last=False, **common_kwargs),
        "test": DataLoader(datasets["test"], batch_size=cfg.batch_size_classifier, shuffle=False, drop_last=False, **common_kwargs),
    }

# ==============================================================================
# 5. Execution Interface
# ==============================================================================
if __name__ == "__main__":
    # Execute window building pipeline
    if os.path.exists("./results_phase1/train.npz"):
        run_window_builder()
    
    # Execute dataloader assembly
    try:
        loaders = get_dataloaders()
        for name, loader in loaders.items():
            batch = next(iter(loader))
            print(f"[{name}] Batch 'x' shape: {batch['x'].shape}")
    except FileNotFoundError as e:
        print(f"DataLoader initialization failed: {e}")
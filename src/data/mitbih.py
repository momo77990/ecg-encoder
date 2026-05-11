"""MIT-BIH 心拍级心律失常数据集（来自 Kaggle shayanfazeli/heartbeat）。

CSV 格式（无表头）:
    每行 188 列: 前 187 列是信号（已 padding/truncate 到 187 个采样点，125 Hz 采样），
                  最后一列是标签 (float 0-4)。

类别:
    0: N (Normal)
    1: S (Supraventricular ectopic)
    2: V (Ventricular ectopic)
    3: F (Fusion)
    4: Q (Unknown)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = 5
SIGNAL_LEN = 187


class MITBIHDataset(Dataset):
    """读取 mitbih_train.csv 或 mitbih_test.csv。

    返回 (signal, label):
        signal: torch.float32, shape (1, 187)
        label:  torch.int64,   scalar in [0, 4]

    所有 CSV 一次性加载到内存（train 仅 ~330MB float32），训练时无 IO 瓶颈。
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        csv_path = os.path.join(data_dir, f"mitbih_{split}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run scripts/extract_data.sh first."
            )
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] != SIGNAL_LEN + 1:
            raise ValueError(
                f"expected {SIGNAL_LEN + 1} columns, got {df.shape[1]}"
            )

        # 信号 (N, 187) float32, 标签 (N,) int64
        self.signals = df.iloc[:, :SIGNAL_LEN].to_numpy(dtype=np.float32)
        self.labels = df.iloc[:, SIGNAL_LEN].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sig = self.signals[idx]                 # (187,)
        label = int(self.labels[idx])

        if self.transform is not None:
            sig = self.transform(sig)

        sig = torch.from_numpy(np.ascontiguousarray(sig)).float().unsqueeze(0)  # (1, 187)
        return sig, torch.tensor(label, dtype=torch.long)


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """按类别频率的倒数计算权重，归一化到均值=1。

    weights[c] = (N_total / N_classes) / (count[c] + smoothing)
    再缩放使所有权重的均值为 1（保持 loss scale 与无权重时相近）。

    smoothing 默认 1.0（Laplace 平滑），保证 count=0 时不会出现 inf/nan。
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    weights = (total / num_classes) / (counts + smoothing)
    weights = weights / weights.mean()              # mean-normalized
    return torch.tensor(weights, dtype=torch.float32)

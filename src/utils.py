"""通用工具：config 加载、随机种子、调度器、评测指标。"""
from __future__ import annotations

import math
import os
import random
from typing import Any

import numpy as np
import torch
import yaml


# ---------------- 配置 ----------------


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------- 随机种子 ----------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- 学习率调度 ----------------


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """线性 warmup + cosine 衰减到 min_lr_ratio * base_lr。"""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        # cosine: 1 → min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------- 评测指标 ----------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> dict[str, Any]:
    """返回 accuracy、macro F1、每类 F1。

    使用 sklearn 实现，避免自己写有 bug。
    """
    from sklearn.metrics import accuracy_score, f1_score

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(
            y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0
        )
    )
    per_class_f1 = f1_score(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": [float(v) for v in per_class_f1],
    }


def confusion_matrix_str(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> str:
    """返回带表头的混淆矩阵字符串，方便 print/log。"""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    header = "       " + " ".join(f"{n:>6s}" for n in class_names)
    lines = [header]
    for i, name in enumerate(class_names):
        row = " ".join(f"{int(v):>6d}" for v in cm[i])
        lines.append(f"{name:>5s}: {row}")
    return "\n".join(lines)


# ---------------- 杂项 ----------------


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

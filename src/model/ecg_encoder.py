"""ECG encoder + 分类器。

ECGEncoder = CNN1DFeatureExtractor + TransformerBackbone
ECGClassifier = ECGEncoder + mean pool（排除 CLS）+ Linear head
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .feature_extractor import CNN1DFeatureExtractor
from .transformer import TransformerBackbone


class ECGEncoder(nn.Module):
    """CNN 特征提取 + Transformer。返回含 CLS token 的序列特征。"""

    def __init__(
        self,
        in_channels: int = 1,
        conv_dim: int = 128,
        num_conv_blocks: int = 3,
        kernel_size: int = 3,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.feature_extractor = CNN1DFeatureExtractor(
            in_channels=in_channels,
            conv_dim=conv_dim,
            num_blocks=num_conv_blocks,
            kernel_size=kernel_size,
        )
        self.transformer = TransformerBackbone(
            in_dim=conv_dim,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            ffn_dim=ffn_dim,
            max_len=max_len,
            dropout=dropout,
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T)
        x = self.feature_extractor(x)        # (B, conv_dim, L)
        x = self.transformer(x)              # (B, L+1, d_model) 含 CLS
        return x


class ECGClassifier(nn.Module):
    """ECGEncoder + mean pool（排除 CLS）+ 线性分类头。"""

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        conv_dim: int = 128,
        num_conv_blocks: int = 3,
        kernel_size: int = 3,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.encoder = ECGEncoder(
            in_channels=in_channels,
            conv_dim=conv_dim,
            num_conv_blocks=num_conv_blocks,
            kernel_size=kernel_size,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T)
        feats = self.encoder(x)              # (B, L+1, d_model)
        # 排除 CLS token（位置 0），对 patch tokens 做 mean pool
        pooled = feats[:, 1:, :].mean(dim=1) # (B, d_model)
        logits = self.head(pooled)           # (B, num_classes)
        return logits

    @classmethod
    def from_config(cls, model_cfg: dict[str, Any]) -> "ECGClassifier":
        """从 config dict 构造（用于 train/evaluate 脚本）。"""
        return cls(**model_cfg)

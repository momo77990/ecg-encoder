"""LLaVA 风格 2-layer MLP projector。

把 ECG encoder 的 d_model 输出投影到 LLM 的 hidden_dim 空间，
让 LLM 把 ECG 特征当作 soft token 处理。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """2-layer MLP: in_dim → hidden_dim → out_dim，中间 GELU。

    LLaVA-1.5 即采用此结构。比单层 Linear 表达力强但参数增量很小。
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 1536,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) → (..., out_dim)
        return self.fc2(self.act(self.fc1(x)))

"""Transformer encoder backbone（BERT-style，Pre-LN）。

输入: CNN 特征提取器的输出 (B, C_conv, L)
输出: 含 CLS token 的序列 (B, L+1, d_model)

实现:
  1. permute → (B, L, C_conv)
  2. Linear(C_conv → d_model) 投影
  3. prepend learnable CLS token
  4. + learnable absolute positional embedding
  5. Pre-LN nn.TransformerEncoder
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,        # CNN 输出通道数
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 4,
        ffn_dim: int = 1024,
        max_len: int = 256,        # 位置编码的最大序列长度（含 CLS），187/8=24 远小于此值
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.input_proj = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 学习的绝对位置编码（包含 CLS 位）
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,        # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_conv, L)
        x = x.transpose(1, 2)                       # (B, L, C_conv)
        x = self.input_proj(x)                      # (B, L, d_model)

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)              # (B, L+1, d_model)

        if x.size(1) > self.max_len:
            raise ValueError(
                f"Sequence length {x.size(1)} > max_len {self.max_len}. "
                f"Increase TransformerBackbone(max_len=...)."
            )
        x = x + self.pos_embed[:, : x.size(1), :]   # (B, L+1, d_model)
        x = self.dropout(x)

        x = self.encoder(x)                         # (B, L+1, d_model)
        return x

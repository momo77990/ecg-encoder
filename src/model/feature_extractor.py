"""1D CNN 特征提取器（wav2vec 2.0 / ECG-FM 风格）。

每个 block: Conv1d(k, stride=2) → LayerNorm(channel-last) → GELU
默认 3 个 block，总计 8x 时间维降采样。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ChannelLastLayerNorm(nn.Module):
    """在 channel 维上做 LayerNorm。

    PyTorch 的 LayerNorm 默认作用于最后一维，所以输入张量
    形状 (B, C, T) 时需要先 transpose 到 (B, T, C)，做 norm，
    再 transpose 回 (B, C, T)。
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.transpose(1, 2)        # (B, T, C)
        x = self.norm(x)
        x = x.transpose(1, 2)        # (B, C, T)
        return x


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()
        # padding 让输出长度 ≈ ceil(T / stride)
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = _ChannelLastLayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CNN1DFeatureExtractor(nn.Module):
    """1D CNN 特征提取器。

    输入  (B, in_channels, T)
    输出  (B, conv_dim, T // (stride ** num_blocks))

    默认: in_channels=1, conv_dim=128, num_blocks=3, kernel_size=3, stride=2
    则 187 → 94 → 47 → 24 个 patch。
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_dim: int = 128,
        num_blocks: int = 3,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        prev = in_channels
        for _ in range(num_blocks):
            blocks.append(
                _ConvBlock(prev, conv_dim, kernel_size=kernel_size, stride=stride)
            )
            prev = conv_dim
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = conv_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T)
        return self.blocks(x)

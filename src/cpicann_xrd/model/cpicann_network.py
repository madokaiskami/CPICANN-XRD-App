"""Minimal CPICANN network definition for inference.

Adapted from upstream `src/model/CPICANN.py` in
https://huggingface.co/AI4Cryst/CPICANN at commit
3dbfaeab51d272e013d211c7f957760b46ab41cc. The upstream project is distributed
under the MIT license through https://github.com/WPEM/CPICANN.

Product-local changes:
- keep only the single-phase inference network;
- add type hints where practical;
- rename helper classes to private names.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_


class CPICANNSinglePhaseNet(nn.Module):
    """CPICANN single-phase classifier network."""

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        num_classes: int = 23073,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.conv = _ConvModule()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 142))
        sa_layer = _SelfAttnLayer(embed_dim, nhead, dim_feedforward, dropout, activation)
        self.encoder = _SelfAttnModule(sa_layer, num_encoder_layers)
        self.norm_after = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * 4)),
            nn.BatchNorm1d(int(embed_dim * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(int(embed_dim * 4), int(embed_dim * 4)),
            nn.BatchNorm1d(int(embed_dim * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(int(embed_dim * 4), num_classes),
        )
        self._reset_parameters()
        self._init_weights()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _init_weights(self) -> None:
        trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed.requires_grad = False
        pos_embed = _get_1d_sincos_pos_embed_from_grid(
            self.embed_dim,
            np.array(range(self.pos_embed.shape[2])),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).T.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        batch_size = x.shape[0]
        if x.shape[1] == 2:
            x = x[:, 1:, :]
        x = x / 100
        x = self.conv(x)
        x = x.permute(2, 0, 1).contiguous()
        cls_token = self.cls_token.expand(-1, batch_size, -1)
        x = torch.cat((cls_token, x), dim=0)
        pos_embed = self.pos_embed.permute(2, 0, 1).contiguous().repeat(1, batch_size, 1)
        feats = self.encoder(x, pos_embed)
        feats = self.norm_after(feats)
        return cast(torch.Tensor, self.cls_head(feats[0]))


class _ConvModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=35, stride=2, padding=17)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _Layer(64, 64, kernel_size=3, stride=2, downsample=True)
        self.layer2 = _Layer(64, 128, kernel_size=3, stride=2, downsample=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return cast(torch.Tensor, self.maxpool2(x))


class _SelfAttnModule(nn.Module):
    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class _SelfAttnLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        q = k = _with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return cast(torch.Tensor, self.norm2(src))


class _Layer(nn.Module):
    def __init__(
        self,
        inchannel: int,
        outchannel: int,
        *,
        kernel_size: int,
        stride: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.block1 = _BasicBlock(
            inchannel,
            outchannel,
            kernel_size=kernel_size,
            stride=stride,
            downsample=downsample,
        )
        self.block2 = _BasicBlock(outchannel, outchannel, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        return cast(torch.Tensor, self.block2(x))


class _BasicBlock(nn.Module):
    def __init__(
        self,
        inchannel: int,
        outchannel: int,
        *,
        kernel_size: int,
        stride: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            inchannel,
            outchannel,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.conv2 = nn.Conv1d(
            outchannel,
            outchannel,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=2),
                nn.BatchNorm1d(outchannel),
            )
            if downsample
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        return cast(torch.Tensor, self.act2(x))


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return cast(Callable[[torch.Tensor], torch.Tensor], F.relu)
    if activation == "gelu":
        return cast(Callable[[torch.Tensor], torch.Tensor], F.gelu)
    if activation == "glu":
        return cast(Callable[[torch.Tensor], torch.Tensor], F.glu)
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    return tensor if pos is None else tensor + pos


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out).astype(np.float32)
    emb_cos = np.cos(out).astype(np.float32)
    return np.concatenate([emb_sin, emb_cos], axis=1)

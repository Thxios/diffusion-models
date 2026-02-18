
import math
import torch
from torch import nn
from typing import Optional

from modeling.activation import get_activation


class PeriodicEncoding(nn.Module):
    def __init__(
            self,
            n_dim: int,
            max_period: int = 10000,
    ):
        super().__init__()
        assert n_dim % 2 == 0

        half_dim = n_dim // 2
        freq_base = torch.linspace(0, 1, steps=half_dim, dtype=torch.float32)
        freq = torch.exp(-math.log(max_period) * freq_base)
        self.register_buffer('freq', freq)

    def forward(self, x: torch.Tensor):
        freq = x[:, None] * self.freq[None, :]
        emb = torch.cat([torch.sin(freq), torch.cos(freq)], dim=-1)
        return emb


class MLPEmbedding(nn.Module):
    def __init__(
            self,
            time_emb_dim: int,
            out_dim: int,
            activation: str = 'swish',
    ):
        super().__init__()

        self.linear1 = nn.Linear(time_emb_dim, out_dim)
        self.nonlinearity = get_activation(activation)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.nonlinearity(x)
        x = self.linear2(x)
        return x


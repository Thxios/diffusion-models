import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


"""
def attn_qkvpacked_func(
        qkv: torch.Tensor,
        dropout_p: float = 0.,
        softmax_scale = None,
):
    if softmax_scale is None:
        softmax_scale = qkv.size(-1) ** -0.5
    q, k, v = rearrange(qkv, 'b l a h d -> a b h l d', a=3)
    score = torch.matmul(q, k.transpose(-1, -2))
    weight = F.softmax(score * softmax_scale, dim=-1)
    if dropout_p > 0:
        weight = F.dropout(weight, p=dropout_p)
    out = torch.matmul(weight, v)
    # out = out.transpose(1, 2)
    return out

def enable_flash_attn():
    try:
        from flash_attn import flash_attn_qkvpacked_func
    except ImportError:
        return False
    Attention2D._attn_processor = flash_attn_qkvpacked_func
    return True
"""


class Attention2D(nn.Module):
    def __init__(
            self,
            n_dim: int,
            n_heads: Optional[int] = None,
            n_dim_head: Optional[int] = None,
            norm_groups: int = 32,
            dropout: float = 0.,
    ):
        super().__init__()
        assert n_dim_head is not None or n_heads is not None
        if n_dim_head is None:
            assert n_dim % n_heads == 0
            n_dim_head = n_dim // n_heads
        elif n_heads is None:
            assert n_dim % n_dim_head == 0
            n_heads = n_dim // n_dim_head

        self.n_heads = n_heads
        self.n_dim_head = n_dim_head
        self.n_dim_inner = n_heads * n_dim_head
        self.register_buffer('dropout_p', torch.tensor(dropout))

        self.norm = nn.GroupNorm(norm_groups, n_dim)
        self.qkv_proj = nn.Linear(n_dim, 3 * self.n_dim_inner)
        self.out_proj = nn.Linear(self.n_dim_inner, n_dim)

    def forward(self, x: torch.Tensor):
        batch_size, n_dim, h, w = x.shape
        seq_len = h * w
        res = x

        x = self.norm(x)
        x = x.view(batch_size, n_dim, seq_len).transpose(-1, -2)

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.n_dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0
        )
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.n_dim_inner)
        out = self.out_proj(attn)
        out = out.transpose(-1, -2).view(batch_size, n_dim, h, w)

        x = out + res
        return x

    def attn_qkvpacked_func(self, qkv: torch.Tensor):
        q, k, v = rearrange(
            qkv,
            'b l (n h d) -> n b h l d',
            n=3, h=self.n_heads, d=self.n_dim_head
        )
        score = torch.matmul(q, k.transpose(-1, -2))
        weight = F.softmax(score * (q.size(-1) ** -0.5), dim=-1)
        if self.training and self.dropout_p > 0:
            weight = F.dropout(weight, p=self.dropout_p)
        out = torch.matmul(weight, v)
        return out




import torch
from torch import nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Optional



class Dynamic2DRoPE(nn.Module):
    # RoPE for 2D data (e.g., images) with dynamic resolution (dynamic aspect ratio)

    def __init__(self, head_dim, max_res=1024, theta=10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        half_dim = head_dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
        t = torch.arange(max_res).float()
        freqs = torch.outer(t, freqs)
        
        self.register_buffer("freqs_cos", torch.cos(freqs), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(freqs), persistent=False)

    def forward(self, h, w):
        cos_h, cos_w = self.freqs_cos[:h, :], self.freqs_cos[:w, :]
        sin_h, sin_w = self.freqs_sin[:h, :], self.freqs_sin[:w, :]
        
        cos_h = cos_h.unsqueeze(1).expand(h, w, -1)
        cos_w = cos_w.unsqueeze(0).expand(h, w, -1)
        sin_h = sin_h.unsqueeze(1).expand(h, w, -1)
        sin_w = sin_w.unsqueeze(0).expand(h, w, -1)
        
        cos_2d = torch.cat([cos_h, cos_w], dim=-1).reshape(h * w, -1)
        sin_2d = torch.cat([sin_h, sin_w], dim=-1).reshape(h * w, -1)
        
        cos_2d = cos_2d.unsqueeze(0).unsqueeze(1).repeat_interleave(2, dim=-1)
        sin_2d = sin_2d.unsqueeze(0).unsqueeze(1).repeat_interleave(2, dim=-1)
        return cos_2d, sin_2d


def apply_rotary_emb(x, cos, sin):
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_reshaped.unbind(-1)
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    return x * cos + x_rot * sin


class PeriodicEncoding(nn.Module):
    def __init__(self, hidden_size, time_encoding_dim=256, max_period=10000):
        super().__init__()
        assert time_encoding_dim % 2 == 0, "time_encoding_dim must be even for sinusoidal embedding"

        self.mlp = nn.Sequential(
            nn.Linear(time_encoding_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.time_encoding_dim = time_encoding_dim

        half_dim = time_encoding_dim // 2
        freq_base = torch.arange(half_dim, dtype=torch.float32) / half_dim
        freq = torch.exp(-math.log(max_period) * freq_base)
        
        self.register_buffer('freq', freq, persistent=False)

    def forward(self, x: torch.Tensor):
        x_f32 = x.to(torch.float32)
        freq = x_f32[:, None] * self.freq[None, :]
        emb = torch.cat([torch.sin(freq), torch.cos(freq)], dim=-1)
        return emb


class TimestepEmbedder(nn.Sequential):
    def __init__(self, time_encoding_dim, embedding_dim, max_period=10000):
        super().__init__(
            PeriodicEncoding(embedding_dim, time_encoding_dim, max_period),
            nn.Linear(time_encoding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )


class BottleneckPatchEmbedder(nn.Module):
    def __init__(self, patch_size, in_channels, hidden_size, bottleneck_size):
        super().__init__()
        self.proj1 = nn.Conv2d(
            in_channels, bottleneck_size, 
            kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.proj2 = nn.Linear(bottleneck_size, hidden_size)
    
    def forward(self, x):
        x = self.proj1(x)  # (B, bottleneck_size, H//P, W//P)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, bottleneck_size)
        x = self.proj2(x)  # (B, N_patches, hidden_size)
        return x


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLUFFNBlock(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            intermediate_size, 
            embedding_size, 
            ffn_bias=True,
            dropout=0.0,
    ):
        super().__init__()

        self.norm = nn.RMSNorm(hidden_size)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_size, 3 * hidden_size)
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        self.w12 = nn.Linear(hidden_size, intermediate_size * 2, bias=ffn_bias)
        self.dropout = nn.Dropout(dropout)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=ffn_bias)

    def forward(self, x, cond):
        # pre_norm
        hidden = self.norm(x)
        shift, scale, gate1 = self.adaln(cond).chunk(3, dim=-1)
        hidden = modulate(hidden, scale, shift)
        
        gate2, up = self.w12(hidden).chunk(2, dim=-1)
        hidden = F.silu(gate2) * up
        hidden = self.dropout(hidden)
        ffn_out = self.w3(hidden)

        x = x + gate1.unsqueeze(1) * ffn_out
        return x
    

class AttentionBlock(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            embedding_size,
            num_heads,
            qkv_bias=True,
            dropout=0.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        head_dim = hidden_size // num_heads

        self.num_heads = num_heads
        self.dropout_p = dropout

        self.norm = nn.RMSNorm(hidden_size)
        self.adaln = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embedding_size, 3 * hidden_size)
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond, rope):
        batch_size, seq_len, hidden_size = x.shape

        # pre_norm
        hidden = self.norm(x)
        shift, scale, gate = self.adaln(cond).chunk(3, dim=-1)
        hidden = modulate(hidden, scale, shift)

        qkv = self.qkv_proj(hidden)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous() # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)

        # qk norm + RoPE
        q = rope(self.q_norm(q))
        k = rope(self.k_norm(k))

        attn = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0
        )
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

        out = self.out_proj(attn)
        out = self.dropout(out)

        x = x + gate.unsqueeze(1) * out
        return x


class TransformerBlock(nn.Module):
    def __init__(
            self, 
            hidden_size, 
            intermediate_size, 
            embedding_size, 
            num_heads: Optional[int] = None,
            ffn_bias=True,
            attn_bias=True,
            dropout=0.0,
    ):
        super().__init__()
        self.attention = AttentionBlock(
            hidden_size, embedding_size, num_heads, attn_bias, dropout
        )
        self.swiglu_ffn = SwiGLUFFNBlock(
            hidden_size, intermediate_size, embedding_size, ffn_bias, dropout
        )

    def forward(self, x, cond, rope):
        x = self.attention(x, cond, rope)
        x = self.swiglu_ffn(x, cond)
        return x
    

class OutputLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, cond):
        shift, scale = self.adaln(cond).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class UltimateTransformer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_layers,
            hidden_size,
            embedding_dim,
            patch_size,
            bottleneck_size,
            intermediate_size,
            num_heads,
            num_class_embeddings: Optional[int] = None,
            time_encoding_dim=256,
            max_period=10000,
            ffn_bias=True,
            attn_bias=True,
            dropout=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.patchify = BottleneckPatchEmbedder(
            patch_size, in_channels, hidden_size, bottleneck_size=bottleneck_size
        )
        self.time_embedding = TimestepEmbedder(time_encoding_dim, embedding_dim, max_period)
        if num_class_embeddings is not None:
            self.class_embedding = nn.Embedding(
                num_class_embeddings + 1,
                embedding_dim,
                padding_idx=num_class_embeddings,
            )
            self.register_buffer('uncond_class', torch.tensor(num_class_embeddings))
        else:
            self.class_embedding = None

        self.rope_2d = Dynamic2DRoPE(hidden_size // num_heads, max_res=1024)

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size, 
                intermediate_size, 
                embedding_dim, 
                num_heads, 
                ffn_bias, 
                attn_bias, 
                dropout
            ) for _ in range(num_layers)
        ])
        self.out = OutputLayer(hidden_size, patch_size, out_channels)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cls: Optional[torch.Tensor] = None,
            uncond_mask: Optional[torch.Tensor] = None
    ):
        batch_size, _, height, width = x.shape
        assert height % self.patch_size == 0 and width % self.patch_size == 0, \
            "Height and width must be divisible by patch size"
        h, w = height // self.patch_size, width // self.patch_size
        rope_cos, rope_sin = self.rope_2d(h, w)
        rope = partial(apply_rotary_emb, cos=rope_cos, sin=rope_sin)

        embedding = self.time_embedding(t)
        if self.class_embedding is not None:
            if cls is not None:
                if uncond_mask is not None:
                    cls = torch.masked_fill(cls, uncond_mask.bool(), self.uncond_class)
                cls_embedding = self.class_embedding(cls)
            else:
                cls_embedding = self.class_embedding(self.uncond_class.repeat(x.size(0)))
            embedding = embedding + cls_embedding
        
        x = self.patchify(x)
        for layer in self.layers:
            x = layer(x, embedding, rope)
        x = self.out(x, embedding)  # (batch_size, seq_len, patch_size*patch_size*out_channels)

        x = x.view(batch_size, h, w, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (batch_size, out_channels, h, patch_size, w, patch_size)
        x = x.view(batch_size, -1, height, width)

        return x


if __name__ == '__main__':
    model = UltimateTransformer(
        in_channels=3,
        out_channels=3,
        num_layers=4,
        hidden_size=256,
        embedding_dim=256,
        patch_size=16,
        bottleneck_size=32,
        intermediate_size=768,
        num_heads=8,
        time_encoding_dim=128,
    )
    x = torch.randn(2, 3, 128, 128)
    t = torch.tensor([10, 20])
    out = model(x, t)
    print(out.shape)


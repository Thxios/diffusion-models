
import torch
from torch import nn
from typing import Optional, Tuple

from modeling.blocks import get_down_block, get_up_block, MidBlock
from modeling.activation import get_activation
from modeling.embeddings import PeriodicEncoding, MLPEmbedding



class UNet(nn.Module):
    """
    UNet
    same architecture with `diffusers.UNet2DModel`
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d.py
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str, ...] = ('UNetDownBlock',),
            up_block_types: Tuple[str, ...] = ('UNetUpBlock',),
            block_out_channels: Tuple[int, ...] = (64,),
            n_blocks_per_layer: int = 2,
            n_attention_heads: Optional[int] = None,
            n_dim_attention_head: Optional[int] = None,
            mid_attention: bool = True,
            norm_groups: int = 32,
            activation: str = 'silu',
            time_encoding_dim: int = 64,
            embedding_dim: int = 64,
            embedding_mixing: str = 'addition',
            n_class_embeddings: Optional[int] = None,
            dropout: float = 0.,
            downsample_use_conv: bool = True,
            upsample_use_conv: bool = True,
    ):
        super().__init__()
        assert len(down_block_types) == len(up_block_types) == len(block_out_channels)

        self.in_conv = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding='same')

        self.time_embedding = nn.Sequential(
            PeriodicEncoding(time_encoding_dim),
            MLPEmbedding(time_encoding_dim, embedding_dim),
        )
        if n_class_embeddings is not None:
            self.class_embedding = nn.Embedding(
                n_class_embeddings + 1,
                embedding_dim,
                padding_idx=n_class_embeddings,
            )
            self.register_buffer('uncond_class', torch.tensor(n_class_embeddings))
        else:
            self.class_embedding = None

        down = []
        prev_channels = block_out_channels[0]
        for i, block_type in enumerate(down_block_types):
            is_last_block = (i == len(down_block_types) - 1)
            block = get_down_block(
                block_type,
                in_channels=prev_channels,
                out_channels=block_out_channels[i],
                n_blocks=n_blocks_per_layer,
                n_heads=n_attention_heads,
                n_dim_head=n_dim_attention_head,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=embedding_dim,
                embedding_mixing=embedding_mixing,
                dropout=dropout,
                downsample=True if not is_last_block else False,
                downsample_use_conv=downsample_use_conv,
            )
            down.append(block)
            prev_channels = block_out_channels[i]

        up = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_block_out_channels += [block_out_channels[0]]
        for i, block_type in enumerate(up_block_types):
            is_last_block = (i == len(down_block_types) - 1)
            up_out_channels = reversed_block_out_channels[i]
            next_out_channels = reversed_block_out_channels[i + 1]

            block = get_up_block(
                block_type,
                in_channels=prev_channels,
                out_channels=up_out_channels,
                next_out_channels=next_out_channels,
                n_blocks=n_blocks_per_layer + 1,
                n_heads=n_attention_heads,
                n_dim_head=n_dim_attention_head,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=embedding_dim,
                embedding_mixing=embedding_mixing,
                dropout=dropout,
                upsample=True if not is_last_block else False,
                upsample_use_conv=upsample_use_conv,
            )
            up.append(block)
            prev_channels = up_out_channels


        self.down = nn.ModuleList(down)
        self.mid = MidBlock(
            channels=block_out_channels[-1],
            n_heads=n_attention_heads,
            n_dim_head=n_dim_attention_head,
            norm_groups=norm_groups,
            activation=activation,
            embedding_dim=embedding_dim,
            embedding_mixing=embedding_mixing,
            add_attention=mid_attention,
            dropout=dropout,
        )
        self.up = nn.ModuleList(up)

        self.out_norm = nn.GroupNorm(norm_groups, block_out_channels[0])
        self.out_act = get_activation(activation)
        self.out_conv = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding='same')


    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cls: Optional[torch.Tensor] = None,
            uncond_mask: Optional[torch.Tensor] = None
    ):
        embedding = self.time_embedding(t)

        if self.class_embedding is not None:
            if cls is not None:
                if uncond_mask is not None:
                    cls = torch.masked_fill(cls, uncond_mask.bool(), self.uncond_class)
                cls_embedding = self.class_embedding(cls)
            else:
                cls_embedding = self.class_embedding(self.uncond_class.repeat(x.size(0)))
            embedding += cls_embedding

        x = self.in_conv(x)
        skips = (x,)
        for down in self.down:
            x, block_skips = down(x, embedding=embedding)
            skips += block_skips

        x = self.mid(x, embedding=embedding)

        for up in self.up:
            n_blocks = len(up.blocks)
            block_skips, skips = skips[-n_blocks:], skips[:-n_blocks]
            x = up(x, block_skips, embedding=embedding)

        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        return x

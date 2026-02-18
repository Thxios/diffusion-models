import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

from modeling.attention import Attention2D
from modeling.activation import get_activation


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_groups: int = 32,
            activation: str = 'swish',
            embedding_dim: Optional[int] = None,
            embedding_mixing: str = 'addition',
            dropout: float = 0.,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.embedding_mixing = embedding_mixing

        self.norm1 = nn.GroupNorm(norm_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.norm2 = nn.GroupNorm(norm_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')

        if embedding_dim is not None:
            if embedding_mixing == 'addition':
                self.emb_proj = nn.Linear(embedding_dim, out_channels)
            elif embedding_mixing == 'scale_shift':
                self.emb_proj = nn.Linear(embedding_dim, out_channels * 2)
            else:
                raise ValueError(f'unknown embedding mixing "{embedding_mixing}"')
        else:
            self.emb_proj = None

        self.nonlinearity = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        if in_channels == out_channels:
            self.res_proj = nn.Identity()
        else:
            self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None):
        res = x

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        if self.emb_proj is not None:
            assert embedding is not None
            embedding = self.nonlinearity(embedding)
            embedding = self.emb_proj(embedding)[:, :, None, None]

            if self.embedding_mixing == 'addition':
                x = x + embedding
                x = self.norm2(x)

            elif self.embedding_mixing == 'scale_shift':
                scale, shift = torch.chunk(embedding, 2, dim=1)
                x = self.norm2(x)
                x = (1 + scale) * x + shift
        else:
            x = self.norm2(x)

        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)

        x = x + self.res_proj(res)
        return x


class DownsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            use_conv: bool = True
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            assert in_channels == out_channels
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            use_conv: bool = True,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        else:
            assert in_channels == out_channels
            self.conv = None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.conv is not None:
            x = self.conv(x)
        return x


class UNetDownBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            norm_groups: int = 32,
            activation: str = 'swish',
            embedding_dim: Optional[int] = None,
            embedding_mixing: str = 'addition',
            dropout: float = 0.,
            downsample: bool = True,
            downsample_use_conv: bool = True,
    ):
        super().__init__()

        blocks = []
        for i in range(n_blocks):
            block = ConvBlock(
                in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=embedding_dim,
                embedding_mixing=embedding_mixing,
                dropout=dropout,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if downsample:
            self.downsample = DownsampleBlock(
                out_channels,
                out_channels,
                use_conv=downsample_use_conv,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None):
        skips = ()

        for block in self.blocks:
            x = block(x, embedding=embedding)
            skips += (x,)

        if self.downsample is not None:
            x = self.downsample(x)
            skips += (x,)

        return x, skips


class UNetAttnDownBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            n_heads: Optional[int] = None,
            n_dim_head: Optional[int] = None,
            norm_groups: int = 32,
            activation: str = 'swish',
            embedding_dim: Optional[int] = None,
            embedding_mixing: str = 'addition',
            dropout: float = 0.,
            downsample: bool = True,
            downsample_use_conv: bool = True,
    ):
        super().__init__()

        blocks = []
        attentions = []
        for i in range(n_blocks):
            block = ConvBlock(
                in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=embedding_dim,
                embedding_mixing=embedding_mixing,
                dropout=dropout,
            )
            attention = Attention2D(
                out_channels,
                n_heads=n_heads,
                n_dim_head=n_dim_head,
                norm_groups=norm_groups,
                dropout=dropout,
            )
            blocks.append(block)
            attentions.append(attention)
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)

        if downsample:
            self.downsample = DownsampleBlock(
                out_channels,
                out_channels,
                use_conv=downsample_use_conv,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None):
        skips = ()

        for block, attention in zip(self.blocks, self.attentions):
            x = block(x, embedding=embedding)
            x = attention(x)
            skips += (x,)

        if self.downsample is not None:
            x = self.downsample(x)
            skips += (x,)

        return x, skips


class UNetUpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            next_out_channels: int,
            n_blocks: int = 2,
            norm_groups: int = 32,
            activation: str = 'swish',
            embedding_dim: Optional[int] = None,
            embedding_mixing: str = 'addition',
            dropout: float = 0.,
            upsample: bool = True,
            upsample_use_conv: bool = True,
    ):
        super().__init__()

        blocks = []
        for i in range(n_blocks):
            prev_in_channels = (in_channels if i == 0 else out_channels)
            skip_channels = (out_channels if i < n_blocks - 1 else next_out_channels)
            block = ConvBlock(
                prev_in_channels + skip_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=embedding_dim,
                embedding_mixing=embedding_mixing,
                dropout=dropout,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if upsample:
            self.upsample = UpsampleBlock(
                out_channels,
                out_channels,
                use_conv=upsample_use_conv
            )
        else:
            self.upsample = None

    def forward(
            self,
            x: torch.Tensor,
            skips: Tuple[torch.Tensor],
            embedding: Optional[torch.Tensor] = None
    ):
        for block, skip in zip(self.blocks, reversed(skips)):
            x = torch.cat([x, skip], dim=1)
            x = block(x, embedding=embedding)

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class UNetAttnUpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            next_out_channels: int,
            n_blocks: int = 2,
            n_heads: Optional[int] = None,
            n_dim_head: Optional[int] = None,
            norm_groups: int = 32,
            activation: str = 'swish',
            embedding_dim: Optional[int] = None,
            embedding_mixing: str = 'addition',
            dropout: float = 0.,
            upsample: bool = True,
            upsample_use_conv: bool = True,
    ):
        super().__init__()

        blocks = []
        attentions = []
        for i in range(n_blocks):
            prev_in_channels = (in_channels if i == 0 else out_channels)
            skip_channels = (out_channels if i < n_blocks - 1 else next_out_channels)
            block = ConvBlock(
                prev_in_channels + skip_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=embedding_dim,
                embedding_mixing=embedding_mixing,
                dropout=dropout,
            )
            attention = Attention2D(
                out_channels,
                n_heads=n_heads,
                n_dim_head=n_dim_head,
                norm_groups=norm_groups,
                dropout=dropout,
            )
            blocks.append(block)
            attentions.append(attention)
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)

        if upsample:
            self.upsample = UpsampleBlock(
                out_channels,
                out_channels,
                use_conv=upsample_use_conv
            )
        else:
            self.upsample = None

    def forward(
            self,
            x: torch.Tensor,
            skips: Tuple[torch.Tensor],
            embedding: Optional[torch.Tensor] = None
    ):
        for block, attention, skip in zip(self.blocks, self.attentions, reversed(skips)):
            x = torch.cat([x, skip], dim=1)
            x = block(x, embedding=embedding)
            x = attention(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MidBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            n_blocks: int = 1,
            n_heads: Optional[int] = None,
            n_dim_head: Optional[int] = None,
            norm_groups: int = 32,
            activation: str = 'swish',
            embedding_dim: Optional[int] = None,
            embedding_mixing: str = 'addition',
            add_attention: bool = True,
            dropout: float = 0.,
    ):
        super().__init__()

        def make_conv_block():
            return ConvBlock(
                channels,
                out_channels=channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=embedding_dim,
                embedding_mixing=embedding_mixing,
                dropout=dropout
            )

        blocks = [make_conv_block()]
        attentions = []
        for _ in range(n_blocks):
            blocks.append(make_conv_block())
            if add_attention:
                attention = Attention2D(
                    channels,
                    n_heads=n_heads,
                    n_dim_head=n_dim_head,
                    norm_groups=norm_groups,
                    dropout=dropout,
                )
                attentions.append(attention)
            else:
                attentions.append(None)
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None):
        x = self.blocks[0](x, embedding=embedding)
        for block, attention in zip(self.blocks[1:], self.attentions):
            if attention is not None:
                x = attention(x)
            x = block(x, embedding=embedding)
        return x


class EncoderDownBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            norm_groups: int = 32,
            activation: str = 'swish',
            dropout: float = 0.,
            downsample: bool = True,
            downsample_use_conv: bool = True,
    ):
        super().__init__()

        blocks = []
        for i in range(n_blocks):
            block = ConvBlock(
                in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=None,
                dropout=dropout,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if downsample:
            self.downsample = DownsampleBlock(
                out_channels,
                out_channels,
                use_conv=downsample_use_conv,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x



class EncoderAttnDownBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            n_heads: Optional[int] = None,
            n_dim_head: Optional[int] = None,
            norm_groups: int = 32,
            activation: str = 'swish',
            dropout: float = 0.,
            downsample: bool = True,
            downsample_use_conv: bool = True,
    ):
        super().__init__()

        blocks = []
        attentions = []
        for i in range(n_blocks):
            block = ConvBlock(
                in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=None,
                dropout=dropout,
            )
            attention = Attention2D(
                out_channels,
                n_heads=n_heads,
                n_dim_head=n_dim_head,
                norm_groups=norm_groups,
                dropout=dropout,
            )
            blocks.append(block)
            attentions.append(attention)
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)

        if downsample:
            self.downsample = DownsampleBlock(
                out_channels,
                out_channels,
                use_conv=downsample_use_conv,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        for block, attention in zip(self.blocks, self.attentions):
            x = block(x)
            x = attention(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x



class DecoderUpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            norm_groups: int = 32,
            activation: str = 'swish',
            dropout: float = 0.,
            upsample: bool = True,
            upsample_use_conv: bool = True,
    ):
        super().__init__()

        blocks = []
        for i in range(n_blocks):
            block = ConvBlock(
                in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=None,
                dropout=dropout,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        if upsample:
            self.upsample = UpsampleBlock(
                out_channels,
                out_channels,
                use_conv=upsample_use_conv
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class DecoderAttnUpBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_blocks: int = 2,
            n_heads: Optional[int] = None,
            n_dim_head: Optional[int] = None,
            norm_groups: int = 32,
            activation: str = 'swish',
            dropout: float = 0.,
            upsample: bool = True,
            upsample_use_conv: bool = True,
    ):
        super().__init__()

        blocks = []
        attentions = []
        for i in range(n_blocks):
            block = ConvBlock(
                in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm_groups=norm_groups,
                activation=activation,
                embedding_dim=None,
                dropout=dropout,
            )
            attention = Attention2D(
                out_channels,
                n_heads=n_heads,
                n_dim_head=n_dim_head,
                norm_groups=norm_groups,
                dropout=dropout,
            )
            blocks.append(block)
            attentions.append(attention)
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)

        if upsample:
            self.upsample = UpsampleBlock(
                out_channels,
                out_channels,
                use_conv=upsample_use_conv
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor):
        for block, attention in zip(self.blocks, self.attentions):
            x = block(x)
            x = attention(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x



def get_down_block(
        block_type: str,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        n_heads: Optional[int] = None,
        n_dim_head: Optional[int] = None,
        norm_groups: int = 32,
        activation: str = 'swish',
        embedding_dim: Optional[int] = None,
        embedding_mixing: str = 'addition',
        dropout: float = 0.,
        downsample: bool = True,
        downsample_use_conv: bool = True,
):
    if block_type == 'UNetDownBlock':
        return UNetDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            n_blocks=n_blocks,
            norm_groups=norm_groups,
            activation=activation,
            embedding_dim=embedding_dim,
            embedding_mixing=embedding_mixing,
            dropout=dropout,
            downsample=downsample,
            downsample_use_conv=downsample_use_conv,
        )
    elif block_type == 'UNetAttnDownBlock':
        return UNetAttnDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_dim_head=n_dim_head,
            norm_groups=norm_groups,
            activation=activation,
            embedding_dim=embedding_dim,
            embedding_mixing=embedding_mixing,
            dropout=dropout,
            downsample=downsample,
            downsample_use_conv=downsample_use_conv,
        )
    elif block_type == 'EncoderDownBlock':
        return EncoderDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            n_blocks=n_blocks,
            norm_groups=norm_groups,
            activation=activation,
            dropout=dropout,
            downsample=downsample,
            downsample_use_conv=downsample_use_conv,
        )
    elif block_type == 'EncoderAttnDownBlock':
        return EncoderAttnDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_dim_head=n_dim_head,
            norm_groups=norm_groups,
            activation=activation,
            dropout=dropout,
            downsample=downsample,
            downsample_use_conv=downsample_use_conv,
        )
    else:
        raise ValueError(f'unknown down block type "{block_type}"')


def get_up_block(
        block_type: str,
        in_channels: int,
        out_channels: int,
        next_out_channels: Optional[int] = None,
        n_blocks: int = 2,
        n_heads: Optional[int] = None,
        n_dim_head: Optional[int] = None,
        norm_groups: int = 32,
        activation: str = 'swish',
        embedding_dim: Optional[int] = None,
        embedding_mixing: str = 'addition',
        dropout: float = 0.,
        upsample: bool = True,
        upsample_use_conv: bool = True,
):
    if block_type == 'UNetUpBlock':
        return UNetUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            next_out_channels=next_out_channels,
            n_blocks=n_blocks,
            norm_groups=norm_groups,
            activation=activation,
            embedding_dim=embedding_dim,
            embedding_mixing=embedding_mixing,
            dropout=dropout,
            upsample=upsample,
            upsample_use_conv=upsample_use_conv,
        )
    elif block_type == 'UNetAttnUpBlock':
        return UNetAttnUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            next_out_channels=next_out_channels,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_dim_head=n_dim_head,
            norm_groups=norm_groups,
            activation=activation,
            embedding_dim=embedding_dim,
            embedding_mixing=embedding_mixing,
            dropout=dropout,
            upsample=upsample,
            upsample_use_conv=upsample_use_conv,
        )
    elif block_type == 'DecoderUpBlock':
        return DecoderUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            n_blocks=n_blocks,
            norm_groups=norm_groups,
            activation=activation,
            dropout=dropout,
            upsample=upsample,
            upsample_use_conv=upsample_use_conv,
        )
    elif block_type == 'DecoderAttnUpBlock':
        return DecoderAttnUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_dim_head=n_dim_head,
            norm_groups=norm_groups,
            activation=activation,
            dropout=dropout,
            upsample=upsample,
            upsample_use_conv=upsample_use_conv,
        )
    else:
        raise ValueError(f'unknown up block type "{block_type}"')



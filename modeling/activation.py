
from torch import nn


def get_activation(name: str):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'swish' or name == 'silu':
        return nn.SiLU()
    else:
        raise ValueError(f'unknown activation "{name}"')



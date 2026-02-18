

from torch import nn


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_named_parameters(model: nn.Module, verbose=False):
    total_n_param = 0
    for name, param in model.named_parameters():
        n_param = param.numel()
        if verbose:
            print(f'{name}: {n_param}')
        if param.requires_grad:
            total_n_param += n_param
    return total_n_param





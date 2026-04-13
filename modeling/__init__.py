
from modeling.predictor import BasePredictor
from modeling.unet import UNet
from modeling.transformer import JITTransformer


def get_model(name, **kwargs) -> BasePredictor:
    if name == 'unet':
        return UNet(**kwargs)
    elif name == 'jit':
        return JITTransformer(**kwargs)
    else:
        raise ValueError(f'Unknown model name: {name}')

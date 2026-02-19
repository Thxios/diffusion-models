
from predictor import BasePredictor
from modeling.unet import UNet


def get_model(name, **kwargs) -> BasePredictor:
    if name == 'unet':
        return UNet(**kwargs)
    else:
        raise ValueError(f'Unknown model name: {name}')

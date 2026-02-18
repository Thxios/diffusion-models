
from modeling.unet import UNet


def get_model(name, **kwargs):
    if name == 'unet':
        return UNet(**kwargs)
    else:
        raise ValueError(f'Unknown model name: {name}')

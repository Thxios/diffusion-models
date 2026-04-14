
import torch


class BasePredictor:
    pass


class GuidedPredictor(BasePredictor):
    def pred_conditional(self, z, t, cond=None, uncond_mask=None, **kwargs):
        raise NotImplementedError()

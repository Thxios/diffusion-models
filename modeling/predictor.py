
import torch
from typing import Callable


class BasePredictor:
    def get_pred_fn(self, args, **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        raise NotImplementedError()


class GuidedPredictor(BasePredictor):
    def pred_conditional(self, z, t, cond=None, uncond_mask=None):
        raise NotImplementedError()

    def get_pred_fn(self, cond=None, guidance_scale=1.0):
        def pred_fn(z, t):
            if cond is None or guidance_scale == 0.0:
                pred = self.pred_conditional(z, t)
            elif guidance_scale == 1.0:
                pred = self.pred_conditional(z, t, cond=cond)
            else:
                pred = self.pred_conditional(
                    torch.repeat_interleave(z, 2, dim=0), 
                    torch.repeat_interleave(t, 2, dim=0), 
                    cond=torch.repeat_interleave(cond, 2, dim=0), 
                    uncond_mask=torch.cat([torch.ones_like(cond), torch.zeros_like(cond)], dim=0)
                )
                pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            return pred
        
        return pred_fn

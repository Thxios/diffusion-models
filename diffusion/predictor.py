
import torch
from torch import nn
from typing import Optional
import diffusers


class GuidedNoisePredictor(nn.Module):
    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: Optional[torch.Tensor] = None,
            uncond_mask: Optional[torch.Tensor] = None
    ):
        raise NotImplementedError()

    @torch.no_grad()
    def predict_guided_noise(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: Optional[torch.Tensor] = None,
            guidance_scale=0.,
    ):
        if guidance_scale == -1.:
            cond = None

        if cond is None or guidance_scale == 0.:
            eps = self.forward(x, t, cond=cond)
        else:
            uncond_mask = torch.cat(
                (torch.zeros(x.size(0), device=x.device),
                 torch.ones(x.size(0), device=x.device)),
                dim=0
            )
            x = x.repeat(2, 1, 1, 1)
            t = t.repeat(2)
            cond = cond.repeat(2)

            eps, eps_uncond = self.forward(
                x, t, cond=cond, uncond_mask=uncond_mask
            ).chunk(2)
            eps = (1 + guidance_scale) * eps - guidance_scale * eps_uncond

        return eps



class ClsCondUNetPredictor(GuidedNoisePredictor):
    def __init__(self, **kwargs):
        super().__init__()

        num_classes = kwargs.get('num_class_embeds', None)
        if num_classes is not None:
            uncond_cls = num_classes
            kwargs.update(num_class_embeds=num_classes + 1)
        else:
            raise ValueError('"num_class_embeds" should be provided')

        self.unet = diffusers.UNet2DModel(**kwargs)
        self.register_buffer('uncond_cls', torch.tensor(uncond_cls, dtype=torch.int64))

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: Optional[torch.LongTensor] = None,
            uncond_mask: Optional[torch.Tensor] = None
    ):
        if cond is None:
            cond = torch.full(
                (x.size(0),),
                fill_value=self.uncond_cls,
                dtype=torch.int64,
                device=x.device
            )
        elif uncond_mask is not None:
            cond = cond.masked_fill(uncond_mask.bool(), self.uncond_cls)

        return self.unet(x, t, class_labels=cond).sample


class FourierFeaturePredictor(ClsCondUNetPredictor):
    def __init__(self, freqs=(64, 128), **kwargs):
        self.n_freqs = len(freqs)
        in_channels = kwargs.get('in_channels', 3)
        encoded_channels = in_channels * (1 + 2 * self.n_freqs)
        kwargs.update(in_channels=encoded_channels)

        super().__init__(**kwargs)

        freqs = torch.tensor(freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

    def encode_fourier(self, x: torch.Tensor):
        bs, c, h, w = x.shape

        xt = x.unsqueeze(1) * self.freqs.view(1, -1, 1, 1, 1)
        xt = xt.view(bs, -1, w, h)

        sin_, cos_ = torch.sin(xt), torch.cos(xt)
        ret = torch.cat((x, sin_, cos_), dim=1)
        return ret

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: Optional[torch.LongTensor] = None,
            uncond_mask: Optional[torch.Tensor] = None
    ):
        x = self.encode_fourier(x)

        return super().forward(
            x, t,
            cond=cond,
            uncond_mask=uncond_mask
        )




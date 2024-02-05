

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from functools import partial

from diffusion.scheduler import ContinuousScheduler, DiscreteScheduler
from diffusion.sampler import ContinuousSampler, BaseSampler
from diffusion.predictor import GuidedNoisePredictor


class BaseDiffusion(nn.Module):
    def train_step_loss(self, x, cond=None):
        raise NotImplementedError()

    def sample(self, z, sampler: BaseSampler, cond=None, guidance_scale=0.):
        raise NotImplementedError()


class ContinuousDiffusion(BaseDiffusion):
    def __init__(
            self,
            model: GuidedNoisePredictor,
            scheduler: ContinuousScheduler,
            p_uncond=0.2
    ):
        super().__init__()

        self.model = model
        self.scheduler = scheduler
        self.register_buffer('p_uncond', torch.tensor(p_uncond))

    def train_step_loss(
            self,
            x: torch.Tensor,
            cond: Optional[torch.Tensor] = None,
    ):
        t = torch.rand(x.size(0), dtype=torch.float64, device=x.device)
        t = self.scheduler.get_schedule(t)

        eps = torch.randn_like(x)
        x_t = self.scheduler.diffuse(x, eps, schedule=t)

        uncond_mask = torch.bernoulli(self.p_uncond * torch.ones_like(cond))
        eps_pred = self.model(
            x_t,
            t.log_snr,
            cond=cond,
            uncond_mask=uncond_mask
        )

        loss = F.mse_loss(eps_pred, eps)
        return loss

    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            sampler: ContinuousSampler,
            cond: Optional[torch.Tensor] = None,
            guidance_scale=0.
    ):
        pred_fn = partial(
            self.model.predict_guided_noise,
            cond=cond,
            guidance_scale=guidance_scale,
        )

        x0_tilde = sampler.sample(z, self.scheduler, pred_fn)
        return x0_tilde



class DiscreteDiffusion(BaseDiffusion):
    def __init__(
            self,
            model: GuidedNoisePredictor,
            scheduler: DiscreteScheduler,
            p_uncond=0.2,
    ):
        super().__init__()

        self.model = model
        self.scheduler = scheduler
        self.register_buffer('p_uncond', torch.tensor(p_uncond))

    def train_step_loss(
            self,
            x: torch.Tensor,
            cond: Optional[torch.Tensor] = None,
    ):
        t = torch.randint(0, self.scheduler.n_steps,
                          size=(x.size(0),),
                          device=x.device)

        eps = torch.randn_like(x)
        x_t = self.scheduler.diffuse(x, eps, t=t)

        uncond_mask = torch.bernoulli(self.p_uncond * torch.ones_like(cond))
        eps_pred = self.model(
            x_t,
            t,
            cond=cond,
            uncond_mask=uncond_mask
        )

        loss = F.mse_loss(eps_pred, eps)
        return loss

    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            sampler: DiscreteScheduler,
            cond: Optional[torch.Tensor] = None,
            guidance_scale=0.
    ):
        pred_fn = partial(
            self.model.predict_guided_noise,
            cond=cond,
            guidance_scale=guidance_scale,
        )

        x0_tilde = sampler.sample(z, self.scheduler, pred_fn)
        return x0_tilde


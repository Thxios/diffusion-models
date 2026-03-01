

import torch
import tqdm.auto as tqdm
from typing import Callable, Optional

from diffusion.scheduler import VariancePreservingScheduler, BetaScheduler, \
    RectifiedFlowScheduler


def get_sampler(name, **kwargs):
    if name == 'ddpm':
        return DDPMSampler(**kwargs)
    elif name == 'ddim':
        return DDIMSampler(**kwargs)
    elif name == 'euler':
        return RectifiedFlowEulerSampler(**kwargs)
    else:
        raise ValueError(f'unknown sampler {name}')
    

class BaseSampler:
    pred_type = 'noise'

    def __init__(self, n_steps, pbar=False, pbar_kwargs=None):
        self.n_steps = n_steps
        self.pbar = pbar
        self.pbar_kwargs = {'leave': False}
        if pbar_kwargs is not None:
            self.pbar_kwargs.update(pbar_kwargs)
        
    def set_steps(self, n_steps):
        self.n_steps = n_steps

    def prepare_iterator(self, iterator):
        if self.pbar:
            iterator = tqdm.tqdm(iterator, **self.pbar_kwargs)
        return iterator

    def sample(
            self,
            z: torch.Tensor,
            scheduler,
            pred_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            **kwargs
    ):
        raise NotImplementedError()


class DDPMSampler(BaseSampler):
    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: BetaScheduler,
            pred_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            seed: Optional[int] = None
    ):
        gen = torch.Generator(device=z.device)
        if seed is not None:
            gen.manual_seed(seed)

        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device
        )[:-1]

        # align notation with ddpm paper
        beta = scheduler.beta.to(device=z.device, dtype=z.dtype)
        coef = beta / torch.sqrt(1 - scheduler.alpha_sq).to(device=z.device, dtype=z.dtype) # (1 - alpha_t) / sqrt(1 - alpha_bar_t)
        inv_alpha_t_sqrt = 1 / torch.sqrt(1 - beta)
        sigma_t = torch.sqrt(beta)

        iterator = self.prepare_iterator(range(self.n_steps))
        for i in iterator:
            t = t_steps[i]
            eps_pred = pred_fn(z, t.repeat(z.size(0)))
            z = inv_alpha_t_sqrt[t] * (z - coef[t] * eps_pred)
            
            if i < self.n_steps - 1:
                noise = torch.randn_like(z, generator=gen)
                z = z + sigma_t[t] * noise

        return z



class DDIMSampler(BaseSampler):
    def __init__(self, n_steps, eta=0.0, clip_latent=True, pbar=False, pbar_kwargs=None):
        super().__init__(n_steps, pbar, pbar_kwargs)
        self.eta = eta
        self.clip_latent = clip_latent
    
    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: VariancePreservingScheduler,
            pred_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):

        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device
        )[:-1]

        iterator = self.prepare_iterator(range(self.n_steps))
        for i in iterator:
            eps_pred = pred_fn(z, t_steps[i].repeat(z.size(0)))
            t = scheduler.get_schedule(t_steps[i])

            alpha_t_ = torch.sqrt(t.alpha_sq).to(z.dtype)
            sigma_t_ = torch.sqrt(t.sigma_sq).to(z.dtype)
            x0_pred = (z - sigma_t_ * eps_pred) / alpha_t_
            if self.clip_latent:
                x0_pred = torch.clip(x0_pred, -1, 1)

            if i < self.n_steps - 1:
                t_prime = scheduler.get_schedule(t_steps[i + 1])

                alpha_t_prime_ = torch.sqrt(t_prime.alpha_sq).to(z.dtype)
                sigma_t_prime_ = torch.sqrt(t_prime.sigma_sq).to(z.dtype)
                z = alpha_t_prime_ * x0_pred + sigma_t_prime_ * eps_pred

            else:
                z = x0_pred

        return z
    

class RectifiedFlowEulerSampler(BaseSampler):
    pred_type = 'velocity'

    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: RectifiedFlowScheduler,
            pred_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device
        )[:-1]
        iterator = self.prepare_iterator(range(self.n_steps))

        for i in iterator:
            v_pred = pred_fn(z, t_steps[i].repeat(z.size(0)))
            # optimal: v_pred = x0 - eps

            sigma_t = scheduler.sigmas[t_steps[i]]
            sigma_next = scheduler.sigmas[t_steps[i + 1]] if i < self.n_steps - 1 else 0
            dt = sigma_next - sigma_t
            # dt < 0 since sigmas are increasing, so we are doing Euler *backward* steps

            z = z - dt * v_pred

        return z

    

if __name__ == '__main__':
    from diffusion.scheduler import RectifiedFlowScheduler

    scheduler = RectifiedFlowScheduler(n_steps=1000)
    sampler = RectifiedFlowEulerSampler(n_steps=50)
    z = torch.randn(2, 3)
    pred_fn = lambda x, t: torch.randn_like(x)
    samples = sampler.sample(z, scheduler, pred_fn)

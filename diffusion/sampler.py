

import torch
import tqdm.auto as tqdm
from typing import Callable, Optional

from diffusion.scheduler import VariancePreservingScheduler, BetaScheduler, \
    RectifiedFlowScheduler, JITScheduler


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
            return_intermediates=False,
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
            return_intermediates=False,
            gen: Optional[torch.Generator] = None
    ):
        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device
        )[:-1]

        schedule = scheduler.get_schedule(t_steps)
        beta = scheduler.beta.to(device=z.device, dtype=z.dtype)[t_steps]
        coef = beta / schedule.sigma.to(device=z.device, dtype=z.dtype)
        inv_alpha_sqrt = 1 / torch.sqrt(1 - beta)
        sigma = torch.sqrt(beta)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        for i in iterator:
            eps_pred = pred_fn(z, t_steps[i].repeat(z.size(0)))
            z = inv_alpha_sqrt[i] * (z - coef[i] * eps_pred)
            
            if i < self.n_steps - 1:
                noise = torch.randn_like(z, generator=gen)
                z = z + sigma[i] * noise
            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            return z, torch.stack(intermediates, dim=0)
        else:
            return z


class DDIMSampler(BaseSampler):
    def __init__(self, n_steps, eta=0.0, clip_latent=True, pbar=False, pbar_kwargs=None):
        super().__init__(n_steps, pbar, pbar_kwargs)
        # TODO: implement stochastic sampling when eta > 0
        self.eta = eta
        self.clip_latent = clip_latent
    
    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: VariancePreservingScheduler,
            pred_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            return_intermediates=False
    ):
        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device
        )[:-1]

        schedule = scheduler.get_schedule(t_steps)
        alpha = schedule.alpha.to(device=z.device, dtype=z.dtype)
        sigma = schedule.sigma.to(device=z.device, dtype=z.dtype)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        for i in iterator:
            eps_pred = pred_fn(z, t_steps[i].repeat(z.size(0)))

            x0_pred = (z - sigma[i] * eps_pred) / alpha[i]
            if self.clip_latent:
                x0_pred = torch.clip(x0_pred, -1, 1)

            if i < self.n_steps - 1:
                z = alpha[i + 1] * x0_pred + sigma[i + 1] * eps_pred
            else:
                z = x0_pred
            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            return z, torch.stack(intermediates, dim=0)
        else:
            return z

    
class DPMpp2MSolver(BaseSampler):
    def __init__(self, n_steps, clip_latent=True, pbar=False, pbar_kwargs=None):
        super().__init__(n_steps, pbar, pbar_kwargs)
        self.clip_latent = clip_latent

    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: VariancePreservingScheduler,
            pred_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            return_intermediates=False
    ):
        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device
        )  # total N+1 steps (in order to fit NFE to N)

        schedule = scheduler.get_schedule(t_steps)
        alpha = schedule.alpha.to(device=z.device, dtype=z.dtype)
        sigma = schedule.sigma.to(device=z.device, dtype=z.dtype)
        # h[i] is h_{i+1} (due to shifting)
        h = torch.diff(0.5 * schedule.log_snr).to(device=z.device, dtype=z.dtype)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        prev_x0_pred = None

        for i in iterator:
            eps_pred = pred_fn(z, t_steps[i].repeat(z.size(0)))

            x0_pred = (z - sigma[i] * eps_pred) / alpha[i]
            if self.clip_latent:
                x0_pred = torch.clip(x0_pred, -1, 1)

            if i == 0:
                d = x0_pred
            else:
                inv_2r = 0.5 * h[i] / h[i - 1]
                d = (1 + inv_2r) * x0_pred - inv_2r * prev_x0_pred
            
            z = (sigma[i + 1] / sigma[i]) * z - (alpha[i + 1] * torch.expm1(-h[i])) * d
            prev_x0_pred = x0_pred

            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            return z, torch.stack(intermediates, dim=0)
        else:
            return z


class RectifiedFlowEulerSampler(BaseSampler):
    pred_type = 'rect_flow'

    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: RectifiedFlowScheduler,
            pred_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            return_intermediates=False
    ):
        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device
        )[:-1]
        sigmas = scheduler.sigmas.to(device=z.device, dtype=z.dtype)
        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        for i in iterator:
            v_pred = pred_fn(z, t_steps[i].repeat(z.size(0)))
            # optimal: v_pred = x0 - eps

            sigma_t = sigmas[t_steps[i]]
            sigma_next = sigmas[t_steps[i + 1]] if i < self.n_steps - 1 else sigmas.new_zeros(1)
            dt = sigma_next - sigma_t
            # dt < 0 since sigmas are increasing, so we are doing Euler *backward* steps

            z = z - dt * v_pred

            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            return z, torch.stack(intermediates, dim=0)
        else:
            return z


from modeling.predictor import GuidedPredictor

class JITSampler(BaseSampler):
    pred_type = 'data'
    def __init__(
            self, 
            n_steps, 
            scheduler: JITScheduler, 
            ode_solver='euler',
            guidance_scale=1.0,
            cfg_interval: Optional[tuple[float, float]] = None,
            pbar=False, 
            pbar_kwargs=None
    ):
        super().__init__(n_steps, pbar, pbar_kwargs)
        self.scheduler: JITScheduler = scheduler

        self.ode_solver = ode_solver
        self.guidance_scale = guidance_scale
        self.cfg_interval = cfg_interval

    @torch.no_grad()
    def pred_v_guided(
            self, 
            model: GuidedPredictor, 
            z, t, cond=None, 
    ):
        x_cond = model.pred_conditional(z, t, cond=cond)
        v_cond = self.scheduler.x_to_v(x_cond, z, t)
        if self.guidance_scale == 1.0 or cond is None:
            return v_cond
        
        x_uncond = model.pred_conditional(z, t)
        v_uncond = self.scheduler.x_to_v(x_uncond, z, t)

        if self.cfg_interval is None:
            cfg = self.guidance_scale
        else:
            low, high = self.cfg_interval
            interval_mask = (t < high) & ((low == 0) | (t > low))
            cfg = torch.where(interval_mask, self.guidance_scale, 1.0)
            cfg = cfg.view(-1, *([1] * (z.ndim - 1)))

        v_guided = v_uncond + cfg * (v_cond - v_uncond)
        return v_guided

    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            model: GuidedPredictor,
            cond: Optional[torch.Tensor] = None,
            return_intermediates=False
    ):
        t = torch.linspace(0, 1, steps=self.n_steps + 1, device=z.device)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        for i in iterator:
            v_pred = self.pred_v_guided(model, z, t[i].repeat(z.size(0)), cond=cond)

            if self.ode_solver == 'euler':
                z = z + (t[i + 1] - t[i]) * v_pred
            elif self.ode_solver == 'heun':
                z_euler = z + (t[i + 1] - t[i]) * v_pred
                v_pred_next = self.pred_v_guided(model, z_euler, t[i + 1].repeat(z.size(0)), cond=cond)
                z = z + (t[i + 1] - t[i]) * 0.5 * (v_pred + v_pred_next)

            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            return z, torch.stack(intermediates, dim=0)
        else:
            return z

    

if __name__ == '__main__':
    from diffusion.scheduler import RectifiedFlowScheduler

    scheduler = RectifiedFlowScheduler(n_steps=1000)
    sampler = RectifiedFlowEulerSampler(n_steps=50)
    z = torch.randn(2, 3)
    pred_fn = lambda x, t: torch.randn_like(x)
    samples = sampler.sample(z, scheduler, pred_fn)

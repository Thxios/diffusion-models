

import torch
import tqdm.auto as tqdm
from typing import Optional


def get_sampler(name, scheduler, **kwargs):
    return {
        'ddpm':  DDPMSampler,
        'ddim':  DDIMSampler,
        'euler': RectifiedFlowEulerSampler,
        'jit':   JITSampler,
    }[name](scheduler=scheduler, **kwargs)


class BaseSampler:
    pred_type_out = 'noise'  # what sample() expects as the model's effective output

    def __init__(
            self,
            scheduler,
            n_steps,
            guidance_scale=1.0,
            cfg_interval=None,
            pbar=False,
            pbar_kwargs=None,
    ):
        self.scheduler = scheduler
        self.n_steps = n_steps
        self.guidance_scale = guidance_scale
        self.cfg_interval = cfg_interval
        self.pbar = pbar
        self.pbar_kwargs = {'leave': False, **(pbar_kwargs or {})}

    def prepare_iterator(self, iterator):
        if self.pbar:
            iterator = tqdm.tqdm(iterator, **self.pbar_kwargs)
        return iterator

    @torch.no_grad()
    def _predict(self, model, z, t, cond):
        """Run model with CFG and return the sampler-appropriate prediction type."""
        if self.guidance_scale == 1.0 or cond is None:
            return self._to_pred_type(model.pred_conditional(z, t, cond=cond), z, t)

        # Batch-doubled CFG forward pass
        z2 = torch.cat([z, z], dim=0)
        t2 = torch.cat([t, t], dim=0)
        cond2 = torch.cat([cond, cond], dim=0)
        uncond_mask2 = torch.cat([
            torch.ones(len(cond), dtype=torch.bool, device=cond.device),
            torch.zeros(len(cond), dtype=torch.bool, device=cond.device),
        ], dim=0)
        pred = model.pred_conditional(z2, t2, cond=cond2, uncond_mask=uncond_mask2)
        pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)

        p_uncond = self._to_pred_type(pred_uncond, z, t)
        p_cond   = self._to_pred_type(pred_cond,   z, t)

        cfg = self._cfg_at(t, z.ndim)
        return p_uncond + cfg * (p_cond - p_uncond)

    def _cfg_at(self, t, ndim):
        if self.cfg_interval is None:
            return self.guidance_scale
        low, high = self.cfg_interval
        mask = (t < high) & ((low == 0) | (t > low))
        cfg = torch.where(mask,
                          torch.full_like(t, self.guidance_scale),
                          torch.ones_like(t))
        return cfg.view(-1, *([1] * (ndim - 1)))

    def _to_pred_type(self, model_out, z, t):
        """Convert raw model output to the prediction type this sampler consumes.
        Default: identity (noise predictor → noise consumer)."""
        return model_out

    def sample(self, z, model, cond=None, return_intermediates=False, **kwargs):
        raise NotImplementedError


class DDPMSampler(BaseSampler):
    pred_type_out = 'noise'

    @torch.no_grad()
    def sample(self, z, model, cond=None, return_intermediates=False,
               gen: Optional[torch.Generator] = None):
        from diffusion.scheduler import BetaScheduler
        scheduler: BetaScheduler = self.scheduler

        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device,
        )[:-1]

        schedule = scheduler.get_schedule(t_steps)
        beta = scheduler.beta.to(device=z.device, dtype=z.dtype)[t_steps]
        coef = beta / schedule.sigma.to(device=z.device, dtype=z.dtype)
        inv_alpha_sqrt = 1 / torch.sqrt(1 - beta)
        sigma = torch.sqrt(beta)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        for i in iterator:
            eps_pred = self._predict(model, z, t_steps[i].repeat(z.size(0)), cond)
            z = inv_alpha_sqrt[i] * (z - coef[i] * eps_pred)

            if i < self.n_steps - 1:
                noise = torch.randn_like(z, generator=gen)
                z = z + sigma[i] * noise
            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            return z, torch.stack(intermediates, dim=0)
        return z


class DDIMSampler(BaseSampler):
    pred_type_out = 'noise'

    def __init__(self, scheduler, n_steps, eta=0.0, clip_latent=True,
                 guidance_scale=1.0, cfg_interval=None, pbar=False, pbar_kwargs=None):
        super().__init__(scheduler, n_steps, guidance_scale, cfg_interval, pbar, pbar_kwargs)
        self.eta = eta
        self.clip_latent = clip_latent

    @torch.no_grad()
    def sample(self, z, model, cond=None, return_intermediates=False,
               initial_timestep=None, return_x0_preds=False):
        from diffusion.scheduler import VariancePreservingScheduler
        scheduler: VariancePreservingScheduler = self.scheduler

        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device,
        )[:-1]

        schedule = scheduler.get_schedule(t_steps)
        alpha = schedule.alpha.to(device=z.device, dtype=z.dtype)
        sigma = schedule.sigma.to(device=z.device, dtype=z.dtype)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        x0_preds = []
        for i in iterator:
            eps_pred = self._predict(model, z, t_steps[i].repeat(z.size(0)), cond)

            x0_pred = (z - sigma[i] * eps_pred) / alpha[i]
            if self.clip_latent:
                x0_pred = torch.clip(x0_pred, -1, 1)

            if return_x0_preds:
                x0_preds.append(x0_pred.cpu())

            if i < self.n_steps - 1:
                z = alpha[i + 1] * x0_pred + sigma[i + 1] * eps_pred
            else:
                z = x0_pred
            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            ret = (z, torch.stack(intermediates, dim=0))
            if return_x0_preds:
                ret = ret + (torch.stack(x0_preds, dim=0),)
            return ret
        if return_x0_preds:
            return z, torch.stack(x0_preds, dim=0)
        return z


class DPMpp2MSolver(BaseSampler):
    pred_type_out = 'noise'

    def __init__(self, scheduler, n_steps, clip_latent=True,
                 guidance_scale=1.0, cfg_interval=None, pbar=False, pbar_kwargs=None):
        super().__init__(scheduler, n_steps, guidance_scale, cfg_interval, pbar, pbar_kwargs)
        self.clip_latent = clip_latent

    @torch.no_grad()
    def sample(self, z, model, cond=None, return_intermediates=False):
        from diffusion.scheduler import VariancePreservingScheduler
        scheduler: VariancePreservingScheduler = self.scheduler

        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device,
        )

        schedule = scheduler.get_schedule(t_steps)
        alpha = schedule.alpha.to(device=z.device, dtype=z.dtype)
        sigma = schedule.sigma.to(device=z.device, dtype=z.dtype)
        h = torch.diff(0.5 * schedule.log_snr).to(device=z.device, dtype=z.dtype)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        prev_x0_pred = None

        for i in iterator:
            eps_pred = self._predict(model, z, t_steps[i].repeat(z.size(0)), cond)

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
        return z


class RectifiedFlowEulerSampler(BaseSampler):
    pred_type_out = 'rect_flow'

    @torch.no_grad()
    def sample(self, z, model, cond=None, return_intermediates=False):
        from diffusion.scheduler import RectifiedFlowScheduler
        scheduler: RectifiedFlowScheduler = self.scheduler

        t_steps = torch.linspace(
            scheduler.n_steps - 1, 0,
            steps=self.n_steps + 1,
            dtype=torch.int64,
            device=z.device,
        )[:-1]
        sigmas = scheduler.sigmas.to(device=z.device, dtype=z.dtype)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        for i in iterator:
            v_pred = self._predict(model, z, t_steps[i].repeat(z.size(0)), cond)

            sigma_t = sigmas[t_steps[i]]
            sigma_next = sigmas[t_steps[i + 1]] if i < self.n_steps - 1 else sigmas.new_zeros(1)
            dt = sigma_next - sigma_t
            z = z - dt * v_pred

            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            return z, torch.stack(intermediates, dim=0)
        return z


class JITSampler(BaseSampler):
    pred_type_out = 'data'

    def __init__(self, scheduler, n_steps, ode_solver='euler',
                 guidance_scale=1.0, cfg_interval=None, pbar=False, pbar_kwargs=None):
        super().__init__(scheduler, n_steps, guidance_scale, cfg_interval, pbar, pbar_kwargs)
        self.ode_solver = ode_solver

    def _to_pred_type(self, model_out, z, t):
        """JIT model predicts x; convert to v for the ODE solver."""
        return self.scheduler.x_to_v(model_out, z, t)

    @torch.no_grad()
    def sample(self, z, model, cond=None, return_intermediates=False):
        t = torch.linspace(0, 1, steps=self.n_steps + 1, device=z.device)

        iterator = self.prepare_iterator(range(self.n_steps))
        intermediates = []
        for i in iterator:
            t_i = t[i].repeat(z.size(0))
            v_pred = self._predict(model, z, t_i, cond)

            if self.ode_solver == 'euler':
                z = z + (t[i + 1] - t[i]) * v_pred
            elif self.ode_solver == 'heun':
                z_euler = z + (t[i + 1] - t[i]) * v_pred
                t_next = t[i + 1].repeat(z.size(0))
                v_next = self._predict(model, z_euler, t_next, cond)
                z = z + (t[i + 1] - t[i]) * 0.5 * (v_pred + v_next)

            if return_intermediates:
                intermediates.append(z.cpu())

        if return_intermediates:
            return z, torch.stack(intermediates, dim=0)
        return z

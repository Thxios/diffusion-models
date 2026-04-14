
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Callable
import math


def get_scheduler(name, **kwargs):
    if name == 'beta':
        return BetaScheduler(**kwargs)
    elif name == 'rectified_flow':
        return RectifiedFlowScheduler(**kwargs)
    elif name == 'jit':
        return JITScheduler(**kwargs)
    else:
        raise ValueError(f'unknown scheduler type {name}')


class BaseScheduler:
    def get_loss(self, x: torch.Tensor, model, gen=None, **model_call_kwargs):
        raise NotImplementedError()

    def diffuse(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()


@dataclass
class Schedule:
    alpha: torch.Tensor
    sigma: torch.Tensor
    log_snr: torch.Tensor

@dataclass
class VPSchedule(Schedule):
    @staticmethod
    def from_alpha(alpha):
        sigma = torch.sqrt(1 - alpha ** 2)
        return VPSchedule(
            alpha=alpha,
            sigma=sigma,
            log_snr=2 * torch.log(alpha / sigma)
        )

    @staticmethod
    def from_log_snr(log_snr):
        alpha_sq = torch.sigmoid(log_snr)
        alpha = torch.sqrt(alpha_sq)
        sigma = torch.sqrt(1 - alpha_sq)
        return VPSchedule(
            alpha=alpha,
            sigma=sigma,
            log_snr=log_snr
        )

@dataclass
class RectFlowSchedule(Schedule):
    @staticmethod
    def from_alpha(alpha):
        sigma = 1 - alpha
        return RectFlowSchedule(
            alpha=alpha,
            sigma=sigma,
            log_snr=2 * torch.log(alpha / sigma)
        )

    @staticmethod
    def from_log_snr(log_snr):
        alpha_sq = torch.sigmoid(log_snr)
        alpha = torch.sqrt(alpha_sq)
        return RectFlowSchedule(
            alpha=alpha,
            sigma=1 - alpha,
            log_snr=log_snr
        )
    

class VariancePreservingScheduler(BaseScheduler):
    def __init__(self, n_steps, pred_type='noise'):
        self.n_steps = n_steps

        assert pred_type in ['noise', 'velocity']
        self.pred_type = pred_type

    def get_schedule(self, t: torch.LongTensor) -> VPSchedule:
        raise NotImplementedError()
        
    def get_loss(
            self, 
            x: torch.Tensor, 
            model, 
            gen=None, 
            **model_call_kwargs
    ):
        eps = torch.randn_like(x, generator=gen)
        t = torch.randint(
            0, self.n_steps, size=(x.size(0),), 
            device=x.device,
            generator=gen
        )
        schedule = self.get_schedule(t)

        shape = (-1,) + (1,) * (len(x.shape) - 1)
        alpha = schedule.alpha.to(dtype=x.dtype, device=x.device).view(*shape)
        sigma = schedule.sigma.to(dtype=x.dtype, device=x.device).view(*shape)

        x_t = alpha * x + sigma * eps
        pred = model.pred_conditional(x_t, t, **model_call_kwargs)

        if self.pred_type == 'noise':
            target = eps
        elif self.pred_type == 'velocity':
            target = alpha * eps - sigma * x
        else:
            raise ValueError(f'unknown pred type {self.pred_type}')

        loss = F.mse_loss(pred, target)
        return loss

    def diffuse(self, x, t, eps):
        schedule = self.get_schedule(t)

        shape = (-1,) + (1,) * (len(x.shape) - 1)
        alpha_ = schedule.alpha.to(dtype=x.dtype, device=x.device).view(*shape)
        sigma_ = schedule.sigma.to(dtype=x.dtype, device=x.device).view(*shape)

        x_t = alpha_ * x + sigma_ * eps
        return x_t


class BetaScheduler(VariancePreservingScheduler):
    def __init__(
            self,
            schedule='linear',
            beta_start=1e-4,
            beta_end=2e-2,
            cos_s=0.008,
            max_beta=0.999,
            n_steps=1000,
            pred_type='noise',
    ):
        super().__init__(n_steps=n_steps, pred_type=pred_type)

        if schedule == 'linear':
            beta = torch.linspace(beta_start, beta_end, steps=n_steps, dtype=torch.float64)
        elif schedule == 'cosine':
            beta = BetaScheduler.get_betas_for_alpha(
                'cosine', 
                cos_s=cos_s, 
                max_beta=max_beta, 
                n_steps=n_steps
            )
        elif schedule == 'square':
            beta = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5,
                steps=n_steps, dtype=torch.float64
            ) ** 2
        elif schedule == 'laplace':
            beta = BetaScheduler.get_betas_for_alpha(
                'laplace', 
                max_beta=max_beta, 
                n_steps=n_steps
            )
        elif schedule == 'sigmoid':
            sig_x = torch.linspace(-6, 6, n_steps)
            beta = torch.sigmoid(sig_x) * (beta_end - beta_start) + beta_start
        else:
            raise ValueError(f'unknown schedule type {schedule}')
        self.schedule = schedule

        self.beta = beta
        self.alpha_sq = torch.cumprod(1 - beta, dim=0)

    def get_schedule(self, t: torch.LongTensor) -> VPSchedule:
        alpha = torch.sqrt(self.alpha_sq.to(t.device)[t])
        return VPSchedule.from_alpha(alpha)

    @staticmethod
    def get_betas_for_alpha(alpha_type, cos_s=0.008, max_beta=0.999, n_steps=1000):
        if alpha_type == 'cosine':
            def alpha_bar_fn(t):
                return math.cos((t + cos_s) / (1 + cos_s) * math.pi / 2) ** 2
        elif alpha_type == 'laplace':
            def alpha_bar_fn(t):
                lmb = -0.5 * math.copysign(1, 0.5 - t) * math.log(1 - 2 * math.fabs(0.5 - t) + 1e-6)
                snr = math.exp(lmb)
                return math.sqrt(snr / (1 + snr))
        else:
            raise ValueError(f'unknown alpha type {alpha_type}')

        beta = []
        for i in range(n_steps):
            t1, t2 = i / n_steps, (i + 1) / n_steps
            beta.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        beta = torch.tensor(beta, dtype=torch.float64)
        return beta


class LogSNRScheduler:
    def __init__(
            self,
            distribution='gaussian',
            mean=0.0,
            std=1.0,
            low=-20.0,
            high=20.0,
            cos_s=0.008,
            cos_shift=0.0,
            cos_eps=1e-7,
    ):
        if distribution == 'gaussian':
            self.distribution = distribution
            self.dist_kwargs = {'mean': mean, 'std': std}
        elif distribution == 'uniform':
            self.distribution = distribution
            self.dist_kwargs = {'low': low, 'high': high}
        elif distribution == 'cosine':
            self.distribution = 'cosine'
            self.dist_kwargs = {'s': cos_s, 'shift': cos_shift, 'eps': cos_eps}
        else:
            raise ValueError(f'unknown distribution type {distribution}')
    
    def sample_log_snr(self, shape, device, gen=None):
        pass

    def get_loss(self, x: torch.Tensor, model, gen=None, **model_call_kwargs):
        raise NotImplementedError()

    def diffuse(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    @staticmethod
    def sample_gaussian(mean, std, shape, device, gen=None):
        return torch.randn(shape, device=device, generator=gen) * std + mean
    
    @staticmethod
    def sample_uniform(low, high, shape, device, gen=None):
        return torch.rand(shape, device=device, generator=gen) * (high - low) + low
    
    @staticmethod
    def sample_cosine(s, shift, eps, shape, device, gen=None):
        t = torch.rand(shape, device=device, generator=gen)
        alpha_sq = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_sq = torch.clip(alpha_sq, eps, 1 - eps)
        
        
class RectifiedFlowScheduler(BaseScheduler):
    pred_type = 'rect_flow'

    def __init__(
            self,
            n_steps=1000,
    ):
        self.n_steps = n_steps
        self.sigmas = torch.linspace(1, n_steps, steps=n_steps) / n_steps
    
    def get_loss(self, x, model, gen=None, **model_call_kwargs):
        # velocity prediction
        eps = torch.randn_like(x, generator=gen)
        t = torch.randint(
            0, self.n_steps, size=(x.size(0),), 
            device=x.device, 
            generator=gen
        )

        x_t = self.diffuse(x, t, eps)
        v_pred = model.pred_conditional(x_t, t, **model_call_kwargs)

        loss = F.mse_loss(v_pred, x - eps)
        return loss
        
    def diffuse(self, x, t, eps):
        shape = (-1,) + (1,) * (len(x.shape) - 1)
        sigma_ = self.sigmas.to(dtype=x.dtype, device=x.device)[t].view(*shape)
        x_t = (1 - sigma_) * x + sigma_ * eps
        return x_t
        
    
class JITScheduler(BaseScheduler):
    pred_type = 'data'

    # x_0: noise
    # x_1: data

    def __init__(
            self,
            noise_scale=1.0,
            t_eps=0.05,
            logit_t_mean=-0.8,
            logit_t_std=1.0,
    ):
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.logit_t_mean = logit_t_mean
        self.logit_t_std = logit_t_std
    
    def x_to_v(self, x, x_t, t):
        t = t.view(-1, *((1,) * (x.ndim - 1)))
        v = (x - x_t) / (1 - t).clamp_min(self.t_eps)
        return v

    def get_loss(self, x, model, gen=None, **model_call_kwargs):
        # x prediction
        eps = torch.randn_like(x, generator=gen) * self.noise_scale

        t_logit = torch.randn(
            x.size(0),
            device=x.device, dtype=torch.float32,
            generator=gen
        ) * self.logit_t_std + self.logit_t_mean
        t = torch.sigmoid(t_logit)

        x_t = self.diffuse(x, t, eps)
        v = self.x_to_v(x, x_t, t)

        # model_call_kwargs carries cond= and uncond_mask= from the unified trainer
        x_pred = model.pred_conditional(x_t, t, **model_call_kwargs)
        v_pred = self.x_to_v(x_pred, x_t, t)

        loss = F.mse_loss(v_pred, v)
        return loss
        
    def diffuse(self, x, t, eps):
        shape = (-1,) + (1,) * (x.ndim - 1)
        t = t.view(*shape)
        x_t = t * x + (1 - t) * eps
        return x_t




if __name__ == '__main__':
    n_steps = 1000
    scheduler = BetaScheduler(schedule='linear', n_steps=n_steps)
    print(scheduler.alpha_sq.shape, scheduler.alpha_sq[0], scheduler.alpha_sq[-1])

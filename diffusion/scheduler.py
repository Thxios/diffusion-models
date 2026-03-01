
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
    else:
        raise ValueError(f'unknown scheduler type {name}')


class BaseScheduler:
    pred_type = 'noise'

    def get_loss(self, x: torch.Tensor, model, gen=None, **model_call_kwargs):
        raise NotImplementedError()

    def diffuse(self, x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()
    

@dataclass
class VPSchedule:
    alpha_sq: torch.Tensor
    sigma_sq: torch.Tensor
    log_snr: torch.Tensor

    @staticmethod
    def from_alpha_sq(alpha_sq):
        sigma_sq = 1 - alpha_sq
        return VPSchedule(
            alpha_sq=alpha_sq,
            sigma_sq=sigma_sq,
            log_snr=torch.log(alpha_sq / sigma_sq)
        )

    @staticmethod
    def from_log_snr(log_snr):
        alpha_sq = torch.sigmoid(log_snr)
        return VPSchedule(
            alpha_sq=alpha_sq,
            sigma_sq=1 - alpha_sq,
            log_snr=log_snr
        )
    

class VariancePreservingScheduler(BaseScheduler):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def get_schedule(self, t: torch.LongTensor) -> VPSchedule:
        raise NotImplementedError()
        
    def get_loss(self, x: torch.Tensor, model, gen=None, **model_call_kwargs):
        # noise prediction
        eps = torch.randn_like(x, generator=gen)
        t = torch.randint(
            0, self.n_steps, size=(x.size(0),), 
            device=x.device,
            generator=gen
        )

        x_t = self.diffuse(x, t, eps)
        eps_pred = model(x_t, t, **model_call_kwargs)

        loss = F.mse_loss(eps_pred, eps)
        return loss

    def diffuse(self, x, t, eps):
        schedule = self.get_schedule(t)

        shape = (-1,) + (1,) * (len(x.shape) - 1)
        alpha_ = torch.sqrt(schedule.alpha_sq).to(dtype=x.dtype, device=x.device).view(*shape)
        sigma_ = torch.sqrt(schedule.sigma_sq).to(dtype=x.dtype, device=x.device).view(*shape)

        x_t = alpha_ * x + sigma_ * eps
        return x_t


class BetaScheduler(VariancePreservingScheduler):
    def __init__(
            self,
            schedule='linear',
            beta_start=1e-4,
            beta_end=2e-2,
            cos_s=0.008,
            cos_max_beta=0.999,
            n_steps=1000,
    ):
        if schedule == 'linear':
            beta = torch.linspace(beta_start, beta_end, steps=n_steps, dtype=torch.float64)
        elif schedule == 'cosine':
            beta = self.alpha_cosine(cos_s, cos_max_beta, n_steps)
        else:
            raise ValueError(f'unknown schedule type {schedule}')

        self.n_steps = n_steps
        self.beta = beta
        self.alpha_sq = torch.cumprod(1 - beta, dim=0)

    def get_schedule(self, t: torch.LongTensor) -> VPSchedule:
        alpha_sq = self.alpha_sq.to(t.device)[t]
        return VPSchedule.from_alpha_sq(alpha_sq)

    @staticmethod
    def alpha_cosine(s=0.008, max_beta=0.999, n_steps=1000):
        def alpha_bar_fn(t):
            return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

        beta = []
        for i in range(n_steps):
            t1, t2 = i / n_steps, (i + 1) / n_steps
            beta.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        beta = torch.tensor(beta, dtype=torch.float64)
        return beta


class RectifiedFlowScheduler(BaseScheduler):
    pred_type = 'velocity'

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
        v_pred = model(x_t, t, **model_call_kwargs)

        loss = F.mse_loss(v_pred, x - eps)
        return loss
        
    def diffuse(self, x, t, eps):
        shape = (-1,) + (1,) * (len(x.shape) - 1)
        sigma_ = self.sigmas.to(dtype=x.dtype, device=x.device)[t].view(*shape)
        x_t = (1 - sigma_) * x + sigma_ * eps
        return x_t



if __name__ == '__main__':
    n_steps = 1000
    scheduler = BetaScheduler(schedule='linear', n_steps=n_steps)
    print(scheduler.alpha_sq.shape, scheduler.alpha_sq[0], scheduler.alpha_sq[-1])


import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class Schedule:
    alpha_sq: torch.Tensor
    sigma_sq: torch.Tensor
    log_snr: torch.Tensor

    @staticmethod
    def from_alpha_sq(alpha_sq):
        sigma_sq = 1 - alpha_sq
        return Schedule(
            alpha_sq=alpha_sq,
            sigma_sq=sigma_sq,
            log_snr=torch.log(alpha_sq / sigma_sq)
        )
    @staticmethod
    def from_log_snr(log_snr):
        alpha_sq = torch.sigmoid(log_snr)
        return Schedule(
            alpha_sq=alpha_sq,
            sigma_sq=1 - alpha_sq,
            log_snr=log_snr
        )


class ContinuousScheduler(nn.Module):
    def get_schedule(self, t: torch.Tensor) -> Schedule:
        raise NotImplementedError()

    def diffuse(
            self,
            x: torch.Tensor,
            eps: torch.Tensor,
            t: Optional[torch.Tensor] = None,
            schedule: Optional[Schedule] = None,
    ):
        if schedule is not None:
            pass
        elif t is not None:
            schedule = self.get_schedule(t)
        else:
            raise ValueError(f'at least one of t and schedule must be provided')

        shape = (-1,) + (1,) * (len(x.shape) - 1)
        alpha_ = torch.sqrt(schedule.alpha_sq).to(x.dtype).view(*shape)
        sigma_ = torch.sqrt(schedule.sigma_sq).to(x.dtype).view(*shape)

        x_t = alpha_ * x + sigma_ * eps
        return x_t


class DiscreteScheduler(nn.Module):
    n_steps: int

    def get_schedule(self, t: torch.LongTensor) -> Schedule:
        raise NotImplementedError()

    def diffuse(
            self,
            x: torch.Tensor,
            eps: torch.Tensor,
            t: Optional[torch.LongTensor] = None,
    ):
        schedule = self.get_schedule(t)

        shape = (-1,) + (1,) * (len(x.shape) - 1)
        alpha_ = torch.sqrt(schedule.alpha_sq).to(x.dtype).view(*shape)
        sigma_ = torch.sqrt(schedule.sigma_sq).to(x.dtype).view(*shape)

        x_t = alpha_ * x + sigma_ * eps
        return x_t


class CosineAlphaScheduler(ContinuousScheduler):
    def __init__(
            self,
            snr_start=12,
            snr_end=-8,
    ):
        super().__init__()

        alpha_sq_min = torch.sigmoid(
            torch.tensor(snr_end, dtype=torch.float64))
        alpha_sq_max = torch.sigmoid(
            torch.tensor(snr_start, dtype=torch.float64))

        self.register_buffer('alpha_sq_min', alpha_sq_min)
        self.register_buffer('alpha_sq_max', alpha_sq_max)

    def get_schedule(self, t):
        alpha_sq_raw = (torch.cos(t * torch.pi) + 1) / 2
        alpha_sq = self.alpha_sq_min \
            + (self.alpha_sq_max - self.alpha_sq_min) * alpha_sq_raw
        return Schedule.from_alpha_sq(alpha_sq)



class LogLinearSNRScheduler(ContinuousScheduler):
    def __init__(
            self,
            snr_start=12,
            snr_end=-8,
    ):
        super().__init__()

        snr_min = torch.tensor(snr_end, dtype=torch.float64)
        snr_max = torch.tensor(snr_start, dtype=torch.float64)

        self.register_buffer('snr_min', snr_min)
        self.register_buffer('snr_max', snr_max)

    def get_schedule(self, t):
        log_snr = self.snr_max - (self.snr_max - self.snr_min) * t
        return Schedule.from_log_snr(log_snr)



class CosineSquareAlpha(ContinuousScheduler):
    def __init__(
            self,
            snr_start=12,
            snr_end=-8,
            exponent=2,
    ):
        super().__init__()

        alpha_sq_min = torch.sigmoid(
            torch.tensor(snr_end, dtype=torch.float64))
        alpha_sq_max = torch.sigmoid(
            torch.tensor(snr_start, dtype=torch.float64))

        cos_start = torch.acos(2 * alpha_sq_max - 1) / torch.pi
        cos_end = torch.acos(2 * alpha_sq_min - 1) / torch.pi

        exponent = torch.tensor(exponent, dtype=torch.float64)
        lin_start = torch.pow(cos_start, 1 / exponent)
        lin_end = torch.pow(cos_end, 1 / exponent)

        self.register_buffer('exponent', exponent)
        self.register_buffer('lin_start', lin_start)
        self.register_buffer('lin_end', lin_end)

    def get_schedule(self, t):
        t_rescaled = self.lin_start + (self.lin_end - self.lin_start) * t
        t_rescaled = torch.pow(t_rescaled, self.exponent)
        alpha_sq = (torch.cos(torch.pi * t_rescaled) + 1) / 2
        return Schedule.from_alpha_sq(alpha_sq)


class BetaScheduler(DiscreteScheduler):
    def __init__(
            self,
            schedule='linear',
            beta_start=1e-4,
            beta_end=2e-2,
            cos_s=0.008,
            cos_max_beta=0.999,
            n_steps=1000,
    ):
        super().__init__()

        if schedule == 'linear':
            beta = torch.linspace(beta_start, beta_end,
                                  steps=n_steps,
                                  dtype=torch.float64)
        elif schedule == 'cosine':
            beta = self.alpha_cosine(cos_s, cos_max_beta, n_steps)
        else:
            raise ValueError(f'unknown schedule type {schedule}')

        self.n_steps = beta.size(0)
        alpha_sq = torch.cumprod(1 - beta, dim=0)

        # self.register_buffer('n_steps', n_steps)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha_sq', alpha_sq)

    def get_schedule(self, t: torch.LongTensor) -> Schedule:
        return Schedule.from_alpha_sq(self.alpha_sq[t])

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


class CosCosAlphaScheduler(ContinuousScheduler):
    def __init__(
            self,
            clip_st=0.05,
            clip_ed=0.95,
    ):
        super().__init__()

        clip_st = torch.tensor(clip_st, dtype=torch.float64)
        clip_ed = torch.tensor(clip_ed, dtype=torch.float64)

        self.register_buffer('clip_st', clip_st)
        self.register_buffer('clip_ed', clip_ed)

    def get_schedule(self, t):
        t1 = self.clip_st + (self.clip_ed - self.clip_st) * t
        t2 = (1 - torch.cos(torch.pi * t1)) / 2
        alpha_sq = (torch.cos(torch.pi * t2) + 1) / 2
        return Schedule.from_alpha_sq(alpha_sq)


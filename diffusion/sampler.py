

import torch
import tqdm.auto as tqdm
from typing import Callable

from diffusion.scheduler import ContinuousScheduler, DiscreteScheduler, Schedule


class BaseSampler:
    def __init__(self, n_steps, pbar=False):
        self.n_steps = n_steps
        self.pbar = pbar

    def prepare_iterator(self, iterator):
        if self.pbar:
            iterator = tqdm.tqdm(iterator, leave=False)
        return iterator

    def sample(
            self,
            z: torch.Tensor,
            scheduler,
            pred_fn: Callable[..., torch.Tensor],
    ):
        raise NotImplementedError()



class ContinuousSampler(BaseSampler):

    def sample(
            self,
            z: torch.Tensor,
            scheduler: ContinuousScheduler,
            pred_fn: Callable[..., torch.Tensor],
    ):
        raise NotImplementedError()

    @classmethod
    @torch.no_grad()
    def pred_eps(cls, pred_fn, z: torch.Tensor, t: Schedule):
        log_snr_t_ = t.log_snr.to(z.dtype).repeat(z.size(0))
        eps_pred = pred_fn(z, log_snr_t_)
        return eps_pred



class DiscreteSampler(BaseSampler):

    def sample(
            self,
            z: torch.Tensor,
            scheduler: DiscreteScheduler,
            pred_fn: Callable[..., torch.Tensor],
    ):
        raise NotImplementedError()

    @classmethod
    @torch.no_grad()
    def pred_eps(cls, pred_fn, z: torch.Tensor, t: torch.LongTensor):
        t_ = t.repeat(z.size(0))
        eps_pred = pred_fn(z, t_)
        return eps_pred


class ContinuousNaiveSampler(ContinuousSampler):
    def __init__(self, n_steps, v=0.3, pbar=False):
        super().__init__(n_steps, pbar)
        self.v = v

    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: ContinuousScheduler,
            pred_fn: Callable[..., torch.Tensor]
    ):
        t_steps = torch.linspace(
            1, 0,
            steps=self.n_steps,
            dtype=torch.float64,
            device=z.device
        )

        iterator = self.prepare_iterator(range(self.n_steps))
        for i in iterator:
            t = scheduler.get_schedule(t_steps[i])
            eps_pred = self.pred_eps(pred_fn, z, t)

            alpha_t_ = torch.sqrt(t.alpha_sq).to(z.dtype)
            sigma_t_ = torch.sqrt(t.sigma_sq).to(z.dtype)
            x0_pred = (z - sigma_t_ * eps_pred) / alpha_t_
            x0_pred = torch.clip(x0_pred, -1, 1)

            if i < self.n_steps - 1:
                t_prime = scheduler.get_schedule(t_steps[i + 1])
                exp_snr_diff = torch.exp(t.log_snr - t_prime.log_snr)

                # next sample mu
                mult_z_ = (exp_snr_diff * torch.sqrt(t_prime.alpha_sq / t.alpha_sq)) \
                    .to(z.dtype)
                mult_x_ = ((1 - exp_snr_diff) * torch.sqrt(t_prime.alpha_sq)) \
                    .to(z.dtype)
                mu = mult_z_.to(z.dtype) * z + mult_x_.to(z.dtype) * x0_pred

                # next sample sigma
                sigma_blended_sq = t_prime.sigma_sq ** (1 - self.v) * t.sigma_sq ** self.v
                sigma_ = torch.sqrt((1 - exp_snr_diff) * sigma_blended_sq) \
                    .to(z.dtype)

                eps = torch.randn_like(z)
                z = mu + sigma_ * eps

            else:
                z = x0_pred

        return z



class ContinuousDDIMSampler(ContinuousSampler):
    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: ContinuousScheduler,
            pred_fn: Callable[..., torch.Tensor]
    ):
        t_steps = torch.linspace(
            1, 0,
            steps=self.n_steps,
            dtype=torch.float64,
            device=z.device
        )

        iterator = self.prepare_iterator(range(self.n_steps))
        for i in iterator:
            t = scheduler.get_schedule(t_steps[i])
            eps_pred = self.pred_eps(pred_fn, z, t)

            alpha_t_ = torch.sqrt(t.alpha_sq).to(z.dtype)
            sigma_t_ = torch.sqrt(t.sigma_sq).to(z.dtype)
            x0_pred = (z - sigma_t_ * eps_pred) / alpha_t_
            x0_pred = torch.clip(x0_pred, -1, 1)

            if i < self.n_steps - 1:
                t_prime = scheduler.get_schedule(t_steps[i + 1])

                alpha_t_prime_ = torch.sqrt(t_prime.alpha_sq).to(z.dtype)
                sigma_t_prime_ = torch.sqrt(t_prime.sigma_sq).to(z.dtype)
                z = alpha_t_prime_ * x0_pred + sigma_t_prime_ * eps_pred

            else:
                z = x0_pred

        return z


class ContinuousDPM2Solver(ContinuousSampler):
    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: ContinuousScheduler,
            pred_fn: Callable[..., torch.Tensor]
    ):
        # batch_size = z.size(0)
        t_steps = torch.linspace(
            1, 0,
            steps=self.n_steps + 1,
            dtype=torch.float64,
            device=z.device
        )

        iterator = self.prepare_iterator(range(1, self.n_steps + 1))
        for i in iterator:
            t_prev = scheduler.get_schedule(t_steps[i - 1])
            t = scheduler.get_schedule(t_steps[i])
            s = Schedule.from_log_snr((t_prev.log_snr + t.log_snr) / 2)
            half_snr_diff = (t.log_snr - t_prev.log_snr) / 2

            # first order approx
            # log_snr_t_prev_ = t_prev.log_snr.to(z.dtype).repeat(batch_size)
            # eps_u = pred_fn(z, log_snr_t_prev_)
            eps_u = self.pred_eps(pred_fn, z, t_prev)

            x_mult_ = torch.sqrt(s.alpha_sq / t_prev.alpha_sq) \
                .to(z.dtype)
            eps_mult_ = (torch.sqrt(s.sigma_sq) * torch.expm1(half_snr_diff / 2)) \
                .to(z.dtype)
            u = x_mult_ * z - eps_mult_ * eps_u

            # second order approx
            # log_snr_s_ = s.log_snr.to(z.dtype).repeat(batch_size)
            # eps_z = pred_fn(u, log_snr_s_)
            eps_z = self.pred_eps(pred_fn, u, s)

            x_mult2_ = torch.sqrt(t.alpha_sq / t_prev.alpha_sq) \
                .to(z.dtype)
            eps_mult2_ = (torch.sqrt(t.sigma_sq) * torch.expm1(half_snr_diff)) \
                .to(z.dtype)
            z = x_mult2_ * z - eps_mult2_ * eps_z

        return z



class DiscreteDDIMSampler(DiscreteSampler):
    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: DiscreteScheduler,
            pred_fn: Callable[..., torch.Tensor]
    ):

        t_steps = torch.linspace(scheduler.n_steps - 1, 0,
                                 steps=self.n_steps + 1,
                                 dtype=torch.int64,
                                 device=z.device)[:-1]

        iterator = self.prepare_iterator(range(self.n_steps))
        for i in iterator:
            t = scheduler.get_schedule(t_steps[i])
            eps_pred = self.pred_eps(pred_fn, z, t_steps[i])

            alpha_t_ = torch.sqrt(t.alpha_sq).to(z.dtype)
            sigma_t_ = torch.sqrt(t.sigma_sq).to(z.dtype)
            x0_pred = (z - sigma_t_ * eps_pred) / alpha_t_
            x0_pred = torch.clip(x0_pred, -1, 1)

            if i < self.n_steps - 1:
                t_prime = scheduler.get_schedule(t_steps[i + 1])

                alpha_t_prime_ = torch.sqrt(t_prime.alpha_sq).to(z.dtype)
                sigma_t_prime_ = torch.sqrt(t_prime.sigma_sq).to(z.dtype)
                z = alpha_t_prime_ * x0_pred + sigma_t_prime_ * eps_pred

            else:
                z = x0_pred

        return z

"""
class ContinuousDPM2Solver(ContinuousSampler):
    @torch.no_grad()
    def sample(
            self,
            z: torch.Tensor,
            scheduler: ContinuousScheduler,
            pred_fn: Callable[..., torch.Tensor]
    ):
        # batch_size = z.size(0)
        t_steps = torch.linspace(
            1, 0,
            steps=self.n_steps + 1,
            dtype=torch.float64,
            device=z.device
        )

        iterator = self.prepare_iterator(range(1, self.n_steps + 1))
        for i in iterator:
            t_prev = scheduler.get_schedule(t_steps[i - 1])
            t = scheduler.get_schedule(t_steps[i])
            s = Schedule.from_log_snr((t_prev.log_snr + t.log_snr) / 2)
            half_snr_diff = (t.log_snr - t_prev.log_snr) / 2

            # first order approx
            # log_snr_t_prev_ = t_prev.log_snr.to(z.dtype).repeat(batch_size)
            # eps_u = pred_fn(z, log_snr_t_prev_)
            eps_u = self.pred_eps(pred_fn, z, t_prev)

            x_mult_ = torch.sqrt(s.alpha_sq / t_prev.alpha_sq) \
                .to(z.dtype)
            eps_mult_ = (torch.sqrt(s.sigma_sq) * torch.expm1(half_snr_diff / 2)) \
                .to(z.dtype)
            u = x_mult_ * z - eps_mult_ * eps_u

            # second order approx
            # log_snr_s_ = s.log_snr.to(z.dtype).repeat(batch_size)
            # eps_z = pred_fn(u, log_snr_s_)
            eps_z = self.pred_eps(pred_fn, u, s)

            x_mult2_ = torch.sqrt(t.alpha_sq / t_prev.alpha_sq) \
                .to(z.dtype)
            eps_mult2_ = (torch.sqrt(t.sigma_sq) * torch.expm1(half_snr_diff)) \
                .to(z.dtype)
            z = x_mult2_ * z - eps_mult2_ * eps_z

        return z
"""




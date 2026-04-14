"""Unified training entry point — single- and multi-GPU via HF Accelerate.

Usage
─────
# Single-GPU
python train.py --train_arg_json arg_json/unet/beta_mnist2.json

# Multi-GPU
accelerate launch --num_processes 4 train.py --train_arg_json arg_json/jit/base.json

# Resume
python train.py --resume_ckpt_dir outputs/<run>
"""

import os
import shutil
import random as rd
import warnings
import json
import dataclasses
from typing import Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import tqdm.auto as tqdm
import wandb
import fire
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from transformers import get_scheduler as get_lr_scheduler
from pytorch_fid.inception import InceptionV3

from diffusion.scheduler import get_scheduler
from diffusion.sampler import get_sampler
from modeling import get_model
from utils import count_parameters
from utils.fid import (
    load_hidden_parameters,
    calculate_frechet_distance,
    inception_features_to_hidden_parameters,
)
from utils.fid_infinity import fid_extrapolation
from utils.validate_memo import calc_memorization_metric
from utils.data import (
    load_training_dataset,
    make_generation_seed,
    maybe_build_memorization_tensor,
    FIDNoiseDataset,
    image_shape as dataset_image_shape,
    num_classes as dataset_num_classes,
)


WANDB_PROJECT_NAME = 'noh-diffusion'


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainArgs:
    # --- run ---
    output_dir: str
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None

    # --- schedule ---
    max_steps: int = 100_000
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: Optional[int] = None
    eval_n_examples: int = 40

    # --- FID ---
    fid_eval_steps: Optional[int] = None
    fid_ema: bool = True
    fid_reference_dataset: str = 'mnist-train'
    fid_n_examples: int = 10_000
    fid_sample_labels: bool = False
    generation_batch_size: int = 256    # per-GPU
    inception_batch_size: int = 512
    adjust_fid_n: bool = True
    fid_adjust_subsets: List[int] = dataclasses.field(
        default_factory=lambda: [4000, 6000, 8000, 10000])

    # --- optim ---
    batch_size: int = 128               # per-GPU when num_processes > 1
    lr: float = 2e-4
    lr_scheduler: Optional[str] = None
    lr_warmup_steps: int = 0
    lr_scheduler_cfg: dict = dataclasses.field(default_factory=dict)
    optimizer: str = 'adamw'
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    clip_grad_norm: float = 1.0

    # --- EMA ---
    use_ema: bool = True
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75

    # --- data ---
    dataset: str = 'mnist'
    dataset_dir: str = 'datasets'
    augmentations: List[str] = dataclasses.field(default_factory=list)
    image_size: Optional[int] = None
    dataloader_num_workers: int = 2
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True

    # --- misc ---
    seed: int = 42
    bf16: bool = True
    compile: bool = False

    # --- diffusion ---
    p_uncond: float = 0.2
    class_conditioning: bool = False
    guidance_scale: float = 1.0
    cfg_interval: Optional[Tuple[float, float]] = None
    model_type: str = 'unet'
    model_cfg: dict = dataclasses.field(default_factory=dict)
    scheduler_type: str = 'beta'
    scheduler_cfg: dict = dataclasses.field(default_factory=dict)
    sampler_type: str = 'ddim'
    sampler_cfg: dict = dataclasses.field(default_factory=dict)


def _load_train_args(path: str) -> TrainArgs:
    with open(path) as f:
        d = json.load(f)
    # Back-compat shims for legacy JSON files
    if 'lr_schduler_cfg' in d:
        d['lr_scheduler_cfg'] = d.pop('lr_schduler_cfg')
    d.pop('device', None)
    # Drop fields that exist in old train_args.json but not in the new dataclass
    known = {f.name for f in dataclasses.fields(TrainArgs)}
    d = {k: v for k, v in d.items() if k in known}
    return TrainArgs(**d)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        rd.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    rd.seed(worker_seed)


def fix_state_dict(state_dict):
    """Strip _orig_mod. prefix added by torch.compile."""
    return {
        (k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
        for k, v in state_dict.items()
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
            self,
            accelerator: Accelerator,
            arg: TrainArgs,
            model,
            scheduler,
            sampler,
            resume_ckpt_dir: Optional[str] = None,
            overwrite: bool = False,
    ):
        self.accelerator = accelerator
        self.arg = arg
        self.scheduler = scheduler
        self.sampler = sampler

        if arg.dataloader_num_workers == 0:
            warnings.warn('set dataloader_num_workers > 0 for reproducibility')

        accelerator.print(f'TrainArgs:\n{arg}')
        accelerator.print(
            f'num_processes: {accelerator.num_processes}  '
            f'mixed_precision: {accelerator.mixed_precision}'
        )

        # ── Output directory (main process) ──────────────────────────────
        # Normalise trailing slashes so "/foo/bar/" == "/foo/bar"
        self.resume_ckpt_dir = resume_ckpt_dir
        if accelerator.is_main_process:
            if resume_ckpt_dir is not None:
                assert os.path.normpath(resume_ckpt_dir) == os.path.normpath(arg.output_dir), (
                    f'resume_ckpt_dir must equal output_dir; '
                    f'got "{resume_ckpt_dir}" vs "{arg.output_dir}"'
                )
            else:
                if os.path.exists(arg.output_dir) and os.listdir(arg.output_dir):
                    if not overwrite:
                        ans = input(f'\n"{arg.output_dir}" is not empty, overwrite? (Y/n): ')
                        if ans.lower().strip() != 'y':
                            raise ValueError(f'"{arg.output_dir}" exists and is not empty')
                    print(f'Overwriting "{arg.output_dir}"...\n')
                    shutil.rmtree(arg.output_dir)
            os.makedirs(arg.output_dir, exist_ok=True)
            with open(os.path.join(arg.output_dir, 'train_args.json'), 'w') as f:
                json.dump(dataclasses.asdict(arg), f, indent=2)
        accelerator.wait_for_everyone()

        # ── Wandb (main process) ─────────────────────────────────────────
        self.wandb_run = None
        if accelerator.is_main_process and (arg.wandb_run_name or arg.wandb_run_id):
            self._prepare_wandb_run(arg.wandb_run_name)

        # ── Compile (before DDP wrapping) ────────────────────────────────
        if arg.compile:
            accelerator.print('Compiling model...')
            model = torch.compile(model)
            accelerator.print('Model compiled.')

        # ── EMA (track raw params before DDP) ───────────────────────────
        self.ema_model = None
        if arg.use_ema:
            self.ema_model = EMAModel(
                model.parameters(),
                use_ema_warmup=True,
                inv_gamma=arg.ema_inv_gamma,
                power=arg.ema_power,
            )

        # ── Datasets ────────────────────────────────────────────────────
        self.train_dataset = load_training_dataset(
            arg.dataset, arg.dataset_dir,
            train=True,
            augmentations=arg.augmentations,
            image_size=arg.image_size,
        )
        self.valid_dataset = load_training_dataset(
            arg.dataset, arg.dataset_dir, train=False,
        )
        accelerator.print(f'Train dataset: {len(self.train_dataset)} samples')

        # ── DataLoaders ─────────────────────────────────────────────────
        rng = torch.Generator()
        rng.manual_seed(arg.seed)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=arg.batch_size,
            shuffle=True,
            drop_last=arg.dataloader_drop_last,
            pin_memory=arg.dataloader_pin_memory,
            num_workers=arg.dataloader_num_workers,
            worker_init_fn=seed_worker,
            generator=rng,
        )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=arg.generation_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=arg.dataloader_pin_memory,
            num_workers=arg.dataloader_num_workers,
        )

        # ── Optimizer & LR scheduler ────────────────────────────────────
        optimizer = self._make_optimizer(model)
        lr_sched_obj = self._make_lr_scheduler(optimizer)

        # ── Accelerate prepare ──────────────────────────────────────────
        prepare_args = [model, optimizer, train_loader, valid_loader]
        if lr_sched_obj is not None:
            prepare_args.append(lr_sched_obj)

        prepared = accelerator.prepare(*prepare_args)
        self.model            = prepared[0]
        self.optimizer        = prepared[1]
        self.train_dataloader = prepared[2]
        self.valid_dataloader = prepared[3]
        self.lr_scheduler     = prepared[4] if lr_sched_obj is not None else None

        self.raw_model = accelerator.unwrap_model(self.model)
        if self.ema_model is not None:
            self.ema_model.to(accelerator.device)

        self.steps_per_epoch = len(self.train_dataloader)
        accelerator.print(f'Steps per epoch: {self.steps_per_epoch}')

        # ── Counters ────────────────────────────────────────────────────
        self.global_steps   = 0
        self.epochs         = 0
        self.steps_in_epoch = 0

        # ── Eval seed ───────────────────────────────────────────────────
        self.eval_seed = make_generation_seed(
            arg.dataset, arg.eval_n_examples, seed=arg.seed, sample_labels=False,
        )

        # ── FID setup ───────────────────────────────────────────────────
        self.fid_dataset  = None
        self.fid_reference = None
        if arg.fid_eval_steps is not None:
            if arg.adjust_fid_n:
                assert arg.fid_n_examples == max(arg.fid_adjust_subsets), (
                    'fid_n_examples must equal max(fid_adjust_subsets) for FID extrapolation')
                arg.fid_adjust_subsets.sort()

            img_shape = dataset_image_shape(arg.dataset)
            n_cls     = dataset_num_classes(arg.dataset)
            self.fid_dataset = FIDNoiseDataset(
                n_samples=arg.fid_n_examples,
                image_shape=img_shape,
                class_condition=arg.class_conditioning,
                n_classes=n_cls,
                sample_labels=arg.fid_sample_labels,
                seed=arg.seed,
            )
            self.fid_reference = load_hidden_parameters(arg.fid_reference_dataset, save=False)

        # ── Memorization tensor (small datasets only) ────────────────────
        self.memo_tensor = maybe_build_memorization_tensor(arg.dataset, self.train_dataset)

        # ── Checkpoint dir ──────────────────────────────────────────────
        self.ckpt_base_dir = (
            os.path.join(arg.output_dir, 'ckpts') if arg.save_steps else None
        )

    # ── Setup helpers ────────────────────────────────────────────────────

    def _make_optimizer(self, model):
        cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[self.arg.optimizer]
        return cls(model.parameters(), lr=self.arg.lr, betas=self.arg.adam_betas)

    def _make_lr_scheduler(self, optimizer):
        if self.arg.lr_scheduler is None:
            return None
        return get_lr_scheduler(
            self.arg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.arg.lr_warmup_steps,
            num_training_steps=self.arg.max_steps,
            **self.arg.lr_scheduler_cfg,
        )

    def _prepare_wandb_run(self, run_name):
        if self.arg.wandb_run_id is not None:
            self.wandb_run = wandb.init(
                project=WANDB_PROJECT_NAME, id=self.arg.wandb_run_id, resume='must')
        else:
            self.wandb_run = wandb.init(
                name=run_name, project=WANDB_PROJECT_NAME,
                config=dataclasses.asdict(self.arg),
                dir=self.arg.output_dir,
            )
            self.arg.wandb_run_id = self.wandb_run.id
        self.accelerator.print(
            f'Wandb run: {self.wandb_run.name}  id: {self.wandb_run.id}')

    # ── Checkpoint ───────────────────────────────────────────────────────

    def save_ckpt(self, ckpt_dir: str):
        """Save EMA-merged model weights as model.pt (main process only).
        If no EMA, saves raw weights instead."""
        if not self.accelerator.is_main_process:
            return
        os.makedirs(ckpt_dir, exist_ok=True)

        if self.ema_model is not None:
            self.ema_model.store(self.raw_model.parameters())
            self.ema_model.copy_to(self.raw_model.parameters())

        sd = self.raw_model.state_dict()
        if self.arg.compile:
            sd = fix_state_dict(sd)
        torch.save(sd, os.path.join(ckpt_dir, 'model.pt'))

        if self.ema_model is not None:
            self.ema_model.restore(self.raw_model.parameters())

        self.accelerator.print(f'Saved ckpt → {ckpt_dir}')

    def save_latest_ckpt(self):
        """Save full training state for resumption (all processes)."""
        latest_dir = os.path.join(self.arg.output_dir, 'latest_state')
        os.makedirs(latest_dir, exist_ok=True)
        self.accelerator.save_state(latest_dir)

        if self.accelerator.is_main_process:
            extra = {
                'global_steps':   self.global_steps,
                'epochs':         self.epochs,
                'steps_in_epoch': self.steps_in_epoch,
            }
            if self.ema_model is not None:
                extra['ema_state_dict'] = self.ema_model.state_dict()
            torch.save(extra, os.path.join(latest_dir, 'extra.pt'))

        self.accelerator.wait_for_everyone()

    def load_latest_ckpt(self, ckpt_dir: str):
        """Load full training state (all processes)."""
        latest_dir = os.path.join(ckpt_dir, 'latest_state')
        self.accelerator.load_state(latest_dir)

        extra = torch.load(
            os.path.join(latest_dir, 'extra.pt'),
            map_location='cpu', weights_only=False,
        )
        self.global_steps   = extra['global_steps']
        self.epochs         = extra['epochs']
        self.steps_in_epoch = extra['steps_in_epoch']
        if self.ema_model is not None and 'ema_state_dict' in extra:
            self.ema_model.load_state_dict(extra['ema_state_dict'])
            self.ema_model.to(self.accelerator.device)  # state_dict loads to CPU

        self.accelerator.print(
            f'Resumed from step {self.global_steps} (epoch {self.epochs})')
        self.accelerator.wait_for_everyone()

    # ── Logging ──────────────────────────────────────────────────────────

    def log(self, steps: int, **logs):
        if not self.accelerator.is_main_process:
            return
        with open(os.path.join(self.arg.output_dir, 'train_log.jsonl'), 'a') as f:
            f.write(json.dumps({'steps': steps, **logs}) + '\n')
        if self.wandb_run is not None:
            self.wandb_run.log(
                {'global_steps': steps, 'epochs': steps / self.steps_per_epoch, **logs},
                step=steps,
            )

    # ── Training step ────────────────────────────────────────────────────

    def train_on_batch(self, batch) -> float:
        x = batch['image'].to(self.accelerator.device)

        cond = None
        uncond_mask = None
        if self.arg.class_conditioning and 'label' in batch:
            cond = batch['label'].to(self.accelerator.device)
            if self.arg.p_uncond > 0:
                uncond_mask = torch.rand(cond.shape, device=cond.device) < self.arg.p_uncond

        with self.accelerator.autocast():
            loss = self.scheduler.get_loss(
                x, self.raw_model,
                cond=cond,
                uncond_mask=uncond_mask,
            )

        self.accelerator.backward(loss)
        if self.arg.clip_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.arg.clip_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        if self.ema_model is not None:
            self.ema_model.step(self.raw_model.parameters())

        return loss.detach().item()

    # ── Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_eval_examples(self) -> dict:
        """Generate a fixed grid of images. Main process only."""
        if not self.accelerator.is_main_process:
            return {}

        self.raw_model.eval()
        z    = self.eval_seed['z'].to(self.accelerator.device)
        cond = (self.eval_seed['cls'].to(self.accelerator.device)
                if self.arg.class_conditioning and 'cls' in self.eval_seed else None)

        with self.accelerator.autocast():
            samples = self.sampler.sample(z, self.raw_model, cond=cond)
        result = {'examples': torch.clamp(samples, -1, 1).cpu()}

        if self.ema_model is not None:
            self.ema_model.store(self.raw_model.parameters())
            self.ema_model.copy_to(self.raw_model.parameters())
            with self.accelerator.autocast():
                ema_samples = self.sampler.sample(z, self.raw_model, cond=cond)
            result['ema_examples'] = torch.clamp(ema_samples, -1, 1).cpu()
            self.ema_model.restore(self.raw_model.parameters())

        self.raw_model.train()
        return result

    @torch.no_grad()
    def evaluate_validation_loss(self) -> dict:
        """Compute validation loss across all GPUs (all-reduce)."""
        self.raw_model.eval()

        def _loss_over_loader(model):
            loss_sum = torch.zeros(1, device=self.accelerator.device)
            for batch in self.valid_dataloader:
                x = batch['image'].to(self.accelerator.device)
                cond = (batch['label'].to(self.accelerator.device)
                        if self.arg.class_conditioning and 'label' in batch else None)
                with self.accelerator.autocast():
                    loss = self.scheduler.get_loss(x, model, cond=cond)
                loss_sum += loss.detach() * x.size(0)
            total = self.accelerator.reduce(loss_sum, reduction='sum')
            return (total / len(self.valid_dataset)).item()

        result = {'loss': _loss_over_loader(self.raw_model)}

        if self.ema_model is not None:
            self.ema_model.store(self.raw_model.parameters())
            self.ema_model.copy_to(self.raw_model.parameters())
            result['ema_loss'] = _loss_over_loader(self.raw_model)
            self.ema_model.restore(self.raw_model.parameters())

        self.raw_model.train()
        return result

    @torch.no_grad()
    def evaluate_fid_streaming(self) -> Optional[dict]:
        """Streaming multi-GPU FID.

        Each GPU generates its shard of FIDNoiseDataset, immediately runs
        Inception, then features are all-gathered.  Main process computes FID.
        """
        torch.cuda.empty_cache()

        dist_sampler = DistributedSampler(
            self.fid_dataset,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )
        loader = DataLoader(
            self.fid_dataset,
            batch_size=self.arg.generation_batch_size,
            sampler=dist_sampler,
            num_workers=1,
            drop_last=False,
            pin_memory=False,
        )

        inception = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=False
        ).to(self.accelerator.device).eval()

        self.raw_model.eval()
        if self.ema_model is not None and self.arg.fid_ema:
            self.ema_model.store(self.raw_model.parameters())
            self.ema_model.copy_to(self.raw_model.parameters())

        local_features = []
        pbar = tqdm.tqdm(
            loader,
            desc=f'FID [rank {self.accelerator.process_index}]',
            leave=False,
            disable=not self.accelerator.is_main_process,
        )
        for batch in pbar:
            z    = batch['z'].to(self.accelerator.device)
            cond = (batch['cls'].to(self.accelerator.device)
                    if self.arg.class_conditioning and 'cls' in batch else None)
            with self.accelerator.autocast():
                samples = self.sampler.sample(z, self.raw_model, cond=cond)
            samples = torch.clamp(samples, -1, 1)
            feats = inception(samples.float())[0].flatten(1)
            local_features.append(feats.cpu())

        if self.ema_model is not None and self.arg.fid_ema:
            self.ema_model.restore(self.raw_model.parameters())
        self.raw_model.train()

        del inception
        torch.cuda.empty_cache()

        local_tensor  = torch.cat(local_features, dim=0).to(self.accelerator.device)
        all_features  = self.accelerator.gather(local_tensor)

        if not self.accelerator.is_main_process:
            return None

        features = all_features[:self.arg.fid_n_examples].float().cpu().numpy()
        self.accelerator.print(f'Gathered inception features: {features.shape}')

        result = {}
        if self.arg.adjust_fid_n:
            fid_out = fid_extrapolation(
                features,
                ref_mu=self.fid_reference[0],
                ref_sigma=self.fid_reference[1],
                subset_sizes=self.arg.fid_adjust_subsets,
                target_n=50_000,
            )
            result['FID']     = fid_out['fids'][-1]
            result['FID@inf'] = fid_out['fid_infinity']
            result['FID@50k'] = fid_out['fid_target']
        else:
            mu, sigma = inception_features_to_hidden_parameters(features)
            result['FID'] = calculate_frechet_distance(
                mu, sigma, self.fid_reference[0], self.fid_reference[1])

        with open(os.path.join(self.arg.output_dir, 'fid_evaluations.jsonl'), 'a') as f:
            f.write(json.dumps({'steps': self.global_steps, **result}) + '\n')

        self.accelerator.print(f'FID: {result["FID"]:.4f}')
        return result

    def evaluate(self, steps: int):
        """Image grids (main process) + validation loss (all GPUs)."""
        examples = self.generate_eval_examples()
        if self.accelerator.is_main_process and examples:
            save_dir = os.path.join(self.arg.output_dir, 'examples')
            os.makedirs(save_dir, exist_ok=True)
            image_log = {}
            for name, imgs in examples.items():
                grid = TF.to_pil_image(
                    make_grid(imgs, nrow=10, normalize=True, value_range=(-1, 1)))
                grid.save(os.path.join(save_dir, f'{steps:06d}_{name}.png'))
                image_log[name] = wandb.Image(grid)
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {'global_steps': steps,
                     'epochs': steps / self.steps_per_epoch,
                     **image_log},
                    step=steps,
                )

        val_loss = self.evaluate_validation_loss()
        self.log(steps, **{f'val/{k}': v for k, v in val_loss.items()})

    # ── Main training loop ────────────────────────────────────────────────

    def train(self):
        if self.resume_ckpt_dir is not None:
            self.load_latest_ckpt(self.resume_ckpt_dir)

        self.model.train()

        with tqdm.tqdm(
            initial=self.global_steps,
            total=self.arg.max_steps,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.global_steps < self.arg.max_steps:
                for batch in self.train_dataloader:
                    loss = self.train_on_batch(batch)
                    self.global_steps   += 1
                    self.epochs          = self.global_steps // self.steps_per_epoch
                    self.steps_in_epoch  = self.global_steps % self.steps_per_epoch
                    pbar.update(1)

                    if self.global_steps % self.arg.logging_steps == 0:
                        if self.accelerator.is_main_process:
                            pbar.set_postfix({'loss': f'{loss:.5f}'})
                        logs = {'loss': loss}
                        if self.ema_model is not None:
                            logs['ema_decay'] = self.ema_model.cur_decay_value
                        if self.lr_scheduler is not None:
                            logs['lr'] = self.lr_scheduler.get_last_lr()[0]
                        self.log(self.global_steps, **logs)

                    if self.global_steps % self.arg.eval_steps == 0:
                        self.evaluate(self.global_steps)
                        self.model.train()

                    if (self.ckpt_base_dir is not None
                            and self.global_steps % self.arg.save_steps == 0):
                        ckpt_dir = os.path.join(
                            self.ckpt_base_dir, f'ckpt-{self.global_steps:06d}')
                        self.save_ckpt(ckpt_dir)
                        self.save_latest_ckpt()

                    if (self.arg.fid_eval_steps is not None
                            and self.global_steps % self.arg.fid_eval_steps == 0):
                        fid_result = self.evaluate_fid_streaming()
                        if self.accelerator.is_main_process and fid_result:
                            self.log(self.global_steps, **fid_result)
                        self.accelerator.wait_for_everyone()
                        self.model.train()

                    if self.global_steps >= self.arg.max_steps:
                        break

        self.accelerator.print('Training complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
        train_arg_json: Optional[str] = None,
        resume_ckpt_dir: Optional[str] = None,
        overwrite: bool = False,
):
    assert train_arg_json is not None or resume_ckpt_dir is not None, \
        'provide --train_arg_json or --resume_ckpt_dir'

    if resume_ckpt_dir is not None and train_arg_json is None:
        train_arg_json = os.path.join(resume_ckpt_dir, 'train_args.json')

    arg = _load_train_args(train_arg_json)

    accelerator = Accelerator(mixed_precision='bf16' if arg.bf16 else 'no')

    # Per-process seed offset → decorrelated diffusion noise across GPUs
    seed_everything(arg.seed + accelerator.process_index)

    model     = get_model(arg.model_type, **arg.model_cfg)
    scheduler = get_scheduler(arg.scheduler_type, **arg.scheduler_cfg)
    # Top-level guidance_scale / cfg_interval are defaults; sampler_cfg overrides
    sampler_kwargs = {
        'guidance_scale': arg.guidance_scale,
        'cfg_interval': arg.cfg_interval,
        **arg.sampler_cfg,
    }
    sampler = get_sampler(arg.sampler_type, scheduler, **sampler_kwargs)

    accelerator.print(f'Model parameters: {count_parameters(model) / 1e6:.2f}M')

    trainer = Trainer(
        accelerator=accelerator,
        arg=arg,
        model=model,
        scheduler=scheduler,
        sampler=sampler,
        resume_ckpt_dir=resume_ckpt_dir,
        overwrite=overwrite,
    )
    trainer.train()


if __name__ == '__main__':
    fire.Fire(main)

"""
Multi-GPU training of JIT diffusion models via Hugging Face Accelerate.

Key differences from train_jit.py
──────────────────────────────────
* Multi-GPU DDP + bf16 mixed precision via Accelerate.
* batch_size / generation_batch_size are *per-GPU*.
* Validation loss is gathered across all GPUs (all-reduce sum / dataset size).
* FID evaluation uses a streaming generate→inception pipeline:
    - each GPU generates its shard of noise seeds and immediately runs Inception,
    - inception features (not images) are gathered across GPUs,
    - main process computes FID — no full-size image tensor is ever allocated.
* Checkpoint save/load via accelerator.save_state (handles per-process RNG states).
* File I/O, wandb logging, and image saving are main-process-only.

Usage
──────
  accelerate launch --num_processes 4 train_jit_multigpu.py \\
      --train_arg_json path/to/args.json
"""

import os
import shutil
import random as rd
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torchvision.datasets import LSUN
import json
import tqdm.auto as tqdm
import dataclasses
from functools import partial
from typing import Optional, List, Tuple
import wandb
import fire
from diffusers.training_utils import EMAModel
from transformers import get_scheduler as get_lr_scheduler
from datasets import load_dataset as load_hf_dataset
from accelerate import Accelerator
from pytorch_fid.inception import InceptionV3

from diffusion.scheduler import JITScheduler
from diffusion.sampler import JITSampler
from modeling.transformer import JITTransformer
from utils import count_parameters, get_augmentations
from utils.fid import (
    load_hidden_parameters,
    calculate_frechet_distance,
    inception_features_to_hidden_parameters,
)
from utils.fid_infinity import fid_extrapolation


WANDB_PROJECT_NAME = 'noh-diffusion'


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainArgs:
    output_dir: str
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None

    max_steps: int = 100_000
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: Optional[int] = 10000
    eval_n_examples: int = 40

    fid_eval_steps: Optional[int] = 10000
    fid_reference_dataset: str = 'lsun-church_outdoor-train'
    fid_n_examples: int = 10000
    fid_sample_labels: bool = False
    # Per-GPU batch size for both diffusion sampling and Inception in streaming FID.
    generation_batch_size: int = 256
    # Kept for compatibility with train_jit.py args files; unused in streaming FID.
    inception_batch_size: int = 512
    adjust_fid_n: bool = True
    fid_adjust_subsets: List[int] = dataclasses.field(
        default_factory=lambda: [4000, 6000, 8000, 10000])

    batch_size: int = 256          # per GPU
    lr: float = 2e-4
    lr_scheduler: Optional[str] = 'constant_with_warmup'
    lr_warmup_steps: int = 1000
    lr_scheduler_cfg: dict = dataclasses.field(default_factory=dict)
    optimizer: str = 'adamw'
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    clip_grad_norm: Optional[float] = 1.0

    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75

    dataset: str = 'lsun-church_outdoor'
    dataset_dir: str = 'datasets'
    augmentations: List[str] = dataclasses.field(
        default_factory=lambda: ['RandomHorizontalFlip'])
    dataloader_num_workers: int = 4
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True

    seed: int = 42
    compile: bool = True

    class_conditioning: bool = False
    p_uncond: float = 0.1
    model_type: str = 'jit'
    model_cfg: dict = dataclasses.field(default_factory=dict)
    scheduler_type: str = 'jit'
    scheduler_cfg: dict = dataclasses.field(default_factory=dict)
    sampler_type: str = 'jit'
    sampler_cfg: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class FIDNoiseDataset(Dataset):
    """Deterministic per-index noise seeds for FID evaluation.

    sample i always produces the same (z [, cls]) regardless of worker/GPU,
    so features gathered from multiple GPUs are consistent.
    """

    def __init__(
            self,
            n_samples: int,
            image_shape: Tuple,
            class_condition: bool = False,
            n_classes: Optional[int] = None,
            sample_labels: bool = False,
            seed: int = 0,
    ):
        self.n_samples = n_samples
        self.image_shape = image_shape
        self.class_condition = class_condition
        self.n_classes = n_classes
        self.sample_labels = sample_labels
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        item = {'z': torch.randn(self.image_shape, generator=gen)}
        if self.class_condition:
            assert self.n_classes is not None
            if self.sample_labels:
                cls = int(torch.randint(0, self.n_classes, (1,), generator=gen))
            else:
                cls = idx % self.n_classes
            item['cls'] = torch.tensor(cls, dtype=torch.long)
        return item


def get_dataset_info(dataset: str):
    """Return (image_shape, n_classes) for a known dataset."""
    if dataset == 'lsun-church_outdoor':
        return (3, 256, 256), None
    elif dataset == 'imagenet-1k-256':
        return (3, 256, 256), 1000
    else:
        raise ValueError(f'unknown dataset {dataset}')


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


def apply_transform(examples, transform):
    examples['image'] = [transform(image.convert('RGB')) for image in examples['image']]
    return examples


class LSUNWrapper(LSUN):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return {'image': image, 'label': label}


def load_dataset(dataset, data_dir, train=True, augumentations=None):
    transform = get_augmentations(augumentations)
    if dataset == 'lsun-church_outdoor':
        transform.extend([
            T.Resize(256), T.CenterCrop(256),
            T.ToTensor(), T.Normalize(mean=0.5, std=0.5),
        ])
        return LSUNWrapper(
            data_dir,
            classes=['church_outdoor_train' if train else 'church_outdoor_val'],
            transform=T.Compose(transform),
        )
    elif dataset == 'imagenet-1k-256':
        ds = load_hf_dataset(
            'benjamin-paine/imagenet-1k-256x256',
            split='train' if train else 'validation',
        )
        transform.extend([T.ToTensor(), T.Normalize(mean=0.5, std=0.5)])
        ds.set_transform(partial(apply_transform, transform=T.Compose(transform)))
        return ds
    else:
        raise ValueError(f'unknown dataset {dataset}')


def make_generation_seed(dataset, n_examples, seed=None, class_condition=True,
                         sample_labels=False):
    def _gen():
        return torch.Generator().manual_seed(seed) if seed is not None else None

    if dataset == 'lsun-church_outdoor':
        image_shape, n_classes = (3, 256, 256), 1
        assert not class_condition, 'lsun-church_outdoor has no class labels'
    elif dataset == 'imagenet-1k-256':
        image_shape, n_classes = (3, 256, 256), 1000
    else:
        raise ValueError(f'unknown dataset {dataset}')

    z = torch.randn((n_examples, *image_shape), generator=_gen())
    ret = {'z': z}
    if class_condition:
        if sample_labels:
            cls = torch.randint(0, n_classes, (n_examples,), generator=_gen())
        else:
            cls = torch.arange(n_examples) % n_classes
        ret['cls'] = cls
    return ret


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
            model: JITTransformer,
            scheduler: JITScheduler,
            sampler: JITSampler,
            resume_ckpt_dir: Optional[str] = None,
            overwrite: bool = False,
    ):
        self.accelerator = accelerator
        self.arg = arg
        self.scheduler = scheduler
        self.sampler = sampler

        if arg.dataloader_num_workers == 0:
            warnings.warn('set dataloader_num_workers > 0 for reproducibility')

        accelerator.print(f'Trainer args:\n{arg}')
        accelerator.print(f'num_processes: {accelerator.num_processes}  '
                          f'mixed_precision: {accelerator.mixed_precision}')

        # ── Output directory (main process) ──────────────────────────────
        self.resume_ckpt_dir = resume_ckpt_dir
        if accelerator.is_main_process:
            if resume_ckpt_dir is not None:
                assert resume_ckpt_dir == arg.output_dir, (
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

        # ── Model: compile before prepare (cleaner state_dict keys) ──────
        if arg.compile:
            accelerator.print('Compiling model...')
            model = torch.compile(model)
            accelerator.print('Model compiled.')

        # EMA: track raw model params BEFORE DDP wrapping so parameters are
        # the same physical tensors after prepare.
        self.ema_model = EMAModel(
            model.parameters(),
            use_ema_warmup=True,
            inv_gamma=arg.ema_inv_gamma,
            power=arg.ema_power,
        )

        # ── Datasets & Dataloaders ───────────────────────────────────────
        self.dataset = self._get_train_dataset()
        self.valid_dataset = self._get_valid_dataset()

        optimizer = self._make_optimizer(model)
        lr_scheduler_obj = self._make_lr_scheduler(optimizer)
        train_loader = self._make_train_dataloader()
        valid_loader = self._make_valid_dataloader()

        # ── Prepare with accelerate ──────────────────────────────────────
        # Handles DDP wrapping, DistributedSampler insertion, and optional
        # gradient scaler for mixed precision.
        prepare_args = [model, optimizer, train_loader, valid_loader]
        if lr_scheduler_obj is not None:
            prepare_args.append(lr_scheduler_obj)

        prepared = accelerator.prepare(*prepare_args)
        self.model       = prepared[0]
        self.optimizer   = prepared[1]
        self.train_dataloader = prepared[2]
        self.valid_dataloader = prepared[3]
        self.lr_scheduler = prepared[4] if lr_scheduler_obj is not None else None

        # Unwrapped model for inference, EMA updates, and saving.
        self.raw_model = accelerator.unwrap_model(self.model)
        self.ema_model.to(accelerator.device)

        self.steps_per_epoch = len(self.train_dataloader)
        accelerator.print(f'Steps per epoch: {self.steps_per_epoch}')

        # ── Training counters ────────────────────────────────────────────
        self.global_steps = 0
        self.epochs = 0
        self.steps_in_epoch = 0

        # ── Eval seed (fixed across all GPUs for comparable visualizations) ──
        self.eval_seed = make_generation_seed(
            arg.dataset, arg.eval_n_examples,
            seed=arg.seed,
            class_condition=arg.class_conditioning,
            sample_labels=True,
        )

        # ── FID setup ────────────────────────────────────────────────────
        self.fid_dataset = None
        self.fid_reference = None
        if arg.fid_eval_steps is not None:
            if arg.adjust_fid_n:
                assert arg.fid_n_examples == max(arg.fid_adjust_subsets), (
                    'fid_n_examples must equal max(fid_adjust_subsets) for FID extrapolation')
                arg.fid_adjust_subsets.sort()

            image_shape, n_classes = get_dataset_info(arg.dataset)
            self.fid_dataset = FIDNoiseDataset(
                n_samples=arg.fid_n_examples,
                image_shape=image_shape,
                class_condition=arg.class_conditioning,
                n_classes=n_classes,
                sample_labels=arg.fid_sample_labels,
                seed=arg.seed,
            )
            self.fid_reference = load_hidden_parameters(
                arg.fid_reference_dataset, save=False)

        self.ckpt_base_dir = (
            os.path.join(arg.output_dir, 'ckpts') if arg.save_steps else None
        )

    # ── Context helpers ──────────────────────────────────────────────────

    def autocast(self):
        return self.accelerator.autocast()

    # ── Setup helpers ────────────────────────────────────────────────────

    def _get_train_dataset(self):
        ds = load_dataset(self.arg.dataset, self.arg.dataset_dir,
                          train=True, augumentations=self.arg.augmentations)
        self.accelerator.print(f'Train dataset: {ds}')
        return ds

    def _get_valid_dataset(self):
        ds = load_dataset(self.arg.dataset, self.arg.dataset_dir, train=False)
        self.accelerator.print(f'Valid dataset: {ds}')
        return ds

    def _make_train_dataloader(self):
        rng = torch.Generator()
        rng.manual_seed(self.arg.seed)
        return DataLoader(
            self.dataset,
            batch_size=self.arg.batch_size,
            shuffle=True,
            drop_last=self.arg.dataloader_drop_last,
            pin_memory=self.arg.dataloader_pin_memory,
            num_workers=self.arg.dataloader_num_workers,
            worker_init_fn=seed_worker,
            generator=rng,
        )

    def _make_valid_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.arg.generation_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.arg.dataloader_pin_memory,
            num_workers=self.arg.dataloader_num_workers,
        )

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
        """Save EMA model checkpoint (main process only)."""
        if not self.accelerator.is_main_process:
            return
        os.makedirs(ckpt_dir, exist_ok=True)

        fp32_keywords = ['norm', 'class_embedding']

        def to_mixed_dtype(sd):
            return {
                k: v.cpu().to(
                    torch.float32 if any(kw in k.lower() for kw in fp32_keywords)
                    else torch.bfloat16
                )
                for k, v in sd.items()
            }

        self.ema_model.store(self.raw_model.parameters())
        self.ema_model.copy_to(self.raw_model.parameters())

        state_dict = self.raw_model.state_dict()
        if self.arg.compile:
            state_dict = fix_state_dict(state_dict)
        state_dict = to_mixed_dtype(state_dict)
        torch.save(state_dict, os.path.join(ckpt_dir, 'model.pt'))

        self.ema_model.restore(self.raw_model.parameters())
        self.accelerator.print(f'Saved ckpt → {ckpt_dir}')

    def save_latest_ckpt(self):
        """Save full training state for resumption.

        accelerator.save_state writes model / optimizer / scheduler and
        per-process RNG states.  EMA + counters are written to extra.pt by
        the main process.
        Must be called on ALL processes.
        """
        latest_dir = os.path.join(self.arg.output_dir, 'latest_state')
        os.makedirs(latest_dir, exist_ok=True)
        self.accelerator.save_state(latest_dir)

        if self.accelerator.is_main_process:
            extra = {
                'global_steps': self.global_steps,
                'epochs': self.epochs,
                'steps_in_epoch': self.steps_in_epoch,
                'ema_state_dict': self.ema_model.state_dict(),
            }
            torch.save(extra, os.path.join(latest_dir, 'extra.pt'))

        self.accelerator.wait_for_everyone()

    def load_latest_ckpt(self, ckpt_dir: str):
        """Load full training state.  Must be called on ALL processes."""
        latest_dir = os.path.join(ckpt_dir, 'latest_state')
        self.accelerator.load_state(latest_dir)

        extra = torch.load(
            os.path.join(latest_dir, 'extra.pt'),
            map_location='cpu', weights_only=False,
        )
        self.global_steps = extra['global_steps']
        self.epochs = extra['epochs']
        self.steps_in_epoch = extra['steps_in_epoch']
        self.ema_model.load_state_dict(extra['ema_state_dict'])

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
                {'global_steps': steps,
                 'epochs': steps / self.steps_per_epoch,
                 **logs},
                step=steps,
            )

    # ── Training ─────────────────────────────────────────────────────────

    def train_on_batch(self, x: torch.Tensor, label=None) -> float:
        uncond_mask = None
        if label is not None:
            uncond_mask = torch.bernoulli(
                torch.full((label.size(0),), self.arg.p_uncond, device=label.device)
            )

        with self.autocast():
            loss = self.scheduler.get_loss(
                x, self.model, cls=label, uncond_mask=uncond_mask)

        self.accelerator.backward(loss)
        if self.arg.clip_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.arg.clip_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.ema_model.step(self.raw_model.parameters())

        return loss.detach().item()

    # ── Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_eval_examples(self) -> dict:
        """Generate example images.  Runs on main process only."""
        if not self.accelerator.is_main_process:
            return {}

        self.raw_model.eval()
        z = self.eval_seed['z'].to(self.accelerator.device)
        cond = (self.eval_seed['cls'].to(self.accelerator.device)
                if 'cls' in self.eval_seed else None)

        with self.autocast():
            samples = self.sampler.sample(z, self.raw_model, cond=cond)
        result = {'examples': torch.clamp(samples, -1, 1).cpu()}

        self.ema_model.store(self.raw_model.parameters())
        self.ema_model.copy_to(self.raw_model.parameters())
        with self.autocast():
            ema_samples = self.sampler.sample(z, self.raw_model, cond=cond)
        result['ema_examples'] = torch.clamp(ema_samples, -1, 1).cpu()
        self.ema_model.restore(self.raw_model.parameters())

        self.raw_model.train()
        return result

    @torch.no_grad()
    def evaluate_validation_loss(self) -> dict:
        """Compute validation loss across all GPUs.

        Each GPU processes its shard of the (distributed) valid dataloader.
        Loss sums are all-reduced and divided by the full valid dataset size.
        """
        self.raw_model.eval()

        def _loss_over_loader(model):
            loss_sum = torch.zeros(1, device=self.accelerator.device)
            for batch in self.valid_dataloader:
                x = batch['image'].to(self.accelerator.device)
                cls = (batch['label'].to(self.accelerator.device)
                       if 'label' in batch else None)
                with self.autocast():
                    loss = self.scheduler.get_loss(x, model, cls=cls)
                loss_sum += loss.detach() * x.size(0)

            # Sum across all GPUs
            total = self.accelerator.reduce(loss_sum, reduction='sum')
            # Slight over-count if DistributedSampler adds padding, acceptable
            return (total / len(self.valid_dataset)).item()

        result = {'loss': _loss_over_loader(self.raw_model)}

        self.ema_model.store(self.raw_model.parameters())
        self.ema_model.copy_to(self.raw_model.parameters())
        result['ema_loss'] = _loss_over_loader(self.raw_model)
        self.ema_model.restore(self.raw_model.parameters())

        self.raw_model.train()
        return result

    @torch.no_grad()
    def evaluate_fid_streaming(self) -> Optional[dict]:
        """Multi-GPU streaming FID evaluation.

        Each GPU:
          1. Gets a contiguous shard of FIDNoiseDataset via DistributedSampler.
          2. Generates samples with EMA model in generation_batch_size chunks.
          3. Runs Inception immediately (no image accumulation).
          4. Returns local inception features.

        Features are all-gathered; the main process slices to fid_n_examples,
        computes FID (and optionally FID@inf), logs results, and returns the
        result dict.  Non-main processes return None.
        """
        # ── Distribute noise seeds across GPUs ───────────────────────────
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

        # ── Load Inception on this GPU ────────────────────────────────────
        inception = InceptionV3(
            [InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=False
        ).to(self.accelerator.device).eval()

        # ── Generate with EMA weights ─────────────────────────────────────
        self.raw_model.eval()
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
            z = batch['z'].to(self.accelerator.device)
            cond = (batch['cls'].to(self.accelerator.device)
                    if 'cls' in batch else None)

            with self.autocast():
                samples = self.sampler.sample(z, self.raw_model, cond=cond)
            samples = torch.clamp(samples, -1, 1)

            # Inception in float32; output: (B, 2048, 1, 1) → (B, 2048)
            feats = inception(samples.float())[0].flatten(1)
            local_features.append(feats.cpu())

        self.ema_model.restore(self.raw_model.parameters())
        self.raw_model.train()

        del inception
        torch.cuda.empty_cache()

        # ── Gather features across all GPUs ──────────────────────────────
        # DistributedSampler (drop_last=False) pads the last rank so every rank
        # holds the same number of samples → all-gather tensors are same size.
        local_tensor = torch.cat(local_features, dim=0).to(self.accelerator.device)
        all_features = self.accelerator.gather(local_tensor)  # (N_padded, 2048)

        if not self.accelerator.is_main_process:
            return None

        # Strip padding, convert to float64 numpy (expected by pytorch-fid)
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
            result['FID'] = fid_out['fids'][-1]
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
        """Image examples (main process) + validation loss (all GPUs)."""
        # Image examples ─ main process only
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

        # Validation loss ─ all GPUs contribute
        val_loss = self.evaluate_validation_loss()
        self.log(steps, **{f'val/{k}': v for k, v in val_loss.items()})

    # ── Main training loop ────────────────────────────────────────────────

    def train(self):
        if self.resume_ckpt_dir is not None:
            self.load_latest_ckpt(self.resume_ckpt_dir)
            # Note: on resume we start a new epoch rather than resuming
            # mid-epoch.  Model + optimizer + EMA are fully restored.

        self.model.train()

        with tqdm.tqdm(
            initial=self.global_steps,
            total=self.arg.max_steps,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.global_steps < self.arg.max_steps:
                for batch in self.train_dataloader:
                    x = batch['image'].to(self.accelerator.device)
                    cls = (batch['label'].to(self.accelerator.device)
                           if 'label' in batch else None)

                    loss = self.train_on_batch(x, cls)
                    self.global_steps += 1
                    self.epochs = self.global_steps // self.steps_per_epoch
                    self.steps_in_epoch = self.global_steps % self.steps_per_epoch
                    pbar.update(1)

                    # ── Logging ───────────────────────────────────────────
                    if self.global_steps % self.arg.logging_steps == 0:
                        if self.accelerator.is_main_process:
                            pbar.set_postfix({'loss': f'{loss:.5f}'})
                        logs = {
                            'loss': loss,
                            'ema_decay': self.ema_model.cur_decay_value,
                        }
                        if self.lr_scheduler is not None:
                            logs['lr'] = self.lr_scheduler.get_last_lr()[0]
                        self.log(self.global_steps, **logs)

                    # ── Eval (images + val loss) ──────────────────────────
                    if self.global_steps % self.arg.eval_steps == 0:
                        self.evaluate(self.global_steps)
                        self.model.train()

                    # ── Checkpoint ────────────────────────────────────────
                    if (self.ckpt_base_dir is not None
                            and self.global_steps % self.arg.save_steps == 0):
                        ckpt_dir = os.path.join(
                            self.ckpt_base_dir, f'ckpt-{self.global_steps:06d}')
                        self.save_ckpt(ckpt_dir)
                        self.save_latest_ckpt()

                    # ── FID ───────────────────────────────────────────────
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

    if resume_ckpt_dir is not None:
        train_arg_json = os.path.join(resume_ckpt_dir, 'train_args.json')

    with open(train_arg_json) as f:
        args_dict = json.load(f)

    # Backwards-compat: typo fix and device field removal (single-GPU artifact)
    if 'lr_schduler_cfg' in args_dict:
        args_dict['lr_scheduler_cfg'] = args_dict.pop('lr_schduler_cfg')
    args_dict.pop('device', None)

    train_args = TrainArgs(**args_dict)

    accelerator = Accelerator(mixed_precision='bf16')

    # Each process uses a different seed so diffusion noise is uncorrelated
    # across GPUs, improving gradient variance reduction via DDP all-reduce.
    seed_everything(train_args.seed + accelerator.process_index)

    model = JITTransformer(**train_args.model_cfg)
    scheduler = JITScheduler(**train_args.scheduler_cfg)
    sampler = JITSampler(scheduler=scheduler, **train_args.sampler_cfg)

    assert scheduler.pred_type == sampler.pred_type, (
        f'scheduler.pred_type "{scheduler.pred_type}" != '
        f'sampler.pred_type "{sampler.pred_type}"'
    )

    accelerator.print(f'Model parameters: {count_parameters(model) / 1e6:.2f}M')

    trainer = Trainer(
        accelerator=accelerator,
        arg=train_args,
        model=model,
        scheduler=scheduler,
        sampler=sampler,
        resume_ckpt_dir=resume_ckpt_dir,
        overwrite=overwrite,
    )
    trainer.train()


if __name__ == '__main__':
    fire.Fire(main)

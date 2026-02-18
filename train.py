
import os
import sys
import shutil
import random as rd
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
import json
import tqdm.auto as tqdm
import dataclasses
from typing import Optional, List, Tuple, Type
import wandb
import fire
from diffusers.training_utils import EMAModel

from diffusion.scheduler import get_scheduler, BaseScheduler
from diffusion.sampler import get_sampler, BaseSampler
from modeling import get_model
from utils import count_parameters, get_augmentations


WANDB_PROJECT_NAME = 'noh-diffusion'


@dataclasses.dataclass
class TrainArgs:
    output_dir: str
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None

    max_steps: int = 100_000
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: Optional[int] = None
    eval_n_examples: int = 40
    guidance_scale: float = 1.0
    # save_limits: Optional[int] = None

    batch_size: int = 64
    lr: float = 2e-4
    # lr_scheduler: Optional[str] = None
    # lr_warmup_steps: int = 0
    optimizer: str = 'adamw'
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    clip_grad_norm: float = 1.0

    use_ema: bool = False
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75

    dataset: str = 'mnist'
    dataset_dir: str = 'datasets'
    augmentations: List[str] = dataclasses.field(default_factory=list)
    dataloader_num_workers: int = 2
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True

    device: str = 'cuda'
    seed: int = 42

    p_uncond: float = 0.2
    model_type: str = 'unet'
    model_cfg: dict = dataclasses.field(default_factory=dict)
    scheduler_type: str = 'beta'
    scheduler_cfg: dict = dataclasses.field(default_factory=dict)
    sampler_type: str = 'ddim'
    sampler_cfg: dict = dataclasses.field(default_factory=dict)


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


def load_dataset(dataset, data_dir, train=True, augumentations: Optional[List[str]] = None):
    transform = get_augmentations(augumentations)
    transform.extend([
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),
    ])
    transform = T.Compose(transform)

    if dataset == 'cifar10':
        dataset = CIFAR10(data_dir,
                          transform=transform,
                          download=True,
                          train=train)
    elif dataset == 'mnist':
        dataset = MNIST(data_dir,
                        transform=transform,
                        download=True,
                        train=train)
    else:
        raise ValueError(f'unknown dataset {dataset}')
    
    return dataset


def get_eps_pred_func(model, cls=None, guidance_scale=1.0):
    def pred_fn(z, t):
        if guidance_scale == 1.0 and cls is not None:
            eps_pred = model(z, t, cls=cls)
        elif guidance_scale != 0 and cls is not None:
            z = torch.cat([z, z], dim=0)
            t = torch.cat([t, t], dim=0)
            cls_cond = torch.cat([cls, cls], dim=0)
            uncond_mask = torch.cat([torch.ones_like(cls), torch.zeros_like(cls)], dim=0)

            eps_pred = model(z, t, cls=cls_cond, uncond_mask=uncond_mask)
            eps_uncond, eps_cond = torch.chunk(eps_pred, 2, dim=0)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_pred = model(z, t, cls=cls)
        return eps_pred
    return pred_fn


class Trainer:
    def __init__(
            self,
            arg: TrainArgs,
            model: nn.Module,
            scheduler: Type[BaseScheduler],
            sampler: Optional[Type[BaseSampler]] = None,
            resume_ckpt_dir: Optional[str] = None,
    ):
        if arg.dataloader_num_workers == 0:
            warnings.warn(f'set dataloader_num_workers > 0 for reproducibility')

        self.arg = arg
        self.model = model
        self.scheduler = scheduler
        self.sampler = sampler

        print(f'Trainer arg:')
        print(self.arg)

        self.resume_ckpt_dir = resume_ckpt_dir
        if resume_ckpt_dir is not None:
            assert resume_ckpt_dir == self.arg.output_dir, \
                f'resume_ckpt_dir "{resume_ckpt_dir}" does not match arg.output_dir "{self.arg.output_dir}"'
        else:
            if os.path.exists(self.arg.output_dir) and len(os.listdir(self.arg.output_dir)) > 0:
                overwrite_msj = f'"{self.arg.output_dir}" already exists and is not empty, overwrite? (Y/n): '
                user_input = input(overwrite_msj)
                if user_input.lower().strip() != 'y':
                    raise ValueError(f'"{self.arg.output_dir}" already exists and is not empty')
                shutil.rmtree(self.arg.output_dir)

        self.wandb_run = None
        if self.arg.wandb_run_name is not None or self.arg.wandb_run_id is not None:
            self.prepare_wandb_run(self.arg.wandb_run_name)
        
        os.makedirs(self.arg.output_dir, exist_ok=True)
        with open(os.path.join(self.arg.output_dir, 'train_args.json'), 'w') as f:
            json.dump(dataclasses.asdict(self.arg), f, indent=2)

        self.device = torch.device(self.arg.device)
        print(f'using device: {self.device}')

        self.ema_model = None
        if self.arg.use_ema:
            self.ema_model = EMAModel(
                self.model.parameters(),
                use_ema_warmup=True,
                inv_gamma=self.arg.ema_inv_gamma,
                power=self.arg.ema_power,
            )
            self.ema_model.to(self.device)

        self.global_steps = 0
        self.steps_in_epoch = 0
        self.epochs = 0

        self.dataset = self.get_train_dataset()
        self.dataloader = self.get_train_dataloader()
        self.steps_per_epoch = len(self.dataloader)
        print(f'steps per epoch: {self.steps_per_epoch}')
        self.optimizer = self.get_optimizer()

        self.eval_examples = self.get_eval_examples(self.arg.seed)

        self.model.to(self.device)

        self.ckpt_base_dir = None
        if self.arg.save_steps is not None:
            self.ckpt_base_dir = os.path.join(self.arg.output_dir, 'ckpts')
        self.saved_ckpt_paths = []


    def get_optimizer(self):
        if self.arg.optimizer == 'adam':
            optim_cls = torch.optim.Adam
        elif self.arg.optimizer == 'adamw':
            optim_cls = torch.optim.AdamW
        else:
            raise ValueError(f'unknown optimizer {self.arg.optimizer}')

        optim = optim_cls(
            self.model.parameters(),
            lr=self.arg.lr,
            betas=self.arg.adam_betas
        )
        print(f'optimizer ready')
        return optim

    def get_train_dataset(self):
        dataset = load_dataset(
            self.arg.dataset,
            self.arg.dataset_dir, 
            train=True,
            augumentations=self.arg.augmentations
        )
        print(f'dataset ready: {dataset}')
        return dataset

    def get_train_dataloader(self, dataset=None):
        if dataset is None:
            dataset = self.dataset

        rng = torch.Generator()
        rng.manual_seed(self.arg.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.arg.batch_size,
            shuffle=True,
            drop_last=self.arg.dataloader_drop_last,
            pin_memory=self.arg.dataloader_pin_memory,
            num_workers=self.arg.dataloader_num_workers,
            worker_init_fn=seed_worker,
            generator=rng,
        )
        print(f'dataloader ready')
        return loader


    def prepare_wandb_run(self, run_name):
        if self.arg.wandb_run_id is not None:
            self.wandb_run = wandb.init(
                project=WANDB_PROJECT_NAME,
                id=self.arg.wandb_run_id,
                resume='must',
            )
        else:
            cfg = dataclasses.asdict(self.arg)
            self.wandb_run = wandb.init(
                name=run_name,
                project=WANDB_PROJECT_NAME,
                config=cfg,
                dir=self.arg.output_dir,
            )
            self.arg.wandb_run_id = self.wandb_run.id
        print(f'Wandb - run name: {self.wandb_run.name}, id: {self.wandb_run.id}')


    def save_ckpt(self, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(ckpt_dir, 'model.pt'))
        if self.ema_model is not None:
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, 'ema_model.pt'))
            self.ema_model.restore(self.model.parameters())

        print(f'saved ckpt {ckpt_dir}')
    

    def save_latest_ckpt(self):
        latest_ckpt_path = os.path.join(self.arg.output_dir, 'latest.pt')
        ckpt = {
            'global_steps': self.global_steps,
            'epochs': self.epochs,
            'steps_in_epoch': self.steps_in_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'rd_rng_state': rd.getstate(),
        }
        if self.ema_model is not None:
            ckpt['ema_model_state_dict'] = self.ema_model.state_dict()
        torch.save(ckpt, latest_ckpt_path)

    def load_ckpt_legacy(self, ckpt_dir):
        steps = torch.load(os.path.join(ckpt_dir, 'global_steps.pt'))
        self.global_steps = steps['global_steps']
        self.epochs = steps['epochs']
        self.steps_in_epoch = steps['steps_in_epoch']

        self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model.pt')))
        self.optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, 'optimizer.pt')))
        if self.ema_model is not None:
            self.ema_model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'ema_model.pt')))

        random_states = torch.load(os.path.join(ckpt_dir, 'random_states.pt'))
        torch.set_rng_state(random_states['rng_state'])
        torch.cuda.set_rng_state(random_states['cuda_rng_state'])
        np.random.set_state(random_states['np_rng_state'])
        rd.setstate(random_states['rd_rng_state'])

        print(f'loaded ckpt {ckpt_dir}')
    
    def load_latest_ckpt(self, ckpt_dir):
        latest_ckpt_path = os.path.join(ckpt_dir, 'latest.pt')
        ckpt = torch.load(latest_ckpt_path, weights_only=False)
        print(ckpt.keys())

        self.global_steps = ckpt['global_steps']
        self.epochs = ckpt['epochs']
        self.steps_in_epoch = ckpt['steps_in_epoch']

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.ema_model is not None:
            self.ema_model.load_state_dict(ckpt['ema_model_state_dict'])

        torch.set_rng_state(ckpt['rng_state'])
        torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
        np.random.set_state(ckpt['np_rng_state'])
        rd.setstate(ckpt['rd_rng_state'])

        print(f'loaded latest ckpt from {latest_ckpt_path}')


    def log(self, steps, **logs):
        with open(os.path.join(self.arg.output_dir, 'train_log.jsonl'), 'a') as f:
            f.write(json.dumps(dict(steps=steps, **logs)) + '\n')
        if self.wandb_run is not None:
            logs.update({
                'global_steps': steps,
                'epochs': steps / self.steps_per_epoch,
            })
            self.wandb_run.log(logs, step=steps)


    def train_on_batch(self, x, label=None):
        uncond_mask = torch.bernoulli(self.arg.p_uncond * torch.ones_like(label))

        loss = self.scheduler.get_loss(x, self.model, cls=label, uncond_mask=uncond_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.arg.clip_grad_norm
        )
        self.optimizer.step()
        self.model.zero_grad()
        if self.ema_model is not None:
            self.ema_model.step(self.model.parameters())

        return loss.item()
    
    def get_eval_examples(self, seed=None):
        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(seed)
        
        if self.arg.dataset == 'mnist':
            image_shape = (1, 28, 28)
            n_classes = 10
        elif self.arg.dataset == 'cifar10':
            image_shape = (3, 32, 32)
            n_classes = 10
        else:
            raise ValueError(f'unknown dataset {self.arg.dataset}')
        
        z = torch.randn((self.arg.eval_n_examples, *image_shape), device=self.device, generator=gen)
        cls = torch.arange(self.arg.eval_n_examples, device=self.device) % n_classes

        return {'z': z, 'cls': cls}
    
    def generate_eval_examples(self):
        self.model.eval()
        
        pred_fn = get_eps_pred_func(
            self.model, 
            cls=self.eval_examples['cls'], 
            guidance_scale=self.arg.guidance_scale
        )

        with torch.no_grad():
            samples = self.sampler.sample(
                self.eval_examples['z'],
                self.scheduler,
                pred_fn
            ).cpu()
        ret = {'examples': samples}
        
        if self.ema_model is not None:
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())

            with torch.no_grad():
                ema_samples = self.sampler.sample(
                    self.eval_examples['z'],
                    self.scheduler,
                    pred_fn
                ).cpu()
            ret['ema_examples'] = ema_samples

            self.ema_model.restore(self.model.parameters())

        return ret


    def evaluate(self, steps):
        save_dir = os.path.join(self.arg.output_dir, f'examples')
        os.makedirs(save_dir, exist_ok=True)

        def save_samples(samples, name):
            grid_image = T.ToPILImage()(
                make_grid(samples, nrow=10, normalize=True, value_range=(-1, 1)))
            save_path = os.path.join(save_dir, f'{steps:06d}_{name}.png')
            grid_image.save(save_path)
            return grid_image

        examples = self.generate_eval_examples()
        examples = {k: save_samples(v, k) for k, v in examples.items()}

        if self.wandb_run is not None:
            logs = {
                'global_steps': steps,
                'epochs': steps / self.steps_per_epoch,
            }
            logs.update(**{k: wandb.Image(v) for k, v in examples.items()})
            self.wandb_run.log(logs, step=steps)

    def skip_previous_steps(self):
        print(f'skipping {self.epochs} epochs {self.steps_in_epoch} steps...')
        for _ in tqdm.tqdm(range(self.epochs), leave=False):
            next(iter(self.dataloader))
        iterator = iter(self.dataloader)
        for _ in tqdm.tqdm(range(self.steps_in_epoch), leave=False):
            next(iterator)
        return iterator

    def train(self):

        if self.resume_ckpt_dir is not None:
            self.load_latest_ckpt(self.resume_ckpt_dir)

        if self.global_steps > 0:
            iterator = self.skip_previous_steps()
        else:
            iterator = None

        print('train start')

        losses = []
        self.model.train()

        with tqdm.tqdm(initial=self.global_steps, total=self.arg.max_steps) as pbar:
            while self.global_steps < self.arg.max_steps:
                if iterator is None:
                    iterator = iter(self.dataloader)
                for x, cls in iterator:
                    x, cls = x.to(self.device), cls.to(self.device)
                    loss = self.train_on_batch(x, cls)

                    losses.append(loss)
                    self.global_steps += 1
                    self.epochs = self.global_steps // self.steps_per_epoch
                    self.steps_in_epoch = self.global_steps % self.steps_per_epoch
                    pbar.update(1)
                    
                    if self.global_steps % self.arg.logging_steps == 0:
                        loss_mean = np.mean(losses)
                        logs = dict(loss=loss_mean)
                        losses.clear()
                        pbar.set_postfix({'loss': f'{loss_mean:.5f}'})

                        self.log(
                            self.global_steps,
                            **logs
                        )
                    
                    if self.global_steps % self.arg.eval_steps == 0:
                        self.evaluate(self.global_steps)
                        self.model.train()
                    
                    if self.ckpt_base_dir is not None and self.global_steps % self.arg.save_steps == 0:
                        ckpt_dir = os.path.join(self.ckpt_base_dir, f'ckpt-{self.global_steps:06d}')
                        self.save_ckpt(ckpt_dir)
                        self.save_latest_ckpt()

                    if self.global_steps >= self.arg.max_steps:
                        break
                iterator = None

        print('train done')



def main(
        train_arg_json: Optional[str] = None,
        resume_ckpt_dir: Optional[str] = None,
):
    assert train_arg_json is not None or resume_ckpt_dir is not None
    if resume_ckpt_dir is not None:
        print(f'loading train args from resume ckpt dir "{resume_ckpt_dir}"')
        train_arg_json = os.path.join(resume_ckpt_dir, 'train_args.json')

    print(train_arg_json)
    with open(train_arg_json, 'r') as f:
        train_args = TrainArgs(**json.load(f))

    seed_everything(train_args.seed)
    
    model = get_model(train_args.model_type, **train_args.model_cfg)
    scheduler = get_scheduler(train_args.scheduler_type, **train_args.scheduler_cfg)
    sampler = get_sampler(train_args.sampler_type, **train_args.sampler_cfg)

    assert scheduler.pred_type == sampler.pred_type, \
        f'scheduler pred_type "{scheduler.pred_type}" does not match sampler pred_type "{sampler.pred_type}"'

    print(f'params: {count_parameters(model) / 1e+6}')

    trainer = Trainer(train_args, model, scheduler, sampler, resume_ckpt_dir=resume_ckpt_dir)
    trainer.train()


if __name__ == '__main__':
    fire.Fire(main)

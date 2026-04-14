
import os
import sys
import shutil
import random as rd
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import LSUN
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import json
import tqdm.auto as tqdm
import dataclasses
from functools import partial
from typing import Optional, List, Tuple, Type
import wandb
import fire
from diffusers.training_utils import EMAModel
from transformers import get_scheduler as get_lr_scheduler
from datasets import load_dataset as load_hf_dataset

from diffusion.scheduler import JITScheduler
from diffusion.sampler import JITSampler
from modeling.transformer import JITTransformer
from utils import count_parameters, get_augmentations
from utils.fid import load_hidden_parameters, calculate_frechet_distance, \
    calc_inception_features, inception_features_to_hidden_parameters
from utils.fid_infinity import fid_extrapolation


WANDB_PROJECT_NAME = 'noh-diffusion'


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
    # save_limits: Optional[int] = None

    fid_eval_steps: Optional[int] = 10000
    fid_reference_dataset: str = 'lsun-church_outdoor-train'
    fid_n_examples: int = 10000
    fid_sample_labels: bool = False
    generation_batch_size: int = 256
    inception_batch_size: int = 512
    adjust_fid_n: bool = True
    fid_adjust_subsets: List[int] = dataclasses.field(
        default_factory=lambda: [4000, 6000, 8000, 10000])

    batch_size: int = 256
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
    # augmentations: List[str] = dataclasses.field(default_factory=list)
    augmentations: List[str] = dataclasses.field(default_factory=lambda: ['RandomHorizontalFlip'])
    # augmentations: List[str] = ['RandomHorizontalFlip']
    dataloader_num_workers: int = 4
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True

    device: str = 'cuda'
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


def load_dataset(dataset, data_dir, train=True, augumentations: Optional[List[str]] = None):
    transform = get_augmentations(augumentations)

    if dataset == 'lsun-church_outdoor':
        transform.extend([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])
        transform = T.Compose(transform)
        dataset = LSUNWrapper(
            data_dir,
            classes=['church_outdoor_train' if train else 'church_outdoor_val'],
            transform=transform
        )
    elif dataset == 'imagenet-1k-256':
        imagenet_id = 'benjamin-paine/imagenet-1k-256x256'
        dataset = load_hf_dataset(imagenet_id, split='train' if train else 'validation')
        transform.extend([
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])
        transform = T.Compose(transform)
        transform_fn = partial(apply_transform, transform=transform)
        dataset.set_transform(transform_fn)
    else:
        raise ValueError(f'unknown dataset {dataset}')
    
    return dataset


def make_generation_seed(dataset, n_examples, seed=None, class_condition=True, sample_labels=False):
    def get_generator():
        return torch.Generator().manual_seed(seed) if seed is not None else None
    
    if dataset == 'lsun-church_outdoor':
        image_shape = (3, 256, 256)
        n_classes = 1
        assert not class_condition, 'class conditioning is not supported for lsun-church_outdoor'
    elif dataset == 'imagenet-1k-256':
        image_shape = (3, 256, 256)
        n_classes = 1000
    else:
        raise ValueError(f'unknown dataset {dataset}')
    
    z = torch.randn((n_examples, *image_shape), generator=get_generator())
    ret = {'z': z}
    if class_condition:
        if sample_labels:
            cls = torch.randint(0, n_classes, (n_examples,), generator=get_generator())
        else:
            cls = torch.arange(n_examples) % n_classes
        ret['cls'] = cls

    return ret

def fix_state_dict(state_dict):
    state_dict_fix = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_k = k[len('_orig_mod.'):]
            state_dict_fix[new_k] = v
        else:
            state_dict_fix[k] = v
    return state_dict_fix

class Trainer:
    def __init__(
            self,
            arg: TrainArgs,
            model: JITTransformer,
            scheduler: JITScheduler,
            sampler: JITSampler,
            resume_ckpt_dir: Optional[str] = None,
            overwrite: bool = False,
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
                if not overwrite:
                    overwrite_msj = f'\n"{self.arg.output_dir}" already exists and is not empty, overwrite? (Y/n): '
                    user_input = input(overwrite_msj)
                    if user_input.lower().strip() != 'y':
                        raise ValueError(f'"{self.arg.output_dir}" already exists and is not empty')
                    
                print(f'overwriting "{self.arg.output_dir}"...\n')
                shutil.rmtree(self.arg.output_dir)

        self.wandb_run = None
        if self.arg.wandb_run_name is not None or self.arg.wandb_run_id is not None:
            self.prepare_wandb_run(self.arg.wandb_run_name)
        
        os.makedirs(self.arg.output_dir, exist_ok=True)
        with open(os.path.join(self.arg.output_dir, 'train_args.json'), 'w') as f:
            json.dump(dataclasses.asdict(self.arg), f, indent=2)

        self.device = torch.device(self.arg.device)
        print(f'using device: {self.device}')

        self.model.to(self.device)
        self.ema_model = EMAModel(
            self.model.parameters(),
            use_ema_warmup=True,
            inv_gamma=self.arg.ema_inv_gamma,
            power=self.arg.ema_power,
        )
        self.ema_model.to(self.device)
        if self.arg.compile:
            print('compiling model...')
            self.model = torch.compile(self.model)
            print('model compiled')

        self.global_steps = 0
        self.steps_in_epoch = 0
        self.epochs = 0

        self.dataset = self.get_train_dataset()
        self.dataloader = self.get_train_dataloader()
        self.steps_per_epoch = len(self.dataloader)
        print(f'steps per epoch: {self.steps_per_epoch}')

        self.valid_dataset = self.get_valid_dataset()
        self.valid_dataloader = self.get_valid_dataloader()

        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler(self.optimizer)

        self.eval_seed = make_generation_seed(
            self.arg.dataset,
            self.arg.eval_n_examples, 
            seed=self.arg.seed,
            class_condition=self.arg.class_conditioning,
            sample_labels=True,
        )
        print(self.eval_seed['cls'] if 'cls' in self.eval_seed else 'no class conditioning')

        self.fid_seed = None
        self.fid_refence = None
        if self.arg.fid_eval_steps is not None:
            if self.arg.adjust_fid_n:
                assert self.arg.fid_n_examples == max(self.arg.fid_adjust_subsets), \
                    f'fid_n_examples must be equal to max(adjust_subsets) for FID extrapolation'
                self.arg.fid_adjust_subsets.sort()
                
            fid_seed = make_generation_seed(
                self.arg.dataset, 
                self.arg.fid_n_examples, 
                seed=self.arg.seed,
                class_condition=self.arg.class_conditioning,
                sample_labels=self.arg.fid_sample_labels,
            )
            if self.arg.class_conditioning:
                self.fid_seed = TensorDataset(fid_seed['z'], fid_seed['cls'])
            else:
                self.fid_seed = TensorDataset(fid_seed['z'])
            self.fid_refence = load_hidden_parameters(self.arg.fid_reference_dataset, save=False)

        self.ckpt_base_dir = None
        if self.arg.save_steps is not None:
            self.ckpt_base_dir = os.path.join(self.arg.output_dir, 'ckpts')
        self.saved_ckpt_paths = []
    
    def mixed_precision_context(self):
        context = torch.autocast(self.device.type, dtype=torch.bfloat16)
        return context

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

    def get_lr_scheduler(self, optim):
        if self.arg.lr_scheduler is None:
            return None
        lr_scheduler = get_lr_scheduler(
            self.arg.lr_scheduler,
            optimizer=optim,
            num_warmup_steps=self.arg.lr_warmup_steps,
            num_training_steps=self.arg.max_steps,
            **self.arg.lr_scheduler_cfg,
        )
        return lr_scheduler

    def get_train_dataset(self):
        dataset = load_dataset(
            self.arg.dataset,
            self.arg.dataset_dir, 
            train=True,
            augumentations=self.arg.augmentations
        )
        print(f'dataset ready: {dataset}')
        return dataset
    
    def get_valid_dataset(self):
        dataset = load_dataset(
            self.arg.dataset,
            self.arg.dataset_dir, 
            train=False,
        )
        print(f'validation dataset ready: {dataset}')
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
    
    def get_valid_dataloader(self):
        loader = DataLoader(
            self.valid_dataset,
            batch_size=self.arg.generation_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.arg.dataloader_pin_memory,
            num_workers=self.arg.dataloader_num_workers,
        )
        print(f'validation dataloader ready')
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

        def mixed_state_dict(fp32_state_dict):
            mixed_state_dict = {}
            fp32_keywords = ['norm', 'class_embedding'] 
            for key, tensor in fp32_state_dict.items():
                if any(keyword in key.lower() for keyword in fp32_keywords):
                    mixed_state_dict[key] = tensor.cpu().to(torch.float32)
                else:
                    mixed_state_dict[key] = tensor.cpu().to(torch.bfloat16)
            return mixed_state_dict
        
        # save only ema model
        self.ema_model.store(self.model.parameters())
        self.ema_model.copy_to(self.model.parameters())
        state_dict = mixed_state_dict(self.model.state_dict())
        if self.arg.compile:
            state_dict = fix_state_dict(state_dict)
        torch.save(state_dict, os.path.join(ckpt_dir, 'model.pt'))
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
            'ema_model_state_dict': self.ema_model.state_dict(),
        }
        if self.lr_scheduler is not None:
            ckpt['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(ckpt, latest_ckpt_path)
    
    def load_latest_ckpt(self, ckpt_dir):
        latest_ckpt_path = os.path.join(ckpt_dir, 'latest.pt')
        ckpt = torch.load(latest_ckpt_path, weights_only=False)
        print(ckpt.keys())

        self.global_steps = ckpt['global_steps']
        self.epochs = ckpt['epochs']
        self.steps_in_epoch = ckpt['steps_in_epoch']

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.ema_model.load_state_dict(ckpt['ema_model_state_dict'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])

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
        uncond_mask = None
        if label is not None:
            uncond_mask = torch.bernoulli(self.arg.p_uncond * torch.ones_like(label))

        with self.mixed_precision_context():
            loss = self.scheduler.get_loss(x, self.model, cls=label, uncond_mask=uncond_mask)

        loss.backward()
        if self.arg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.arg.clip_grad_norm
            )
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.model.zero_grad()
        self.ema_model.step(self.model.parameters())

        return loss.item()
    
    @torch.no_grad()
    def generate_eval_examples(self):
        self.model.eval()

        z = self.eval_seed['z'].to(self.device)
        cond = self.eval_seed['cls'].to(self.device) if 'cls' in self.eval_seed else None
        with self.mixed_precision_context():
            samples = self.sampler.sample(z, self.model, cond=cond)
        samples = torch.clip(samples, -1, 1).cpu()
        ret = {'examples': samples}
        
        self.ema_model.store(self.model.parameters())
        self.ema_model.copy_to(self.model.parameters())

        with self.mixed_precision_context():
            ema_samples = self.sampler.sample(z, self.model, cond=cond)
        ema_samples = torch.clip(ema_samples, -1, 1).cpu()
        ret['ema_examples'] = ema_samples

        self.ema_model.restore(self.model.parameters())

        return ret
    
    @torch.no_grad()
    def generate_fid_samples(self):
        self.model.eval()
        
        dataloader = DataLoader(
            self.fid_seed, 
            num_workers=1, 
            batch_size=self.arg.generation_batch_size, 
            shuffle=False, 
            drop_last=False
        )
        generated = []

        self.ema_model.store(self.model.parameters())
        self.ema_model.copy_to(self.model.parameters())

        for batch in tqdm.tqdm(dataloader, leave=False, desc='generating fid samples'):
            z = batch[0].to(self.device)
            cond = batch[1].to(self.device) if len(batch) > 1 else None
            
            with self.mixed_precision_context():
                samples = self.sampler.sample(z, self.model, cond=cond)

            samples = torch.clip(samples, -1, 1).cpu()
            generated.append(samples)
        
        self.ema_model.restore(self.model.parameters())
        
        generated = torch.cat(generated, dim=0)
        return generated

    @torch.no_grad()
    def evaluate_fid(self):
        # torch.cuda.empty_cache()
        generated = self.generate_fid_samples()
        torch.cuda.empty_cache()
        inception_features = calc_inception_features(
            generated,
            batch_size=self.arg.inception_batch_size,
            device=self.device,
        )

        ret = {}
        if self.arg.adjust_fid_n:
            fid_result = fid_extrapolation(
                inception_features,
                ref_mu=self.fid_refence[0],
                ref_sigma=self.fid_refence[1],
                subset_sizes=self.arg.fid_adjust_subsets,
                target_n=50_000,
            )
            ret['FID'] = fid_result['fids'][-1]
            ret['FID@inf'] = fid_result['fid_infinity']
            ret['FID@50k'] = fid_result['fid_target']
        else:
            mu, sigma = inception_features_to_hidden_parameters(inception_features)
            fid = calculate_frechet_distance(mu, sigma, self.fid_refence[0], self.fid_refence[1])
            ret['FID'] = fid.item()
        
        torch.cuda.empty_cache()
        print(f'evaluated FID: {ret["FID"]:.4f}')

        with open(os.path.join(self.arg.output_dir, 'fid_evaluations.jsonl'), 'a') as f:
            ret_with_steps = dict(steps=self.global_steps, **ret)
            f.write(json.dumps(ret_with_steps) + '\n')

        if self.wandb_run is not None:
            self.wandb_run.log(ret, step=self.global_steps)
    
    @torch.no_grad()
    def evaluate_validation_loss(self):
        self.model.eval()

        def get_eval_loss():
            loss_sum = 0
            gen = torch.Generator(device=self.device).manual_seed(self.arg.seed)
            for batch in tqdm.tqdm(self.valid_dataloader, leave=False, desc='evaluating validation loss'):
                x = batch['image'].to(self.device)
                cls = batch['label'].to(self.device) if 'label' in batch else None
                batch_size = x.size(0)
                
                with self.mixed_precision_context():
                    loss = self.scheduler.get_loss(x, self.model, gen=gen, cls=cls)
                    
                loss_sum += loss.item() * batch_size
            mean_loss = loss_sum / len(self.valid_dataset)
            return mean_loss
        
        result = {'loss': get_eval_loss()}

        self.ema_model.store(self.model.parameters())
        self.ema_model.copy_to(self.model.parameters())
        result['ema_loss'] = get_eval_loss()
        self.ema_model.restore(self.model.parameters())

        return result


    def evaluate(self, steps):
        # torch.cuda.empty_cache()

        save_dir = os.path.join(self.arg.output_dir, f'examples')
        os.makedirs(save_dir, exist_ok=True)

        def save_samples(samples, name):
            grid_image = TF.to_pil_image(
                make_grid(samples, nrow=10, normalize=True, value_range=(-1, 1)))
            save_path = os.path.join(save_dir, f'{steps:06d}_{name}.png')
            grid_image.save(save_path)
            return grid_image

        examples = self.generate_eval_examples()
        image_log = {}
        for k, v in examples.items():
            img = save_samples(v, k)
            image_log[k] = wandb.Image(img)
        loss_log = {f'val/{k}': v for k, v in self.evaluate_validation_loss().items()}

        if self.wandb_run is not None:
            logs = {
                'global_steps': steps,
                'epochs': steps / self.steps_per_epoch,
            }
            logs.update(**image_log)
            logs.update(**loss_log)
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

        self.model.train()

        with tqdm.tqdm(initial=self.global_steps, total=self.arg.max_steps) as pbar:
            while self.global_steps < self.arg.max_steps:
                if iterator is None:
                    iterator = iter(self.dataloader)
                    
                for batch in iterator:
                    x = batch['image'].to(self.device)
                    cls = batch['label'].to(self.device) if 'label' in batch else None
                    loss = self.train_on_batch(x, cls)

                    self.global_steps += 1
                    self.epochs = self.global_steps // self.steps_per_epoch
                    self.steps_in_epoch = self.global_steps % self.steps_per_epoch
                    pbar.update(1)
                    
                    if self.global_steps % self.arg.logging_steps == 0:
                        pbar.set_postfix({'loss': f'{loss:.5f}'})

                        logs = {'loss': loss}
                        logs['ema_decay'] = self.ema_model.cur_decay_value
                        if self.lr_scheduler is not None:
                            logs['lr'] = self.lr_scheduler.get_last_lr()[0]
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
                    
                    if self.arg.fid_eval_steps is not None and self.global_steps % self.arg.fid_eval_steps == 0:
                        self.evaluate_fid()
                        self.model.train()

                    if self.global_steps >= self.arg.max_steps:
                        break

                iterator = None

        print('train done')



def main(
        train_arg_json: Optional[str] = None,
        resume_ckpt_dir: Optional[str] = None,
        overwrite: bool = False,
):
    assert train_arg_json is not None or resume_ckpt_dir is not None
    if resume_ckpt_dir is not None:
        print(f'loading train args from resume ckpt dir "{resume_ckpt_dir}"')
        train_arg_json = os.path.join(resume_ckpt_dir, 'train_args.json')

    print(train_arg_json)
    with open(train_arg_json, 'r') as f:
        args_dict = json.load(f)
        if 'lr_schduler_cfg' in args_dict:
            print('fixing typo in train args: lr_schduler_cfg -> lr_scheduler_cfg')
            args_dict['lr_scheduler_cfg'] = args_dict.pop('lr_schduler_cfg')
        train_args = TrainArgs(**args_dict)

    seed_everything(train_args.seed)
    
    model = JITTransformer(**train_args.model_cfg)
    scheduler = JITScheduler(**train_args.scheduler_cfg)
    sampler = JITSampler(scheduler=scheduler, **train_args.sampler_cfg)

    assert scheduler.pred_type == sampler.pred_type, \
        f'scheduler pred_type "{scheduler.pred_type}" does not match sampler pred_type "{sampler.pred_type}"'

    print(f'params: {count_parameters(model) / 1e+6}')

    trainer = Trainer(
        train_args, 
        model, 
        scheduler, 
        sampler, 
        resume_ckpt_dir=resume_ckpt_dir, 
        overwrite=overwrite
    )
    trainer.train()


# nohup python train.py arg_json/rf_cifar_attn.json > stdouts/rf_cifar_attn &
if __name__ == '__main__':
    fire.Fire(main)

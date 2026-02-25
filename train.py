
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
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
from utils.fid import load_hidden_parameters, calculate_frechet_distance, \
    calc_inception_features, inception_features_to_hidden_parameters
from utils.fid_infinity import fid_extrapolation
from utils.validate_memo import calc_memorization_metric


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

    fid_eval_steps: Optional[int] = None
    fid_ema: bool = True
    fid_reference_dataset: str = 'mnist-train'
    fid_n_examples: int = 10000
    generation_batch_size: int = 256
    inception_batch_size: int = 512
    adjust_fid_n: bool = True
    fid_adjust_subsets: List[int] = dataclasses.field(
        default_factory=lambda: [4000, 6000, 8000, 10000])

    batch_size: int = 64
    lr: float = 2e-4
    # lr_scheduler: Optional[str] = None
    # lr_warmup_steps: int = 0
    optimizer: str = 'adamw'
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    clip_grad_norm: float = 1.0

    use_ema: bool = True
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
        dataset = CIFAR10(
            data_dir,
            transform=transform,
            download=True,
            train=train
        )
        data_tensor = torch.from_numpy(dataset.data).to(torch.float32).permute(0, 3, 1, 2)
        data_tensor = data_tensor / 127.5 - 1  # scale to [-1, 1]
    elif dataset == 'mnist':
        dataset = MNIST(
            data_dir,
            transform=transform,
            download=True,
            train=train
        )
        data_tensor = dataset.data.unsqueeze(1).to(torch.float32)  # (N, 1, H, W)
        data_tensor = data_tensor / 127.5 - 1  # scale to [-1, 1]
    else:
        raise ValueError(f'unknown dataset {dataset}')
    
    return dataset, data_tensor


def make_generation_seed(dataset, n_examples, seed=None, sample_labels=False):
    def get_generator():
        return torch.Generator().manual_seed(seed) if seed is not None else None
    
    if dataset == 'mnist':
        image_shape = (1, 28, 28)
        n_classes = 10
    elif dataset == 'cifar10':
        image_shape = (3, 32, 32)
        n_classes = 10
    else:
        raise ValueError(f'unknown dataset {dataset}')
    
    z = torch.randn((n_examples, *image_shape), generator=get_generator())
    if sample_labels:
        cls = torch.randint(0, n_classes, (n_examples,), generator=get_generator())
    else:
        cls = torch.arange(n_examples) % n_classes

    return {'z': z, 'cls': cls}


class Trainer:
    def __init__(
            self,
            arg: TrainArgs,
            model: nn.Module,
            scheduler: Type[BaseScheduler],
            sampler: Optional[Type[BaseSampler]] = None,
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

        self.dataset, self.train_data_tensor = self.get_train_dataset()
        self.dataloader = self.get_train_dataloader()
        self.steps_per_epoch = len(self.dataloader)
        print(f'steps per epoch: {self.steps_per_epoch}')

        self.valid_dataset = self.get_valid_dataset()
        self.valid_dataloader = self.get_valid_dataloader()

        self.optimizer = self.get_optimizer()
        self.eval_seed = make_generation_seed(
            self.arg.dataset,
            self.arg.eval_n_examples, 
            seed=self.arg.seed,
            sample_labels=False,
        )

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
                sample_labels=True,
            )
            self.fid_seed = TensorDataset(fid_seed['z'], fid_seed['cls'])
            self.fid_refence = load_hidden_parameters(self.arg.fid_reference_dataset, save=False)


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
        dataset, data_tensor = load_dataset(
            self.arg.dataset,
            self.arg.dataset_dir, 
            train=True,
            augumentations=self.arg.augmentations
        )
        print(f'dataset ready: {dataset}')
        return dataset, data_tensor
    
    def get_valid_dataset(self):
        dataset, _ = load_dataset(
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
            batch_size=self.arg.batch_size,
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
    
    @torch.no_grad()
    def generate_eval_examples(self):
        self.model.eval()

        z, cls = self.eval_seed['z'].to(self.device), self.eval_seed['cls'].to(self.device)
        pred_fn = self.model.get_pred_fn(cond=cls, guidance_scale=self.arg.guidance_scale)
        samples = self.sampler.sample(z, self.scheduler, pred_fn)
        samples = torch.clip(samples, -1, 1).cpu()
        ret = {'examples': samples}
        
        if self.ema_model is not None:
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())

            ema_samples = self.sampler.sample(z, self.scheduler, pred_fn)
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

        if self.arg.fid_ema and self.ema_model is not None:
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())

        for z, cls in tqdm.tqdm(dataloader, leave=False, desc='generating fid samples'):
            z, cls = z.to(self.device), cls.to(self.device)
            pred_fn = self.model.get_pred_fn(cond=cls, guidance_scale=self.arg.guidance_scale)
            samples = self.sampler.sample(z, self.scheduler, pred_fn)
            samples = torch.clip(samples, -1, 1).cpu()
            generated.append(samples)
        
        if self.arg.fid_ema and self.ema_model is not None:
            self.ema_model.restore(self.model.parameters())
        
        generated = torch.cat(generated, dim=0)
        return generated

    def evaluate_fid(self):
        generated = self.generate_fid_samples()
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

        mem_ratio, *_ = calc_memorization_metric(
            generated,
            self.train_data_tensor,
            device=self.device,
        )
        ret['memorization_ratio'] = mem_ratio.item()
        print(f'evaluated FID: {ret["FID"]:.4f}, memorization_ratio: {ret["memorization_ratio"]:.4f}')

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
            for x, cls in tqdm.tqdm(self.valid_dataloader, leave=False, desc='evaluating validation loss'):
                batch_size = x.size(0)
                x, cls = x.to(self.device), cls.to(self.device)
                loss = self.scheduler.get_loss(x, self.model, cls=cls)
                loss_sum += loss.item() * batch_size
            mean_loss = loss_sum / len(self.valid_dataset)
            return mean_loss
        
        result = {'loss': get_eval_loss()}
        if self.ema_model is not None:
            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())
            result['ema_loss'] = get_eval_loss()
            self.ema_model.restore(self.model.parameters())

        return result


    def evaluate(self, steps):
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
        train_args = TrainArgs(**json.load(f))

    seed_everything(train_args.seed)
    
    model = get_model(train_args.model_type, **train_args.model_cfg)
    scheduler = get_scheduler(train_args.scheduler_type, **train_args.scheduler_cfg)
    sampler = get_sampler(train_args.sampler_type, **train_args.sampler_cfg)

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


if __name__ == '__main__':
    fire.Fire(main)

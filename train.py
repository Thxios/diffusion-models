
import os
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random as rd
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import json
import tqdm.auto as tqdm
import dataclasses
from typing import Optional, List, Tuple
from ema_pytorch import EMA
import wandb
import fire

from diffusion.pipeline import BaseDiffusion
from diffusion import sampler
from utils import load_diffusion_pipeline, count_parameters, calc_fid_to_reference



AUG_TYPE_MAPPING = {
    'RandomHorizontalFlip': T.RandomHorizontalFlip,
    'RandomVerticalFlip': T.RandomVerticalFlip,
}
OPTIM_TYPE_MAPPING = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}
N_CLASSES = 10
WANDB_PROJ_NAME = 'diffusion-cifar10'

def get_cifar10_dataset(data_dir, aug: Optional[List[str]] = None):
    transform = []
    if aug is not None:
        for aug_type in aug:
            assert aug_type in AUG_TYPE_MAPPING, f'unknown aug type {aug_type}'
            transform.append(AUG_TYPE_MAPPING[aug_type]())

    transform.extend([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    transform = T.Compose(transform)

    dataset = CIFAR10(data_dir,
                      transform=transform,
                      download=True)
    return dataset


def tensor_to_image_np(img_tensor: torch.Tensor):
    img_tensor = torch.clip(img_tensor, -1, 1).permute(0, 2, 3, 1)
    img_np = img_tensor.cpu().numpy()
    img_np = np.round((img_np + 1) * 127.5).astype(np.uint8)
    return img_np



@dataclasses.dataclass
class TrainArgs:
    output_dir: str
    max_steps: int = 100_000
    logging_steps: int = 50
    eval_steps: int = 1000
    save_limit: int = 2
    ema_beta: float = 0.999
    eval_n_samples: int = 10
    eval_sample_steps: int = 10
    eval_ema_only: bool = True
    batch_size: int = 64
    lr: float = 2e-4
    optim: str = 'adamw'
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    clip_grad_norm: float = 1.
    augmentations: List[str] = \
        dataclasses.field(default_factory=lambda: ['RandomHorizontalFlip'])
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 2
    dataset_dir: str = 'datasets'
    device: str = 'cuda'
    seed: int = 42
    worker_seed: int = 42
    dpm_cfg: Optional[dict] = None
    dpm_discrete: bool = False
    fid_eval: bool = True
    fid_ema: bool = True
    fid_eval_steps: int = 10000
    fid_n_samples: int = 1000
    fid_sample_steps: int = 10
    fid_seed: int = 42


def seed_everything(seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        rd.seed(seed)
    # torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    rd.seed(worker_seed)


class Trainer:
    def __init__(
            self,
            model: BaseDiffusion,
            arg: TrainArgs
    ):
        assert arg.eval_steps % arg.logging_steps == 0
        assert not arg.fid_eval or arg.fid_eval_steps % arg.logging_steps == 0
        if arg.dataloader_num_workers == 0:
            warnings.warn(f'set dataloader_num_workers > 0 for reproducibility')

        self.model = model
        self.arg = arg

        print(f'Trainer arg:')
        print(self.arg)

        os.makedirs(self.arg.output_dir, exist_ok=True)
        with open(os.path.join(self.arg.output_dir, 'train_args.json'), 'w') as f:
            json.dump(dataclasses.asdict(self.arg), f, indent=2)

        self.device = 'cuda' \
            if self.arg.device == 'cuda' and torch.cuda.is_available() \
            else 'cpu'

        self.global_steps = 0
        self.steps_in_epoch = 0
        self.epochs = 0
        self.trained = False

        self.ema_model = EMA(self.model, beta=self.arg.ema_beta)
        self.dataset = self.get_train_dataset()
        self.loader = self.get_train_dataloader()
        self.steps_per_epoch = len(self.loader)
        self.optim = self.get_optimizer()
        self.eval_examples = self.get_eval_examples()
        if self.arg.fid_eval:
            self.fid_seed = self.get_fid_seed()
        else:
            self.fid_seed = None
        if not self.arg.dpm_discrete:
            self.sampler_cls = sampler.ContinuousDPM2Solver
        else:
            self.sampler_cls = sampler.DiscreteDDIMSampler

        self.model.to(self.device)
        self.ema_model.to(self.device)

        self.saved_ckpt_paths = []
        self.wandb_run = None
        self.wandb_run_id = None

    def get_optimizer(self):
        assert self.arg.optim in OPTIM_TYPE_MAPPING, \
            f'unknown optim type {self.arg.optim}'

        optim = OPTIM_TYPE_MAPPING[self.arg.optim](
            self.model.parameters(),
            lr=self.arg.lr,
            betas=self.arg.adam_betas
        )
        print(f'optimizer ready')
        return optim

    def get_train_dataset(self):
        dataset = get_cifar10_dataset(
            self.arg.dataset_dir, aug=self.arg.augmentations)
        print('dataset ready')
        return dataset

    def get_train_dataloader(self, dataset=None):
        if dataset is None:
            dataset = self.dataset

        rng = torch.Generator()
        rng.manual_seed(self.arg.worker_seed)
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

    def get_eval_examples(self):
        return dict(
            z=torch.randn((self.arg.eval_n_samples, 3, 32, 32)),
            cond=torch.tensor(
                [i % N_CLASSES for i in range(self.arg.eval_n_samples)],
                dtype=torch.int64
            )
        )

    def get_fid_seed(self):
        gen = torch.Generator()
        gen.manual_seed(self.arg.fid_seed)
        noise = torch.randn((self.arg.fid_n_samples, 3, 32, 32), generator=gen)
        cls = torch.randint(0, N_CLASSES, size=(self.arg.fid_n_samples,), generator=gen)
        return dict(
            noise=noise,
            cls=cls,
        )

    @torch.no_grad()
    def calc_fid(self):
        loader = DataLoader(TensorDataset(self.fid_seed['noise'], self.fid_seed['cls']),
                            batch_size=self.arg.batch_size,
                            pin_memory=True,
                            drop_last=False)
        model = self.ema_model.ema_model if self.arg.fid_ema else self.model
        model.eval()
        _sampler = self.sampler_cls(self.arg.fid_sample_steps, pbar=False)
        samples = []
        for x, cls in tqdm.tqdm(loader, desc='FID'):
            x, cls = x.to(self.arg.device), cls.to(self.arg.device)
            out = model.sample(x, _sampler, cond=cls)
            out = out.cpu()
            samples.append(out)
        samples = torch.cat(samples, dim=0)
        samples = torch.clip(samples, -1, 1)

        fid = calc_fid_to_reference(
            images=samples,
            reference_file='cifar10-test-inception.npz',
            batch_size=self.arg.batch_size,
            device=self.arg.device,
        )
        return fid


    def prepare_wandb_run(self, run_name, config_dict: Optional[dict] = None):
        if self.wandb_run_id is not None:
            self.wandb_run = wandb.init(
                project=WANDB_PROJ_NAME,
                id=self.wandb_run_id,
                resume='must',
            )
        else:
            cfg = dataclasses.asdict(self.arg)
            if config_dict is not None:
                cfg['model'] = config_dict
            self.wandb_run = wandb.init(
                name=run_name,
                project=WANDB_PROJ_NAME,
                config=cfg
            )
            self.wandb_run_id = self.wandb_run.id
        print(f'Wandb - run name: {self.wandb_run.name}, id: {self.wandb_run_id}')

    def save_ckpt(self, ckpt_dir, ckpt_name):
        ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}.pt')

        state_dict = dict(
            model=self.model.state_dict(),
            ema_model=self.ema_model.ema_model.state_dict(),
            global_steps=self.global_steps,
        )
        torch.save(state_dict, ckpt_path)
        # print(f'saved ckpt {ckpt_path}')

        if len(self.saved_ckpt_paths) >= self.arg.save_limit:
            to_del = self.saved_ckpt_paths.pop(0)
            if os.path.exists(to_del) and to_del != ckpt_path:
                os.remove(to_del)
                # print(f'deleted ckpt {to_del}')
        if self.global_steps % self.arg.fid_eval_steps != 0:
            self.saved_ckpt_paths.append(ckpt_path)

    def load_latest_ckpt(self, ckpt_dir):
        ckpt_path = os.path.join(ckpt_dir, 'latest.pt')
        ckpt = torch.load(ckpt_path)

        self.global_steps = ckpt['global_steps']
        self.epochs = ckpt['epochs']
        self.steps_in_epoch = ckpt['steps_in_epoch']

        self.model.load_state_dict(ckpt['model'])
        self.ema_model.load_state_dict(ckpt['ema_model'])
        self.optim.load_state_dict(ckpt['optim'])
        self.eval_examples = ckpt['eval_examples']
        self.wandb_run_id = ckpt.get('wandb_run_id', None)

        torch.set_rng_state(ckpt['rng_state'])
        torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
        np.random.set_state(ckpt['np_rng_state'])
        rd.setstate(ckpt['rd_rng_state'])

        print(f'loaded ckpt {ckpt_path}')

    def save_latest_ckpt(self, ckpt_dir):
        ckpt_path = os.path.join(ckpt_dir, 'latest.pt')

        state_dict = dict(
            global_steps=self.global_steps,
            epochs=self.epochs,
            steps_in_epoch=self.steps_in_epoch,

            model=self.model.state_dict(),
            ema_model=self.ema_model.state_dict(),
            optim=self.optim.state_dict(),
            eval_examples=self.eval_examples,

            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(),
            np_rng_state=np.random.get_state(),
            rd_rng_state=rd.getstate(),
        )
        if self.wandb_run_id is not None:
            state_dict.update(wandb_run_id=self.wandb_run_id)
        torch.save(state_dict, ckpt_path)

    def log(self, steps, **logs):
        # print(f'Step {steps}: loss {logs["loss"]:.5f}')
        if self.wandb_run is not None:
            logs.update(steps=steps)
            self.wandb_run.log(logs)

    @torch.no_grad()
    def sample_examples(self, ema=True):
        model = self.ema_model.ema_model if ema else self.model
        model.eval()

        _sampler = self.sampler_cls(
                n_steps=self.arg.eval_sample_steps, pbar=True)

        example = {k: v.to(self.device) for k, v in self.eval_examples.items()}
        gen = model.sample(
            sampler=_sampler,
            **example
        )
        gen = tensor_to_image_np(gen)
        return gen

    def evaluate(self):
        self.save_ckpt(self.arg.output_dir, f'step_{self.global_steps}')

        if self.wandb_run is not None:
            ema_gen = self.sample_examples()
            logs = {
                "ema_samples": [wandb.Image(ema_gen[i]) for i in range(len(ema_gen))]
            }
            if not self.arg.eval_ema_only:
                gen = self.sample_examples(ema=False)
                logs.update(
                    samples=[wandb.Image(gen[i]) for i in range(len(gen))]
                )
        else:
            logs = dict()

        self.save_latest_ckpt(self.arg.output_dir)
        return logs

    def train_on_batch(self, x, cls):
        loss = self.model.train_step_loss(x, cls)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.arg.clip_grad_norm
        )
        self.optim.step()
        self.model.zero_grad()
        self.ema_model.update()

        return loss.item()

    def skip_previous_steps(self):
        print(f'skipping {self.epochs} epochs {self.steps_in_epoch} steps...')
        for _ in tqdm.tqdm(range(self.epochs), leave=False):
            next(iter(self.loader))
        iterator = iter(self.loader)
        for _ in tqdm.tqdm(range(self.steps_in_epoch), leave=False):
            next(iterator)
        return iterator

    def train(
            self,
            resume_ckpt=None,
            wandb_run_name: Optional[str] = None,
    ):
        assert not self.trained

        if resume_ckpt is not None:
            self.load_latest_ckpt(resume_ckpt)

        if self.global_steps > 0:
            iterator = self.skip_previous_steps()
        else:
            iterator = None

        print('train start')

        losses = []
        self.model.train()
        self.ema_model.train()

        if wandb_run_name is not None or self.wandb_run_id is not None:
            self.prepare_wandb_run(wandb_run_name)

        with tqdm.tqdm(total=self.arg.max_steps - self.global_steps) as pbar:
            while self.global_steps < self.arg.max_steps:
                if iterator is None:
                    iterator = iter(self.loader)
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

                        if self.global_steps % self.arg.eval_steps == 0:
                            logs.update(self.evaluate())
                            self.model.train()
                            self.ema_model.train()

                        if self.global_steps % self.arg.fid_eval_steps == 0:
                            fid = self.calc_fid()
                            logs.update(fid=fid)
                            self.model.train()
                            self.ema_model.train()

                        self.log(
                            self.global_steps,
                            **logs
                        )

                        if self.global_steps >= self.arg.max_steps:
                            break
                iterator = None

        self.trained = True
        print('train done')



def run(
        train_arg_json,
        dpm_cfg_json: Optional[str] = None,
        resume_ckpt: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
):
    print(train_arg_json)
    with open(train_arg_json, 'r') as f:
        train_args = TrainArgs(**json.load(f))

    if train_args.dpm_cfg is not None:
        pass
    elif dpm_cfg_json is not None:
        with open(dpm_cfg_json, 'r') as f:
            train_args.dpm_cfg = json.load(f)
    else:
        raise ValueError(
            'at least one of train_args.dpm_cfg or dpm_config_json must be provided')

    seed_everything(train_args.seed)
    model = load_diffusion_pipeline(train_args.dpm_cfg)
    print(f'params: {count_parameters(model) / 1e+6}')

    trainer = Trainer(model, train_args)
    trainer.train(
        resume_ckpt=resume_ckpt,
        wandb_run_name=wandb_run_name,
    )


if __name__ == '__main__':
    # main()
    fire.Fire(run)

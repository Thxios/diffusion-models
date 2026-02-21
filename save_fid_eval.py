
import os
import random as rd
import numpy as np
import json
from typing import List, Optional
from tqdm.auto import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR10, MNIST
import fire

from modeling import get_model
from diffusion.scheduler import get_scheduler
from diffusion.sampler import get_sampler
from utils import count_parameters
from utils.fid import load_hidden_parameters, calculate_frechet_distance, \
    calc_inception_features, inception_features_to_hidden_parameters
from utils.fid_infinity import fid_extrapolation
from utils.validate_memo import calc_memorization_metric



def seed_everything(seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        rd.seed(seed)


def load_dataset_tensor(dataset, data_dir, train=True):
    if dataset == 'cifar10':
        dataset = CIFAR10(
            data_dir,
            download=True,
            train=train
        )
        data_tensor = torch.from_numpy(dataset.data).to(torch.float32).permute(0, 3, 1, 2)
        data_tensor = data_tensor / 127.5 - 1  # scale to [-1, 1]
    elif dataset == 'mnist':
        dataset = MNIST(
            data_dir,
            download=True,
            train=train
        )
        data_tensor = dataset.data.unsqueeze(1).to(torch.float32)  # (N, 1, H, W)
        data_tensor = data_tensor / 127.5 - 1  # scale to [-1, 1]
    else:
        raise ValueError(f'unknown dataset {dataset}')
    
    return data_tensor


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



def main(
        ckpt_dir: str,
        ckpt_name: str,
        output_json_path: str,
        use_ema: bool = True,
        guidance_scale: float = 1.0,
        sampling_steps: int = 50,
        fid_reference_dataset: str = 'cifar10-train',
        fid_n_examples: int = 10000,
        generation_batch_size: int = 256,
        inception_batch_size: int = 512,
        adjust_fid_n: bool = True,
        fid_adjust_subsets: List[int] = [4000, 6000, 8000, 10000],
        device: str = "cuda:0",
        seed: int = 42,
):
    # 0. Validate args
    assert max(fid_adjust_subsets) == fid_n_examples, \
        "Maximum subset size must be equal to fid_n_examples"
    fid_adjust_subsets.sort()
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    seed_everything(seed)

    settings = {
        'ckpt_dir': ckpt_dir,
        'ckpt_name': ckpt_name,
        'use_ema': use_ema,
        'guidance_scale': guidance_scale,
        'sampling_steps': sampling_steps,
        'fid_reference_dataset': fid_reference_dataset,
        'fid_n_examples': fid_n_examples,
        'generation_batch_size': generation_batch_size,
        'inception_batch_size': inception_batch_size,
        'adjust_fid_n': adjust_fid_n,
        'fid_adjust_subsets': fid_adjust_subsets,
        'seed': seed,
    }
    print(f"Evaluating FID for checkpoint {ckpt_name} in {ckpt_dir} with settings:")
    for k, v in settings.items():
        print(f"  {k}: {v}")
    print()

    # 1. Load model
    config_json_path = os.path.join(ckpt_dir, "train_args.json")
    with open(config_json_path, "r") as f:
        config = json.load(f)
    
    model = get_model(config['model_type'], **config['model_cfg'])
    scheduler = get_scheduler(config['scheduler_type'], **config['scheduler_cfg'])
    sampler_cfg = config['sampler_cfg']
    sampler_cfg['n_steps'] = sampling_steps
    sampler = get_sampler(config['sampler_type'], **sampler_cfg)
    print(f"loaded model: {count_parameters(model) / 1e6:.2f}M parameters")

    ckpt_file = 'ema_model.pt' if use_ema else 'model.pt'
    ckpt_path = os.path.join(ckpt_dir, 'ckpts', ckpt_name, ckpt_file)
    load_result = model.load_state_dict(torch.load(ckpt_path))
    print(load_result)
    model.to(device)
    model.eval()

    # 2. Prepare dataset
    dataset = config['dataset']
    fid_seed = make_generation_seed(dataset, fid_n_examples, seed=seed, sample_labels=True)
    fid_seed = TensorDataset(fid_seed['z'], fid_seed['cls'])
    data_tensor = load_dataset_tensor(dataset, config['dataset_dir'], train=True)
    fid_refence = load_hidden_parameters(fid_reference_dataset, save=False)

    print(f'fid_seed: {fid_seed}')
    print(f'data_tensor: {data_tensor.shape}, in [{data_tensor.min():.4f}, {data_tensor.max():.4f}]')

    dataloader = DataLoader(
        fid_seed,
        batch_size=generation_batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    # 3. Generate samples
    generated = []
    with torch.no_grad():
        for z, cls in tqdm(dataloader, desc="Generating FID samples"):
            z, cls = z.to(device), cls.to(device)
            pred_fn = model.get_pred_fn(cond=cls, guidance_scale=guidance_scale)
            samples = sampler.sample(z, scheduler, pred_fn)
            samples = torch.clip(samples, -1, 1).cpu()
            generated.append(samples)
    generated = torch.cat(generated, dim=0)
    print(f'generated: {generated.shape}')

    # 4. Compute FID
    inception_features = calc_inception_features(
        generated,
        batch_size=inception_batch_size,
        device=device,
        pbar=True,
        pbar_kwargs={'leave': True},
    )
    print(f'inception_features: {inception_features.shape}')

    result = {}
    if adjust_fid_n:
        fid_result = fid_extrapolation(
            inception_features,
            ref_mu=fid_refence[0],
            ref_sigma=fid_refence[1],
            subset_sizes=fid_adjust_subsets,
            target_n=50_000,
            pbar=True,
            pbar_kwargs={'leave': True},
        )
        result['FID'] = fid_result['fids'][-1]
        result['FID@inf'] = fid_result['fid_infinity']
        result['FID@50k'] = fid_result['fid_target']
    else:
        mu, sigma = inception_features_to_hidden_parameters(inception_features)
        fid = calculate_frechet_distance(mu, sigma, fid_refence[0], fid_refence[1])
        result['FID'] = fid.item()

    # 5. Compute memorization metric
    mem_ratio, *_ = calc_memorization_metric(
        generated,
        data_tensor,
        device=device,
    )
    result['memorization_ratio'] = mem_ratio.item()

    # 6. Save results
    print(f'evaluated FID: {result["FID"]:.4f}, memorization_ratio: {result["memorization_ratio"]:.4f}')
    with open(output_json_path, "w") as f:
        json.dump({**settings, **result}, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)

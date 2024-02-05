
import os
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import fire
import tqdm.auto as tqdm
from typing import Optional, List, Iterable

from diffusion.pipeline import BaseDiffusion
from diffusion.sampler import BaseSampler
import diffusion.sampler as smp
from utils import load_diffusion_pipeline, calc_fid_to_reference
from utils.fid_calc import calc_fid_to_reference_w_param, calc_hidden_parameters



def get_sample_seed(n_samples, seed=None):
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    else:
        gen = None

    noise = torch.randn(n_samples, 3, 32, 32, generator=gen)
    cls = torch.randint(0, 10, size=(n_samples,), generator=gen)
    return noise, cls


@torch.no_grad()
def generate_samples(
        model: BaseDiffusion,
        gen_seed_ds: TensorDataset,
        sampler: BaseSampler,
        guidance_scale=0.,
        batch_size=128,
        device='cpu',
):
    loader = DataLoader(
        gen_seed_ds,
        batch_size=batch_size,
        drop_last=False
    )

    model.eval()
    samples = []
    for noise, cls in tqdm.tqdm(loader, desc='sampling'):
        noise, cls = noise.to(device), cls.to(device)
        sample = model.sample(
            noise,
            sampler,
            cond=cls,
            guidance_scale=guidance_scale
        )
        sample = sample.cpu()
        samples.append(sample)

    samples = torch.cat(samples, dim=0)
    samples = torch.clip(samples, -1, 1)
    return samples


def main(
        ckpt_dir: str,
        ckpt_name: str,
        cfg_name: str = 'train_args.json',
        fid_save_path: str = 'fid.json',
        save_overwrite: bool = False,
        n_samples: int = 50000,
        ema: bool = True,
        sampler: str = 'ContinuousDPM2Solver',
        sample_steps: int = 20,
        guidance_scale: float = 0.5,
        batch_size: int = 128,
        seed: int = 42,
        device: str = 'cuda',
        reference: Iterable[str] = ('cifar10-test-inception.npz', 'cifar10-train-inception.npz'),
):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    cfg_path = os.path.join(ckpt_dir, cfg_name)

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    fid_cfg_dict = dict(
        ckpt=ckpt_path,
        ema=ema,
        n_samples=n_samples,
        sampler=sampler,
        sample_steps=sample_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    for k, v in fid_cfg_dict.items():
        print(f'{k}: {v}')

    model = load_diffusion_pipeline(cfg['dpm_cfg'])
    ckpt = torch.load(ckpt_path)
    if ema:
        model.load_state_dict(ckpt['ema_model'])
    else:
        model.load_state_dict(ckpt['model'])
    del ckpt

    sampler_cls = getattr(smp, sampler)
    sampler_ = sampler_cls(sample_steps, pbar=False)

    noise, cls = get_sample_seed(n_samples, seed=seed)
    gen_seed_ds = TensorDataset(noise, cls)

    print(f'generating {n_samples} samples')
    model.to(device)
    generated = generate_samples(
        model,
        gen_seed_ds,
        sampler_,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        device=device
    )
    print(f'generated samples: {generated.shape}')
    np.save(f'gen_n{n_samples}_w{guidance_scale}_s{sample_steps}.npy', generated.numpy())
    del model
    torch.cuda.empty_cache()

    print(f'evaluating FID')
    mu, sigma = calc_hidden_parameters(generated, batch_size=batch_size, device=device)
    fid0 = None
    for ref in reference:
        fid = calc_fid_to_reference_w_param(mu, sigma, ref)
        if fid0 is None:
            fid0 = fid
        print(f'FID - {ref}: {fid:.8f}')

    out_dir = os.path.dirname(fid_save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(fid_save_path):
        print(f'{fid_save_path} already exists - ', end='')
        if not save_overwrite:
            base, ext = os.path.splitext(fid_save_path)
            idx = 1
            while os.path.exists(f'{base}_{idx}{ext}'):
                idx += 1
            fid_save_path = f'{base}_{idx}{ext}'
            print(f'saving in {fid_save_path}')
        else:
            print('overwrite')
    with open(fid_save_path, 'w') as f:
        json.dump(
            {
                'fid': fid0,
                'config': fid_cfg_dict,
                'train_cfg': cfg,
            },
            f,
            indent=2
        )



if __name__ == '__main__':
    fire.Fire(main)


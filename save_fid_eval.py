
import os
import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

from modeling import get_model
from diffusion.sampler import get_sampler
from diffusion.scheduler import get_scheduler
from utils.fid import load_hidden_parameters, calc_hidden_parameters, calculate_frechet_distance



device = torch.device('cuda:2')
print(f'using device: {device}')

ref_dataset = 'cifar10-test'
ref_mu, ref_sigma = load_hidden_parameters(ref_dataset)
image_shape = (3, 32, 32)
num_classes = 10
batch_size = 256        # doubles when using guidance scale not in [0, 1]
inception_batch_size = 512
num_examples = 10000
seed = 42

ckpt_dir = 'outputs/rf_cifar10_base'
print(f'loading model from {ckpt_dir}')
with open(os.path.join(ckpt_dir, 'train_args.json'), 'r') as f:
    train_args = json.load(f)

scheduler = get_scheduler(train_args['scheduler_type'], **train_args['scheduler_cfg'])

def load_model(ckpt_name, ema=True):
    ckpt_file = 'ema_model.pt' if ema else 'model.pt'
    ckpt_path = os.path.join(train_args['output_dir'], 'ckpts', ckpt_name, ckpt_file)

    model = get_model(train_args['model_type'], **train_args['model_cfg'])
    model.load_state_dict(torch.load(ckpt_path))
    return model

def load_sampler(**kwargs):
    sampler_cfg = train_args['sampler_cfg'].copy()
    sampler_cfg.update(kwargs)
    sampler = get_sampler(train_args['sampler_type'], **sampler_cfg)
    return sampler

def generate_noise(num_examples, image_shape, num_classes, seed):
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn((num_examples, *image_shape), generator=gen)
    labels = torch.randint(0, num_classes, (num_examples,), generator=gen)
    print(f'noise: {noise.shape}, labels: {labels.shape}')
    return noise, labels

noise_dataset = TensorDataset(*generate_noise(num_examples, image_shape, num_classes, seed))

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
            eps_pred = model(z, t)
        return eps_pred
    return pred_fn


@torch.no_grad()
def generate_examples(
        model_ckpt_name,
        model_ema=True,
        n_sampling_steps=None,
        guidance_scale=1.0,
):
    model = load_model(model_ckpt_name, ema=model_ema)
    model.to(device)
    model.eval()
    
    sampler_cfg = {'pbar': True, 'pbar_kwargs': {'position': 2, 'leave': False, 'desc': 'sampling'}}
    if n_sampling_steps is not None:
        sampler_cfg['n_steps'] = n_sampling_steps
    sampler = load_sampler(**sampler_cfg)

    dataloader = DataLoader(noise_dataset, num_workers=1, batch_size=batch_size, shuffle=False, drop_last=False)
    generated = []
    for z, cls in tqdm(dataloader, position=1, leave=False, desc='generating'):
        z = z.to(device)
        cls = cls.to(device)

        pred_fn = get_eps_pred_func(model, cls=cls, guidance_scale=guidance_scale)
        gen = sampler.sample(z, scheduler, pred_fn).cpu()
        generated.append(gen)
    generated = torch.cat(generated, dim=0)
    return generated


def calc_fid_model(
        model_ckpt_name,
        model_ema=True,
        n_sampling_steps=None,
        guidance_scale=1.0,
): 
    gen = generate_examples(
        model_ckpt_name=model_ckpt_name,
        model_ema=model_ema,
        n_sampling_steps=n_sampling_steps,
        guidance_scale=guidance_scale
    )
    # print('gen shape:', gen.shape)
    mu, sigma = calc_hidden_parameters(
        gen, batch_size=inception_batch_size, device=device,
        pbar=True, pbar_kwargs={'position': 1, 'leave': False, 'desc': 'inception'},
    )
    # print('mu:', mu[:5])
    # print('sigma:', sigma[:5, :5])
    fid = calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma)
    # print(f'FID: {fid:.6f}')
    return fid



output_dir = 'resources'
os.makedirs(output_dir, exist_ok=True)
output_file = 'rf_cifar_base_fid_results.csv'
with open(os.path.join(output_dir, output_file), 'w') as f:
    f.write('ckpt_name,ema,n_examples,seed,n_steps,guidance_scale,fid\n')

ckpt_list = glob.glob(os.path.join(ckpt_dir, 'ckpts', 'ckpt-*'))
ckpt_list.sort()

with tqdm(total=len(ckpt_list)) as master_pbar:
    for ckpt_path in ckpt_list:
        ckpt_name = os.path.basename(ckpt_path)
        master_pbar.set_description(f'CKPT: {ckpt_name}')
        ema = True
        n_steps = 50
        guidance_scale = 1.0
        fid = calc_fid_model(
            model_ckpt_name=ckpt_name,
            model_ema=ema,
            n_sampling_steps=n_steps,
            guidance_scale=guidance_scale,
        )
        with open(os.path.join(output_dir, output_file), 'a') as f:
            f.write(f'{ckpt_name},{ema},{num_examples},{seed},{n_steps},{guidance_scale},{fid:.6f}\n')
        master_pbar.update(1)


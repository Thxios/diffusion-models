
import os
import glob
import gc
import random as rd
import numpy as np
import pandas as pd
import json
from typing import List, Optional
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torchvision.transforms.functional import to_pil_image
import fire
from pytorch_fid.inception import InceptionV3

from modeling.transformer import JITTransformer
from diffusion.scheduler import JITScheduler
from diffusion.sampler import JITSampler
from utils import count_parameters
from utils.fid import load_hidden_parameters, calculate_frechet_distance, \
    calc_inception_features, inception_features_to_hidden_parameters
from utils.fid_infinity import fid_extrapolation



def seed_everything(seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        rd.seed(seed)



class NoiseDataset(Dataset):
    def __init__(
            self, 
            n_samples: int, 
            image_shape,
            class_condition: bool = False, 
            n_classes: Optional[int] = None,
            sample_labels: bool = False,
            seed: Optional[int] = None
    ):
        super().__init__()
        self.n_samples = n_samples
        self.image_shape = image_shape
        self.class_condition = class_condition
        self.n_classes = n_classes
        self.sample_labels = sample_labels

        self.gen = torch.Generator()
        self.seed = seed
    
    def __getitem__(self, idx):
        self.gen.manual_seed(self.seed + idx)
        item = {'z': torch.randn(self.image_shape, generator=self.gen)}
        if self.class_condition:
            assert self.n_classes is not None, "n_classes must be specified for class conditioning"
            if self.sample_labels:
                cls = torch.randint(0, self.n_classes, (1,), generator=self.gen).item()
            else:
                cls = idx % self.n_classes
            item['cls'] = cls
        return item

    def __len__(self):
        return self.n_samples


def pseudo_print(*args, **kwargs):
    pass


def load_fid_reference(npz_filename):
    data = np.load(npz_filename)
    return data['mu'], data['sigma']

def tensor2image(tensor, value_range=(-1, 1)):
    if value_range is not None:
        tensor = (tensor - value_range[0]) / (value_range[1] - value_range[0])
    tensor = tensor.clamp(0, 1)
    image = to_pil_image(tensor)
    return image

# nohup python -u jit_fid.py outputs/jit/imgnet_base ckpt-500000 resources/jit_fid/imgnet_base --guidance_scale 2.5 --cfg_start 0.1 --sampling_steps 50 --ode_solver heun --fid_reference_dataset imagenet-1k-256-train --n_save_images 50 --device cuda:4 --seed 42 > stdouts/fid_jit_imgnet_base &

def main(
        ckpt_dir: str,
        ckpt_name: str,
        output_dir: str,
        guidance_scale: float = 2.5,
        cfg_start: Optional[float] = None,
        sampling_steps: int = 100,
        ode_solver: str = 'euler',
        fid_reference_dataset: str = 'lsun-church_outdoor-train',
        additional_fid_reference_file: Optional[str] = 'resources/VIRTUAL_imagenet256_labeled.npz',
        fid_n_examples: int = 50000,
        batch_size: int = 512,
        adjust_fid_n: bool = True,
        fid_adjust_subsets: List[int] = [10000, 20000, 30000, 40000, 50000],
        n_save_images: Optional[int] = None,
        device: str = "cuda:5",
        compile: bool = True,
        verbose: bool = True,
        seed: int = 42,
):
    # 0. Validate args
    assert max(fid_adjust_subsets) == fid_n_examples, \
        "Maximum subset size must be equal to fid_n_examples"
    fid_adjust_subsets.sort()
    os.makedirs(output_dir, exist_ok=True)
    print_ = print if verbose else pseudo_print
    device = torch.device(device)

    seed_everything(seed)

    settings = {
        'ckpt_dir': ckpt_dir,
        'ckpt_name': ckpt_name,
        'guidance_scale': guidance_scale,
        'cfg_start': cfg_start,
        'sampling_steps': sampling_steps,
        'ode_solver': ode_solver,
        'fid_reference_dataset': fid_reference_dataset,
        'fid_n_examples': fid_n_examples,
        'adjust_fid_n': adjust_fid_n,
        'fid_adjust_subsets': fid_adjust_subsets,
        'seed': seed,
    }
    print_(f"Evaluating FID for checkpoint {ckpt_name} in {ckpt_dir} with settings:")
    for k, v in settings.items():
        print_(f"  {k}: {v}")
    print_()

    # 1. Load model
    config_json_path = os.path.join(ckpt_dir, "train_args.json")
    with open(config_json_path, "r") as f:
        config = json.load(f)
    
    # model = get_model(config['model_type'], **config['model_cfg'])
    # scheduler = get_scheduler(config['scheduler_type'], **config['scheduler_cfg'])
    model = JITTransformer(**config['model_cfg'])
    scheduler = JITScheduler(**config['scheduler_cfg'])
    cfg_interval = None
    if cfg_start is not None:
        cfg_interval = (cfg_start, 1.0)
    sampler = JITSampler(
        sampling_steps, 
        scheduler=scheduler, 
        ode_solver=ode_solver,
        guidance_scale=guidance_scale, 
        cfg_interval=cfg_interval, 
        pbar=True
    )
    print_(f"loaded model: {count_parameters(model) / 1e6:.2f}M parameters")

    ckpt_path = os.path.join(ckpt_dir, 'ckpts', ckpt_name, 'model.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # for k, v in state_dict.items():
    #     print(f'{k}: {v.shape} {v.dtype}')
    # Detect whether checkpoint was saved from a torch.compile'd model
    has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict)
    state_dict_fix = {}
    drop_keys = ['uncond_class']
    for k, v in state_dict.items():
        new_k = k[len('_orig_mod.'):] if has_orig_mod else k
        # Drop legacy PeriodicEncoding.mlp keys (removed from current model)
        if '.mlp.' in new_k and new_k.split('.mlp.')[0].endswith('.0'):
            continue
        if new_k in drop_keys:
            continue
        state_dict_fix[new_k] = v
    load_result = model.load_state_dict(state_dict_fix, strict=False, assign=True)
    del state_dict, state_dict_fix
    print_(load_result)
    print_(f'uncond_class: {getattr(model, "uncond_class", None)}')
    model.to(device)
    model.eval()

    if compile:
        model = torch.compile(model)
        print_("Model compiled.")

    # 2. Prepare dataset
    dataset = config['dataset']
    if dataset == 'lsun-church_outdoor':
        image_shape = (3, 256, 256)
        n_classes = None
    elif dataset == 'imagenet-1k-256':
        image_shape = (3, 256, 256)
        n_classes = 1000
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    class_condition = config['class_conditioning']
    fid_reference = load_hidden_parameters(fid_reference_dataset, save=False)

    n_save_image_remaining = n_save_images if n_save_images is not None else 0
    if n_save_images is not None:
        os.makedirs(os.path.join(output_dir, f'{ckpt_name}_fid_samples'), exist_ok=True)

    fid_references = [fid_reference]
    names = [fid_reference_dataset]
    if additional_fid_reference_file is not None:
        additional_fid_ref = load_fid_reference(additional_fid_reference_file)
        fid_references.append(additional_fid_ref)
        names.append('ADM_Imagenet256')

    print(f'dataset: {dataset}')
    print(f'class_condition: {class_condition}')
    print(f'image_shape: {image_shape}, n_classes: {n_classes}')
    print(f'fid_reference: {fid_reference_dataset}')
    print(f'additional_fid_reference: {additional_fid_reference_file}')

    fid_seed = NoiseDataset(
        n_samples=fid_n_examples,
        image_shape=image_shape,
        class_condition=class_condition,
        n_classes=n_classes,
        seed=seed,
    )
    dataloader = DataLoader(
        fid_seed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    # 3. prepare inception
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=False)
    inception.to(device)
    inception.eval()
    print("Inception model loaded.")

    # 4. Generate samples & compute inception features
    inception_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating FID samples"):
            z = batch['z'].to(device)
            cond = batch['cls'].to(device) if class_condition else None

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                samples = sampler.sample(z, model, cond=cond)

            samples = torch.clip(samples, -1, 1)

            n_images_to_save = min(n_save_image_remaining, samples.size(0))
            if n_images_to_save > 0:
                for i in range(n_images_to_save):
                    img = tensor2image(samples[i].cpu())
                    save_name = f'sample_{len(inception_features) + i:05d}'
                    if class_condition:
                        save_name = save_name + f'_cls{cond[i].item()}'
                    img.save(os.path.join(output_dir, f'{ckpt_name}_fid_samples', f'{save_name}.png'))
                print_(f'Saved {n_images_to_save} sample images to {os.path.join(output_dir, f"{ckpt_name}_fid_samples")}')
                n_save_image_remaining -= n_images_to_save
            
            out = inception(samples)[0].squeeze().cpu()
            inception_features.append(out)

    inception_features = torch.cat(inception_features, dim=0).numpy()
    del model, scheduler, sampler, inception, dataloader, fid_seed
    torch.cuda.empty_cache()
    gc.collect()

    print_(f'inception_features: {inception_features.shape}')

    results_all = {}
    for fid_ref, name in zip(fid_references, names):
        result = {}
        if adjust_fid_n:
            fid_result = fid_extrapolation(
                inception_features,
                ref_mu=fid_ref[0],
                ref_sigma=fid_ref[1],
                subset_sizes=fid_adjust_subsets,
                target_n=50_000,
                pbar=True,
                pbar_kwargs={'leave': True},
            )
            result['FID'] = fid_result['fids'][-1]
            result['FID@inf'] = fid_result['fid_infinity']
            result['FID@50k'] = fid_result['fid_target']
            result['FIDs'] = fid_result['fids']
            result['R^2'] = fid_result['regression']['r_squared']
        else:
            mu, sigma = inception_features_to_hidden_parameters(inception_features)
            fid = calculate_frechet_distance(mu, sigma, fid_ref[0], fid_ref[1])
            result['FID'] = fid.item()
        
        results_all[name] = result

    # 6. Save results
    output_json_path = os.path.join(output_dir, f'{ckpt_name}_fid_results.json')
    with open(output_json_path, "w") as f:
        json.dump({**settings, **results_all}, f, indent=2)
    print_(f'evaluated FID: {results_all[fid_reference_dataset]["FID"]:.4f}')


if __name__ == "__main__":
    fire.Fire(main)
    # fire.Fire(main_all_ckpt)


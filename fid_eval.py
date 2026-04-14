"""Unified FID evaluation entry point.

Usage
─────
# Basic FID on the checkpoint's reference dataset
python fid_eval.py --ckpt_dir outputs/<run>

# Override reference dataset
python fid_eval.py --ckpt_dir outputs/<run> --references lsun-church_outdoor-train

# Evaluate every ckpt-* subdirectory (produces CSV)
python fid_eval.py --ckpt_dir outputs/<run> --all_ckpt

# With memorization metric
python fid_eval.py --ckpt_dir outputs/<run> --memorization

# Evaluate a specific checkpoint subdir
python fid_eval.py --ckpt_dir outputs/<run>/ckpts/ckpt-050000
"""

import os
import glob
import json
import dataclasses
from typing import Optional, Union, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm
import fire
from pytorch_fid.inception import InceptionV3

from diffusion.scheduler import get_scheduler
from diffusion.sampler import get_sampler
from modeling import get_model
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


def _load_train_args_dict(base_dir: str) -> dict:
    """Walk up from base_dir until train_args.json is found (max 4 levels)."""
    d = os.path.abspath(base_dir)
    for _ in range(5):
        path = os.path.join(d, 'train_args.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(f'train_args.json not found near {base_dir}')


def _resolve_model_ckpt_dir(ckpt_dir: str) -> str:
    """Return the directory that contains model.pt.

    Accepts:
      - outputs/<run>                         → looks for latest ckpts/ckpt-* subdir
      - outputs/<run>/ckpts/ckpt-XXXXXX       → used directly
      - Any dir already containing model.pt   → used directly
    """
    if os.path.exists(os.path.join(ckpt_dir, 'model.pt')):
        return ckpt_dir

    # Try <ckpt_dir>/ckpts/ckpt-* (run root passed)
    ckpts_dir = os.path.join(ckpt_dir, 'ckpts')
    if os.path.isdir(ckpts_dir):
        subdirs = sorted(glob.glob(os.path.join(ckpts_dir, 'ckpt-*')))
        for d in reversed(subdirs):            # newest first
            if os.path.exists(os.path.join(d, 'model.pt')):
                print(f'Using checkpoint: {d}')
                return d

    raise FileNotFoundError(
        f'No model.pt found in "{ckpt_dir}" or its ckpts/ subdirectories.\n'
        f'Run with --all_ckpt to iterate all saved checkpoints.')


def _load_model(model_ckpt_dir: str, args_dict: dict, device='cpu'):
    """Load model weights from model_ckpt_dir/model.pt.

    New format (train.py): model.pt contains EMA-merged weights.
    Legacy format (train_jit.py): model.pt also contains EMA-merged weights.
    Old train.py format: ema_model.pt contains EMA weights (preferred if present).
    """
    model = get_model(args_dict['model_type'], **args_dict['model_cfg'])

    # Legacy: separate ema_model.pt (old train.py style)
    ema_path = os.path.join(model_ckpt_dir, 'ema_model.pt')
    if os.path.exists(ema_path):
        state_dict = torch.load(ema_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        return model.to(device).eval()

    model_path = os.path.join(model_ckpt_dir, 'model.pt')
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def _run_fid(
        model,
        sampler,
        args_dict: dict,
        reference: str,
        n_examples: int,
        generation_batch_size: int,
        device: str,
        adjust_fid_n: bool,
        fid_adjust_subsets: list,
        class_conditioning: bool,
        fid_sample_labels: bool,
        autocast: bool = True,
) -> dict:
    img_shape = dataset_image_shape(args_dict['dataset'])
    n_cls     = dataset_num_classes(args_dict['dataset'])

    fid_dataset = FIDNoiseDataset(
        n_samples=n_examples,
        image_shape=img_shape,
        class_condition=class_conditioning,
        n_classes=n_cls,
        sample_labels=fid_sample_labels,
        seed=args_dict.get('seed', 42),
    )
    loader = DataLoader(
        fid_dataset,
        batch_size=generation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
    )

    inception = InceptionV3(
        [InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=False
    ).to(device).eval()

    cast_ctx = torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16) \
        if autocast else torch.no_grad()

    all_features = []
    for batch in tqdm.tqdm(loader, desc='generating', leave=False):
        z    = batch['z'].to(device)
        cond = (batch['cls'].to(device)
                if class_conditioning and 'cls' in batch else None)
        with cast_ctx:
            samples = sampler.sample(z, model, cond=cond)
        samples = torch.clamp(samples, -1, 1)
        feats = inception(samples.float())[0].flatten(1).cpu()
        all_features.append(feats)

    del inception

    features = torch.cat(all_features, dim=0).numpy().astype(np.float64)
    ref_mu, ref_sigma = load_hidden_parameters(reference, save=False)

    result = {}
    if adjust_fid_n:
        fid_out = fid_extrapolation(
            features,
            ref_mu=ref_mu, ref_sigma=ref_sigma,
            subset_sizes=sorted(fid_adjust_subsets),
            target_n=50_000,
        )
        result['FID']     = fid_out['fids'][-1]
        result['FID@inf'] = fid_out['fid_infinity']
        result['FID@50k'] = fid_out['fid_target']
    else:
        mu, sigma = inception_features_to_hidden_parameters(features)
        result['FID'] = calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma)

    return result


def _eval_one(
        ckpt_dir: str,
        references: Union[str, List[str]],
        n_examples: int,
        generation_batch_size: int,
        guidance_scale: Optional[float],
        memorization: bool,
        device: str,
        autocast: bool,
) -> dict:
    args_dict = _load_train_args_dict(ckpt_dir)

    # Back-compat shims
    if 'lr_schduler_cfg' in args_dict:
        args_dict['lr_scheduler_cfg'] = args_dict.pop('lr_schduler_cfg')
    args_dict.pop('device', None)

    class_conditioning = args_dict.get('class_conditioning', False)
    fid_sample_labels  = args_dict.get('fid_sample_labels', False)
    adjust_fid_n       = args_dict.get('adjust_fid_n', True)
    fid_adjust_subsets = args_dict.get('fid_adjust_subsets', [4000, 6000, 8000, 10000])

    gs = guidance_scale if guidance_scale is not None else args_dict.get('guidance_scale', 1.0)
    cfg_interval = args_dict.get('cfg_interval', None)

    scheduler = get_scheduler(args_dict['scheduler_type'], **args_dict.get('scheduler_cfg', {}))
    # sampler_cfg may already contain guidance_scale/cfg_interval; top-level are defaults
    sampler_kwargs = {
        'guidance_scale': gs,
        'cfg_interval': cfg_interval,
        **args_dict.get('sampler_cfg', {}),
    }
    sampler = get_sampler(args_dict['sampler_type'], scheduler, **sampler_kwargs)
    model_ckpt_dir = _resolve_model_ckpt_dir(ckpt_dir)
    model = _load_model(model_ckpt_dir, args_dict, device=device)

    if isinstance(references, str):
        references = [references]
    if not references:
        references = [args_dict.get('fid_reference_dataset', 'mnist-train')]

    result = {}
    for ref in references:
        fid_res = _run_fid(
            model=model, sampler=sampler, args_dict=args_dict,
            reference=ref,
            n_examples=n_examples,
            generation_batch_size=generation_batch_size,
            device=device,
            adjust_fid_n=adjust_fid_n,
            fid_adjust_subsets=fid_adjust_subsets,
            class_conditioning=class_conditioning,
            fid_sample_labels=fid_sample_labels,
            autocast=autocast,
        )
        prefix = f'{ref}/' if len(references) > 1 else ''
        result.update({f'{prefix}{k}': v for k, v in fid_res.items()})

    if memorization:
        dataset = load_training_dataset(
            args_dict['dataset'], args_dict.get('dataset_dir', 'datasets'), train=True)
        memo_tensor = maybe_build_memorization_tensor(args_dict['dataset'], dataset)
        if memo_tensor is not None:
            seed = make_generation_seed(
                args_dict['dataset'], n_examples,
                seed=args_dict.get('seed', 42), sample_labels=False,
            )
            z    = seed['z'].to(device)
            cond = (seed['cls'].to(device) if class_conditioning and 'cls' in seed else None)
            cast_ctx = torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16) \
                if autocast else torch.no_grad()
            with torch.no_grad(), cast_ctx:
                samples = sampler.sample(z, model, cond=cond)
            samples = torch.clamp(samples, -1, 1).cpu()
            memo_score = calc_memorization_metric(samples, memo_tensor)
            result['memorization'] = float(memo_score)
        else:
            print(f'Memorization not supported for dataset {args_dict["dataset"]} (too large).')

    return result


def main(
        ckpt_dir: str,
        references: Union[str, List[str]] = None,
        n_examples: int = 50_000,
        generation_batch_size: int = 256,
        guidance_scale: Optional[float] = None,
        memorization: bool = False,
        all_ckpt: bool = False,
        device: str = 'cuda',
        autocast: bool = True,
):
    if all_ckpt:
        import pandas as pd
        subdirs = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpts', 'ckpt-*')))
        if not subdirs:
            raise FileNotFoundError(f'No ckpt-* subdirs found under {ckpt_dir}/ckpts/')
        rows = []
        for d in subdirs:
            step = int(os.path.basename(d).replace('ckpt-', ''))
            print(f'\n=== Evaluating {d} (step {step}) ===')
            try:
                res = _eval_one(
                    ckpt_dir=d,
                    references=references,
                    n_examples=n_examples,
                    generation_batch_size=generation_batch_size,
                    guidance_scale=guidance_scale,
                    memorization=memorization,
                    device=device,
                    autocast=autocast,
                )
                print(res)
                rows.append({'step': step, **res})
            except Exception as e:
                print(f'ERROR: {e}')
                rows.append({'step': step, 'error': str(e)})
        df = pd.DataFrame(rows).set_index('step')
        csv_path = os.path.join(ckpt_dir, 'fid_all_ckpts.csv')
        df.to_csv(csv_path)
        print(f'\nSaved results → {csv_path}')
        print(df)
    else:
        result = _eval_one(
            ckpt_dir=ckpt_dir,
            references=references,
            n_examples=n_examples,
            generation_batch_size=generation_batch_size,
            guidance_scale=guidance_scale,
            memorization=memorization,
            device=device,
            autocast=autocast,
        )
        print(result)
        out_path = os.path.join(ckpt_dir, 'fid_result.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'Saved → {out_path}')


if __name__ == '__main__':
    fire.Fire(main)

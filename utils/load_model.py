
import os
import json
from torch import nn
from diffusion import predictor, scheduler, pipeline


def load_diffusion_pipeline(args) -> pipeline.BaseDiffusion:
    model_type = args['model']['type']
    model_args = args['model']['args']
    model_initializer = getattr(predictor, model_type)
    model = model_initializer(**model_args)

    sc_type = args['scheduler']['type']
    sc_args = args['scheduler']['args']
    sc_initializer = getattr(scheduler, sc_type)
    sc = sc_initializer(**sc_args)

    pl_type = args['pipeline']['type']
    pl_args = args['pipeline']['args']
    pl_initializer = getattr(pipeline, pl_type)
    pl = pl_initializer(model, sc, **pl_args)

    return pl


def load_diffusion_pipeline_from_ckpt(ckpt_dir, ckpt_name='latest.pt'):
    arg_path = os.path.join(ckpt_dir, 'train_args.json')
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    with open(arg_path, 'r') as f:
        dpm_cfg = json.load(f)['dpm_cfg']

    dpm = load_diffusion_pipeline(dpm_cfg)


def count_parameters(model: nn.Module, verbose=False):
    total_n_param = 0
    for name, param in model.named_parameters():
        n_param = param.numel()
        if verbose:
            print(f'{name}: {n_param}')
        if param.requires_grad:
            total_n_param += n_param
    return total_n_param




# nohlab_diffusion

Research codebase for diffusion model experiments (Noh Lab).  
Trains image-generation models on MNIST, CIFAR-10, LSUN-church, and ImageNet-1k-256 with a unified training pipeline.

---

## Overview

### Supported formulations

| Scheduler | Sampler | Prediction | Notes |
|-----------|---------|------------|-------|
| `beta` (DDPM linear/cosine β) | `ddim`, `ddpm`, `dpm++2m` | noise or velocity | Classic VP diffusion |
| `rectified_flow` | `euler` | velocity | Rectified flow |
| `jit` (sigmoid-logit-normal t) | `jit` (Euler / Heun ODE) | x₀ → converted to v | JIT formulation |

### Supported backbones

| Name | Class | Typical dataset |
|------|-------|-----------------|
| `unet` | diffusers-style UNet | MNIST, CIFAR-10 |
| `jit` | `JITTransformer` (ViT-style DiT) | LSUN-church, ImageNet-1k-256 |

### Supported datasets

| Key | Resolution | Classes |
|-----|-----------|---------|
| `mnist` | 1×28×28 | 10 |
| `cifar10` | 3×32×32 | 10 |
| `lsun-church_outdoor` | 3×256×256 | — |
| `imagenet-1k-256` | 3×256×256 | 1000 |

---

## Quick start

### Training

All hyperparameters live in a JSON config file. One entry point for all backbones and datasets:

```bash
# Single-GPU
python train.py --train_arg_json arg_json/unet/beta_mnist2.json

# Multi-GPU (batch_size in JSON is per-GPU)
accelerate launch --num_processes 4 train.py --train_arg_json arg_json/jit/base.json

# Resume from latest checkpoint
python train.py --resume_ckpt_dir outputs/<run>

# Overwrite existing output dir (non-interactive)
python train.py --train_arg_json arg_json/... --overwrite
```

### FID evaluation

```bash
# Evaluate latest checkpoint
python fid_eval.py --ckpt_dir outputs/<run>

# With memorization metric (MNIST / CIFAR only)
python fid_eval.py --ckpt_dir outputs/<run> --memorization

# Override reference dataset
python fid_eval.py --ckpt_dir outputs/<run> --references lsun-church_outdoor-train

# Evaluate every saved checkpoint, produce CSV
python fid_eval.py --ckpt_dir outputs/<run> --all_ckpt
```

### Smoke tests

For quick validation after changes (100 steps, tiny models):

```bash
# UNet + DDIM on MNIST (single GPU)
python train.py --train_arg_json arg_json/test_args/unet_mnist_smoke.json --overwrite

# JIT transformer + Euler ODE on MNIST (single GPU)
python train.py --train_arg_json arg_json/test_args/jit_mnist_smoke.json --overwrite

# Multi-GPU smoke test (2 processes)
accelerate launch --num_processes 2 train.py \
    --train_arg_json arg_json/test_args/jit_mnist_smoke.json --overwrite

# Resume smoke test
python train.py --resume_ckpt_dir outputs/test/unet_mnist_smoke
```

After training completes, verify a checkpoint:

```bash
python fid_eval.py --ckpt_dir outputs/test/unet_mnist_smoke \
    --n_examples 1000 --references mnist-train
```

---

## Architecture

```
train.py              # unified training entry point (single- & multi-GPU via Accelerate)
fid_eval.py           # unified FID / memorization evaluation

diffusion/
  scheduler.py        # get_scheduler: 'beta' | 'rectified_flow' | 'jit'
  sampler.py          # get_sampler:   'ddpm' | 'ddim' | 'dpm++2m' | 'euler' | 'jit'

modeling/
  __init__.py         # get_model: 'unet' | 'jit'
  predictor.py        # BasePredictor, GuidedPredictor.pred_conditional
  unet.py             # diffusers-style UNet
  transformer.py      # JITTransformer (ViT + AdaLN + 2D RoPE)

utils/
  data.py             # load_training_dataset, make_generation_seed, FIDNoiseDataset
  fid.py              # streaming Inception features, FID computation
  fid_infinity.py     # FID-∞ extrapolation
  validate_memo.py    # memorization metric
  augmentation.py     # string-name → torchvision transform

arg_json/
  unet/               # UNet run configs
  jit/                # JIT transformer run configs
  mnist_single/       # per-class single-digit MNIST configs
  test_args/          # (git-ignored) quick smoke-test configs
```

### Sampler / Scheduler API

```python
scheduler = get_scheduler('beta', n_steps=1000)
sampler   = get_sampler('ddim', scheduler, n_steps=50, guidance_scale=1.0)
samples   = sampler.sample(z, model, cond=cls)   # unified signature

loss = scheduler.get_loss(x, model, cond=cls, uncond_mask=uncond_mask)
```

CFG is handled inside `BaseSampler._predict` (batch-doubled forward + linear combination). No `get_pred_fn` — the model only needs to implement `pred_conditional(z, t, cond, uncond_mask)`.

---

## Config format

All fields in `TrainArgs` (see `train.py`). Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `output_dir` | required | where to save checkpoints, logs, sample grids |
| `model_type` | `"unet"` | `"unet"` or `"jit"` |
| `scheduler_type` | `"beta"` | `"beta"`, `"rectified_flow"`, or `"jit"` |
| `sampler_type` | `"ddim"` | `"ddpm"`, `"ddim"`, `"euler"`, or `"jit"` |
| `dataset` | `"mnist"` | `"mnist"`, `"cifar10"`, `"lsun-church_outdoor"`, `"imagenet-1k-256"` |
| `batch_size` | `128` | per-GPU when using multi-GPU |
| `class_conditioning` | `false` | enables class label input + CFG |
| `guidance_scale` | `1.0` | CFG scale (1.0 = no guidance) |
| `bf16` | `true` | bfloat16 mixed precision |
| `compile` | `false` | `torch.compile` the model |
| `fid_eval_steps` | `null` | if set, runs streaming FID every N steps |

### Checkpoint layout

```
outputs/<run>/
  train_args.json        # config snapshot
  train_log.jsonl        # per-step loss / metrics
  fid_evaluations.jsonl  # per-eval FID results
  examples/              # sampled image grids
  latest_state/          # full training state (accelerator.save_state)
  ckpts/
    ckpt-010000/
      model.pt           # raw model weights
      ema_model.pt       # EMA weights
```

---

## Notes

- W&B project: `noh-diffusion`
- Data normalized to `[-1, 1]` everywhere
- `p_uncond` controls the unconditional dropout rate for CFG training
- LMDB caches for LSUN (`_cache_home…lmdb/`) are committed artifacts — do not delete
- `arg_json/test_args/` is git-ignored; use it for temporary experiment configs

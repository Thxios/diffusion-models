# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research codebase for diffusion-model experiments (noh-lab). Trains image-generation models (MNIST, CIFAR-10, LSUN, ImageNet) under several diffusion formulations (variance-preserving/beta, rectified flow, JIT) with UNet and Transformer (JIT) backbones, evaluated with FID / FID-∞ and memorization metrics. Runs are configured entirely by JSON files in `arg_json/`.

## Common commands

Training is driven by per-run JSON configs. There are three entry points; pick by model/backbone:

```bash
# UNet / rectified-flow / beta on MNIST, CIFAR — single GPU, fire CLI
python train.py arg_json/unet/beta_mnist2.json
python train.py --resume_ckpt_dir outputs/<run>              # resume
python train.py arg_json/... --overwrite                    # clobber output_dir

# Class-conditioned MNIST-single-class experiments
python train_cls0.py arg_json/mnist_single/beta_cls3.json
bash run_single_cls.sh                                      # runs cls1..cls9 sequentially

# JIT transformer (single GPU)
python train_jit.py arg_json/jit/base.json

# JIT transformer (multi-GPU via HF Accelerate, bf16)
accelerate launch --num_processes 4 train_jit_multigpu.py \
    --train_arg_json arg_json/jit/base.json
# NOTE: batch_size and generation_batch_size in the JSON are PER-GPU here.
# The JSON's "device" field is ignored; remove or leave it, multigpu pops it.
```

Standalone FID evaluation of trained JIT checkpoints:
```bash
python jit_fid.py ...            # see argparse/fire signature at bottom of file
python save_fid_eval.py ...      # writes FID reference stats
```

No test suite, no linter, and no build system — this is a research repo. Outputs land under `outputs/<run>/` (checkpoints, sampled grids, `train_args.json`) and stdout logs under `stdouts/`. W&B project is `noh-diffusion`.

Legacy arg JSONs use the key `lr_schduler_cfg` (typo). `train_jit.py` and `train_jit_multigpu.py` silently rename it to `lr_scheduler_cfg` on load; `train.py` does not — keep that in mind when porting configs.

## Architecture

Three orthogonal axes compose a run: **model** × **scheduler** × **sampler**. Each is registered by string name and instantiated from `model_cfg` / `scheduler_cfg` / `sampler_cfg` blocks in the JSON.

- `modeling/` — predictor backbones.
  - `predictor.py` defines `BasePredictor` (common interface: takes `(x, t, cls)` → prediction).
  - `unet.py` — diffusers-style UNet built from `modeling/blocks.py` + `attention.py` + `embeddings.py` + `activation.py`.
  - `transformer.py` — `JITTransformer`, a ViT-style patch transformer used for JIT experiments.
  - `__init__.py::get_model(name, **cfg)` dispatches `'unet'` | `'jit'`.

- `diffusion/scheduler.py` — forward process and training loss.
  - `BaseScheduler` + `Schedule`/`VPSchedule` dataclasses (alpha, sigma, log_snr).
  - `BetaScheduler` (DDPM-style linear/cosine β), `RectifiedFlowScheduler`, `JITScheduler`.
  - Each scheduler advertises a `pred_type` (`'noise'`, `'velocity'`, …) and implements `get_loss` / `diffuse`.
  - `get_scheduler(name, **cfg)` dispatches `'beta'` | `'rectified_flow'` (JIT is constructed directly in `train_jit*.py`).

- `diffusion/sampler.py` — reverse process for generation.
  - `BaseSampler` subclasses: `DDPMSampler`, `DDIMSampler`, `RectifiedFlowEulerSampler`, `JITSampler`.
  - Scheduler and sampler must agree on `pred_type`; `train*.py` asserts this at startup.

- `train.py` / `train_jit.py` / `train_jit_multigpu.py` — each defines its own `TrainArgs` dataclass (the JSON schema) and a `Trainer`. Responsibilities: data loading (`CIFAR10`/`MNIST`/LSUN), EMA via `diffusers.training_utils.EMAModel`, periodic eval/sample grid, FID + FID-∞ evaluation, checkpointing, W&B logging. `train_jit_multigpu.py` additionally: DDP via `Accelerator`, per-process seed offset (`seed + process_index`) so diffusion noise is uncorrelated across GPUs, streaming FID (gather Inception features, not images).

- `utils/`
  - `fid.py`, `fid_infinity.py` — Inception features, Fréchet distance, FID-∞ extrapolation over subset sizes.
  - `validate_memo.py` — memorization metric over the training set.
  - `augmentation.py` — string-name → `torchvision.transforms` lookup (JSON `augmentations` list).
  - `model.py` — small model-side helpers (`count_parameters`, …).

- `arg_json/` — run configs, grouped by backbone (`unet/`, `jit/`, `mnist_single/`). `arg_json/mnist_single/make.py` generates the per-class MNIST configs.

- `notebooks/`, `test_notebooks/` — interactive analysis (`interpolate*.ipynb`, `snr_test.ipynb`, etc.); not part of any automated pipeline.

## Conventions worth knowing

- All hyperparameters live in the JSON, including `output_dir`, `wandb_run_name`, and `device` (e.g. `"cuda:5"`) for single-GPU scripts. There is no `argparse` layer beyond `fire` wrapping `main()`.
- `output_dir` non-empty + no `--overwrite` → the trainer interactively prompts on stdin; pass `--overwrite` for non-interactive runs.
- Data is normalized to `[-1, 1]` everywhere (`x/127.5 - 1`).
- `p_uncond` + `guidance_scale` implement classifier-free guidance; `n_class_embeddings` in `model_cfg` toggles class conditioning.
- LMDB caches for LSUN (`_cache_home...lmdb/`) are committed artifacts — do not delete.

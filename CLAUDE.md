# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

Research codebase for diffusion-model experiments (noh-lab). Trains image-generation models (MNIST, CIFAR-10, LSUN, ImageNet) under several diffusion formulations (variance-preserving/beta, rectified flow, JIT) with UNet and Transformer (JIT) backbones, evaluated with FID / FID-∞ and memorization metrics. Runs are configured entirely by JSON files in `arg_json/`.

## Common commands

Training is driven by per-run JSON configs. **One entry point for all backbones and datasets.**

```bash
# Single-GPU (UNet, JIT, any backbone)
python train.py --train_arg_json arg_json/unet/beta_mnist2.json
python train.py --train_arg_json arg_json/jit/base.json

# Multi-GPU via HF Accelerate (batch_size in JSON is PER-GPU)
accelerate launch --num_processes 4 train.py --train_arg_json arg_json/jit/base.json

# Resume from latest checkpoint
python train.py --resume_ckpt_dir outputs/<run>

# Overwrite existing output_dir without interactive prompt
python train.py --train_arg_json arg_json/... --overwrite

# Class-conditioned MNIST single-class experiments
bash run_single_cls.sh                # runs cls1..cls9 sequentially
```

FID evaluation:
```bash
# Evaluate latest checkpoint under outputs/<run>/ckpts/
python fid_eval.py --ckpt_dir outputs/<run>

# Or point directly at a specific checkpoint dir
python fid_eval.py --ckpt_dir outputs/<run>/ckpts/ckpt-050000

# With memorization metric (MNIST / CIFAR-10 only)
python fid_eval.py --ckpt_dir outputs/<run> --memorization

# Override reference dataset
python fid_eval.py --ckpt_dir outputs/<run> --references lsun-church_outdoor-train

# Evaluate every ckpt-* subdir, save CSV
python fid_eval.py --ckpt_dir outputs/<run> --all_ckpt
```

Migrate old `train_args.json` files under `outputs/` (run once after pulling this refactor):
```bash
python migrate_train_args.py           # dry-run
python migrate_train_args.py --apply   # apply in-place
```

No test suite, no linter, and no build system — this is a research repo. Outputs land under `outputs/<run>/` (checkpoints, sampled grids, `train_args.json`) and stdout logs under `stdouts/`. W&B project is `noh-diffusion`.

## Architecture

Three orthogonal axes compose a run: **model** × **scheduler** × **sampler**. Each is registered by string name and instantiated from `model_cfg` / `scheduler_cfg` / `sampler_cfg` blocks in the JSON.

- `modeling/` — predictor backbones.
  - `predictor.py` — `BasePredictor` / `GuidedPredictor`. The only public contract is `pred_conditional(z, t, cond, uncond_mask)`. CFG is handled inside the sampler — there is no `get_pred_fn`.
  - `unet.py` — diffusers-style UNet built from `modeling/blocks.py` + `attention.py` + `embeddings.py` + `activation.py`.
  - `transformer.py` — `JITTransformer`, a ViT-style patch transformer (AdaLN + 2D RoPE). `rope_max_res` kwarg controls the RoPE frequency table size (default 1024).
  - `__init__.py::get_model(name, **cfg)` dispatches `'unet'` | `'jit'`.

- `diffusion/scheduler.py` — forward process and training loss.
  - `BetaScheduler` (DDPM-style linear/cosine β), `RectifiedFlowScheduler`, `JITScheduler` (sigmoid-logit-normal t, x₀ prediction).
  - All schedulers implement `get_loss(x, model, gen=None, **model_call_kwargs)` calling `model.pred_conditional(...)`.
  - `get_scheduler(name, **cfg)` dispatches `'beta'` | `'rectified_flow'` | `'jit'`.

- `diffusion/sampler.py` — reverse process for generation.
  - Unified `BaseSampler.__init__(scheduler, n_steps, guidance_scale, cfg_interval, pbar)`.
  - Unified `BaseSampler.sample(z, model, cond, return_intermediates)` — CFG is handled inside via `_predict`.
  - Subclasses: `DDPMSampler`, `DDIMSampler`, `DPMpp2MSolver`, `RectifiedFlowEulerSampler`, `JITSampler`.
  - `get_sampler(name, scheduler, **kwargs)` dispatches `'ddpm'` | `'ddim'` | `'euler'` | `'jit'`.

- `train.py` — **single unified trainer** (Accelerate-based). Handles single- and multi-GPU, all datasets and backbones. Key points:
  - `TrainArgs` dataclass is the single JSON schema.
  - Streaming FID (per-batch Inception + DDP gather) for all runs.
  - Checkpoint: `latest_state/` (full Accelerate state for resume) + `ckpts/ckpt-XXXXXX/model.pt` (EMA-merged weights).
  - Seed: `seed + process_index` so diffusion noise is decorrelated across ranks.

- `fid_eval.py` — unified FID / memorization evaluator (registry-based). Accepts run root or specific ckpt subdir; auto-resolves the latest checkpoint when given a run root.

- `utils/`
  - `data.py` — `load_training_dataset`, `make_generation_seed`, `maybe_build_memorization_tensor`, `FIDNoiseDataset`.
  - `fid.py`, `fid_infinity.py` — Inception features, Fréchet distance, FID-∞ extrapolation. HF `datasets` import is lazy (ImageNet branch only).
  - `validate_memo.py` — memorization metric over the training set.
  - `augmentation.py` — string-name → `torchvision.transforms` lookup (JSON `augmentations` list).
  - `model.py` — small model-side helpers (`count_parameters`, …).

- `migrate_train_args.py` — one-shot migration script for old `outputs/**/train_args.json` files.

- `arg_json/` — run configs grouped by backbone (`unet/`, `jit/`, `mnist_single/`). `arg_json/test_args/` is git-ignored and holds quick smoke-test configs.

- `notebooks/`, `test_notebooks/` — interactive analysis; not part of any automated pipeline.

## Conventions worth knowing

- All hyperparameters live in the JSON, including `output_dir` and `wandb_run_name`. The `device` field has been removed — Accelerate manages device placement.
- `output_dir` non-empty + no `--overwrite` → the trainer interactively prompts on stdin; pass `--overwrite` for non-interactive runs.
- Data is normalized to `[-1, 1]` everywhere (`x/127.5 - 1`).
- `class_conditioning: true` + `p_uncond` + `guidance_scale` implement classifier-free guidance. `n_class_embeddings` in `model_cfg` controls the embedding table size.
- `sampler_cfg` values for `guidance_scale` / `cfg_interval` take precedence over the top-level TrainArgs fields (sampler_cfg overrides).
- Checkpoint `model.pt` always contains **EMA-merged** weights. `ema_model.pt` is not written by the current trainer (only by the legacy `train.py`).
- LMDB caches for LSUN (`_cache_home...lmdb/`) are committed artifacts — do not delete.

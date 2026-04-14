# Plan: Unify JIT training path with the rest of the framework

> **For future agents**: This file + `CLAUDE.md` should be sufficient to implement this plan without re-deriving context. Read both before starting. When in doubt about the *current* state of the repo, `git log` / `git show` to check, because file-level facts in this plan can drift.

---

## 1. Problem context

### 1.1 The symptom

The repo has three near-duplicate training scripts and two near-duplicate evaluation scripts:

| Script | Purpose | Registry-aware? |
|---|---|---|
| `train.py` | UNet on MNIST/CIFAR with beta/rectified-flow schedulers | Yes (uses `get_model`/`get_scheduler`/`get_sampler`) |
| `train_jit.py` | JIT-transformer on LSUN/ImageNet, single-GPU | **No** — hardcodes `JITTransformer`/`JITScheduler`/`JITSampler` |
| `train_jit_multigpu.py` | Same as above but HF-Accelerate DDP + bf16 + streaming FID | **No** |
| `train_cls0.py` | Per-class MNIST UNet variant of `train.py` | Yes |
| `save_fid_eval.py` | Post-hoc FID + memorization on UNet checkpoints | Yes |
| `jit_fid.py` | Post-hoc multi-reference FID on JIT checkpoints | **No** |

Changes to the training loop (e.g. logging, checkpointing, EMA policy) need to be made in 3–5 places and are drifting apart. Maintenance burden is real and growing.

### 1.2 The root cause — API incompatibilities

The JIT formulation (rectified-flow-style velocity prediction with sigmoid-logit-normal t sampling, trained as an `x`-predictor) was added with a sampler API that doesn't compose with the existing base API:

| Concern | Base samplers (DDPM/DDIM/RFEuler) | `JITSampler` |
|---|---|---|
| Scheduler is passed to | `sample(z, scheduler, pred_fn)` | `__init__(scheduler=...)` |
| Model interface | `pred_fn: (z,t)→pred` (built by `GuidedPredictor.get_pred_fn`) | `model: GuidedPredictor` called directly inside the loop |
| CFG location | `GuidedPredictor.get_pred_fn(cond, guidance_scale)` batches z twice | `JITSampler.pred_v_guided` batches z twice |
| `sample` signature | `(z, scheduler, pred_fn, return_intermediates=False, **kw)` | `(z, model, cond=None, return_intermediates=False)` |
| Registered in `get_sampler` | `'ddpm'/'ddim'/'euler'` | **not registered** |

And the scheduler side:

| Concern | `BetaScheduler` / `RectifiedFlowScheduler` | `JITScheduler` |
|---|---|---|
| `get_loss` signature | `(x, model, gen, **model_call_kwargs)` | takes `cls`, `uncond_mask` as named args |
| Registered in `get_scheduler` | yes | **not registered** |
| t sampling | uniform on timestep indices | `sigmoid(logit_t_mean + logit_t_std * N(0,1))` |

And the trainer side:

| Concern | `train.py` Trainer | `train_jit*.py` Trainer |
|---|---|---|
| Batch format | `(x, cls)` tuples | `{'image', 'label'}` dicts |
| Supported datasets | MNIST, CIFAR-10 | LSUN-church, ImageNet-1k-256 |
| Mixed precision | optional `bf16` flag | always-on bf16 |
| Compile | — | `torch.compile` |
| FID | bulk: accumulate all samples, one Inception pass | streaming: per-batch Inception + DDP gather |
| DDP | — | HF Accelerate |
| Checkpoint | `torch.save` model.pt + ema_model.pt | `torch.save` EMA-merged model.pt; multigpu uses `accelerator.save_state` for `latest/` |
| TrainArgs field names | `lr_schduler_cfg` (typo) | `lr_scheduler_cfg` |
| `device` field | `"cuda:5"` etc. | multigpu pops it silently |

### 1.3 Intent

Collapse the three training scripts into **one** (`train.py`) and the two evaluation scripts into **one** (`fid_eval.py`), by making the JIT formulation just another registered `(model, scheduler, sampler)` triple. The JIT training path should flow through the *same* code as UNet training; only the registered components differ. No JIT-specific branches in the trainer.

Design choices (confirmed with user):

1. **Single trainer on Accelerate** — single-GPU runs with `num_processes=1`; multi-GPU launches via `accelerate launch --num_processes N`. Collapses 3 scripts into 1.
2. **CFG belongs to samplers (JIT style)** — every sampler takes `(scheduler, guidance_scale, cfg_interval)` at construction and calls `model.pred_conditional` directly in the loop. `GuidedPredictor.get_pred_fn` is deleted.
3. **Streaming FID is the only FID** — borrow the multigpu implementation. On a single process it's a no-op gather.
4. **One `fid_eval.py`** — registry-driven, with flags for multi-reference and memorization metric.

---

## 2. Target architecture

```
train.py                 ← single training entry point (single- & multi-GPU via Accelerate)
fid_eval.py              ← single evaluation entry point (registry-based)

diffusion/
  scheduler.py           get_scheduler: 'beta' | 'rectified_flow' | 'jit'
  sampler.py             get_sampler:   'ddpm' | 'ddim' | 'euler' | 'jit'
                         Unified signatures:
                           BaseSampler.__init__(scheduler, n_steps, guidance_scale=1.0,
                                                cfg_interval=None, pbar=False, pbar_kwargs=None)
                           BaseSampler.sample(z, model, cond=None,
                                              return_intermediates=False, **kwargs)

modeling/
  __init__.py            get_model: 'unet' | 'jit'
  predictor.py           BasePredictor + GuidedPredictor (only the pred_conditional contract;
                         no more get_pred_fn — CFG moved to samplers)
  unet.py, transformer.py, …

utils/
  fid.py                 streaming calc_inception_features(dataset=...); lazy HF datasets import
  data.py  (NEW)         load_training_dataset(name, ...) → HF-style dict-batch Dataset
                         make_generation_seed(name, n, ...)
                         maybe_build_memorization_tensor(name, dataset)

arg_json/
  unet/, jit/, mnist_single/   JSON schema: drop 'device', rename 'lr_schduler_cfg' → 'lr_scheduler_cfg'

notebooks/
  interpolate.ipynb, interpolate2.ipynb, snr_test.ipynb  ← updated to new sampler API
```

Deleted: `train_jit.py`, `train_jit_multigpu.py`, `train_cls0.py`, `jit_fid.py`, `save_fid_eval.py`.

---

## 3. Concrete changes

Implement in this order to minimize broken intermediate states (each step leaves the tree in a compilable state if you stop there).

### Step 1 — `modeling/predictor.py` (issue 9 resolved here)

Delete `GuidedPredictor.get_pred_fn`. Keep the class as just:

```python
class GuidedPredictor(BasePredictor):
    def pred_conditional(self, z, t, cond=None, uncond_mask=None, **kwargs):
        raise NotImplementedError()
```

Both `UNet` and `JITTransformer` already implement `pred_conditional`. `JITTransformer` uses the learned `uncond` class-embedding (padding_idx) — that stays.

### Step 2 — `diffusion/sampler.py` (API unification)

Rewrite `BaseSampler` as the single source of truth for construction and CFG. All subclasses share the same `__init__` and a private `_predict(self, model, z, t, cond, pred_type_out)` helper.

```python
class BaseSampler:
    pred_type_out = 'noise'  # what sample() wants the model's effective prediction to be

    def __init__(self, scheduler, n_steps,
                 guidance_scale=1.0, cfg_interval=None,
                 pbar=False, pbar_kwargs=None):
        self.scheduler = scheduler
        self.n_steps = n_steps
        self.guidance_scale = guidance_scale
        self.cfg_interval = cfg_interval
        self.pbar = pbar
        self.pbar_kwargs = {'leave': False, **(pbar_kwargs or {})}

    @torch.no_grad()
    def _predict(self, model, z, t, cond):
        """Returns the scheduler-appropriate prediction type with CFG applied."""
        # 1) conditional branch — always run
        if self.guidance_scale == 1.0 or cond is None:
            return self._to_pred_type(model.pred_conditional(z, t, cond=cond), z, t)
        # 2) CFG: one forward pass over batch-doubled inputs
        z2 = torch.cat([z, z], dim=0)
        t2 = torch.cat([t, t], dim=0)
        cond2 = torch.cat([cond, cond], dim=0)
        uncond_mask2 = torch.cat([torch.ones_like(cond), torch.zeros_like(cond)], dim=0)
        pred = model.pred_conditional(z2, t2, cond=cond2, uncond_mask=uncond_mask2)
        pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
        # convert both to the requested pred_type (noise / v / x0) using the scheduler
        p_uncond = self._to_pred_type(pred_uncond, z, t)
        p_cond   = self._to_pred_type(pred_cond, z, t)
        cfg = self._cfg_at(t, z.ndim)
        return p_uncond + cfg * (p_cond - p_uncond)

    def _cfg_at(self, t, ndim):
        if self.cfg_interval is None:
            return self.guidance_scale
        low, high = self.cfg_interval
        mask = (t < high) & ((low == 0) | (t > low))
        cfg = torch.where(mask, self.guidance_scale, 1.0)
        return cfg.view(-1, *([1] * (ndim - 1)))

    def _to_pred_type(self, model_out, z, t):
        """Subclass hook: convert model's raw output to what this sampler consumes.
        Default: identity (noise predictor feeding a noise-consuming sampler)."""
        return model_out

    def sample(self, z, model, cond=None, return_intermediates=False, **kwargs):
        raise NotImplementedError
```

Per-subclass rewrites:

- **`DDPMSampler`, `DDIMSampler`, `DPMpp2MSolver`** consume `eps` (noise). Existing UNets are trained as noise predictors so `_to_pred_type` stays identity. Replace the old `pred_fn(z, t)` call with `self._predict(model, z, t, cond)`. Drop the `scheduler` parameter from `sample` — use `self.scheduler`.
- **`RectifiedFlowEulerSampler`** consumes `v`. `pred_type_out = 'rect_flow'` stays; `_to_pred_type` is identity because that model is v-trained. Same `pred_fn → _predict` swap. Keep the `sigmas.new_zeros(1)` fix already in the file.
- **`JITSampler`** consumes `v` but the model predicts `x`. Override `_to_pred_type` to call `self.scheduler.x_to_v(model_out, z, t)`. Collapse the current `pred_v_guided` into `_predict`. Keep the subclass-local `ode_solver='euler'|'heun'` option — it's orthogonal to the base class.

Registry update:

```python
def get_sampler(name, scheduler, **kwargs):
    return {
        'ddpm':  DDPMSampler,
        'ddim':  DDIMSampler,
        'euler': RectifiedFlowEulerSampler,
        'jit':   JITSampler,
    }[name](scheduler=scheduler, **kwargs)
```

Remove the broken `if __name__ == '__main__'` smoke test at the bottom of the file.

### Step 3 — `diffusion/scheduler.py`

1. Add `'jit'` to `get_scheduler`:
   ```python
   elif name == 'jit':
       return JITScheduler(**kwargs)
   ```
2. Make `JITScheduler.get_loss` match the `BaseScheduler.get_loss(x, model, gen, **model_call_kwargs)` contract. Current code takes explicit `cls`/`uncond_mask` — change to forward `**model_call_kwargs` into `model.pred_conditional(x_t, t, **model_call_kwargs)`. The trainer will pass `cond=cls, uncond_mask=uncond_mask` uniformly.
3. Leave the sigmoid-logit-normal t sampling and the `noise_scale`/`t_eps` fields as-is (those are the interesting physics and the whole reason JIT exists).

### Step 4 — `modeling/transformer.py` (issue 9 second half)

Promote the hardcoded `Dynamic2DRoPE(head_dim, max_res=1024)` to a kwarg:

```python
def __init__(self, ..., rope_max_res=1024):
    ...
    self.rope_2d = Dynamic2DRoPE(head_dim, max_res=rope_max_res)
```

Update `arg_json/jit/*.json` only if you want non-default values — default stays 1024 so existing configs keep working.

### Step 5 — `utils/fid.py` (issue 8)

Move `from datasets import load_dataset` from the module top into the `elif dataset.startswith('imagenet-1k-256'):` branch of `save_hidden_parameters`. Rationale: without this, the module fails to import on any machine that doesn't have the HF `datasets` package, even though FID for MNIST/CIFAR/LSUN doesn't need it.

No other changes — streaming `calc_inception_features(dataset=...)` path stays.

### Step 6 — `utils/data.py` (NEW)

Factor dataset loading (currently duplicated between `train.py` and `train_jit*.py`). Public API:

```python
def load_training_dataset(
    name: str,                          # 'mnist' | 'cifar10' | 'lsun-church_outdoor' | 'imagenet-1k-256'
    data_dir: str,
    train: bool = True,
    augmentations: Optional[List[str]] = None,
    image_size: Optional[int] = None,   # for lsun/imagenet resize
) -> torch.utils.data.Dataset:
    """Returns a Dataset where __getitem__(i) -> {'image': (C,H,W) float in [-1,1], 'label': int}."""

def image_shape(name: str) -> Tuple[int, int, int]: ...
def num_classes(name: str) -> int: ...

def make_generation_seed(name, n_examples, seed=None, sample_labels=False) -> Dict[str, torch.Tensor]:
    """Returns {'z': (n,C,H,W), 'cls': (n,) int64}. Moved from train.py."""

def maybe_build_memorization_tensor(name, dataset) -> Optional[torch.Tensor]:
    """For small datasets (mnist/cifar10) return the full (N,C,H,W) tensor used by
    calc_memorization_metric. Return None otherwise (LSUN/ImageNet: too large)."""
```

Implementation outline:
- `mnist`/`cifar10`: wrap torchvision datasets; `__getitem__` returns dict. Normalization → `[-1, 1]`.
- `lsun-church_outdoor`: current `train_jit.py` code (torchvision LSUN + resize/crop/normalize).
- `imagenet-1k-256`: current `train_jit_multigpu.py` code (HF `datasets.load_dataset` + `set_transform`). **Lazy-import** `datasets` inside this branch (same reasoning as Step 5).

### Step 7 — Rewrite `train.py` (single unified trainer)

#### 7.1 Unified `TrainArgs`

One dataclass, superset of the three existing ones:

```python
@dataclasses.dataclass
class TrainArgs:
    # --- run ---
    output_dir: str
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    # --- schedule ---
    max_steps: int = 100_000
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: Optional[int] = None
    eval_n_examples: int = 40
    # --- FID ---
    fid_eval_steps: Optional[int] = None
    fid_ema: bool = True
    fid_reference_dataset: str = 'mnist-train'
    fid_n_examples: int = 10_000
    fid_sample_labels: bool = False
    generation_batch_size: int = 256
    inception_batch_size: int = 512
    adjust_fid_n: bool = True
    fid_adjust_subsets: List[int] = field(default_factory=lambda: [4000, 6000, 8000, 10000])
    # --- optim ---
    batch_size: int = 128          # PER PROCESS when num_processes > 1
    lr: float = 2e-4
    lr_scheduler: Optional[str] = None
    lr_warmup_steps: int = 0
    lr_scheduler_cfg: dict = field(default_factory=dict)
    optimizer: str = 'adamw'
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    clip_grad_norm: float = 1.0
    # --- EMA ---
    use_ema: bool = True
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    # --- data ---
    dataset: str = 'mnist'
    dataset_dir: str = 'datasets'
    augmentations: List[str] = field(default_factory=list)
    image_size: Optional[int] = None
    dataloader_num_workers: int = 2
    dataloader_drop_last: bool = True
    dataloader_pin_memory: bool = True
    # --- misc ---
    seed: int = 42
    bf16: bool = True
    compile: bool = False
    # --- diffusion ---
    p_uncond: float = 0.2
    class_conditioning: bool = False
    guidance_scale: float = 1.0
    cfg_interval: Optional[Tuple[float, float]] = None
    model_type: str = 'unet'
    model_cfg: dict = field(default_factory=dict)
    scheduler_type: str = 'beta'
    scheduler_cfg: dict = field(default_factory=dict)
    sampler_type: str = 'ddim'
    sampler_cfg: dict = field(default_factory=dict)
```

**Removed**: `device` (Accelerate decides), `fid_adjust_subsets` default narrows if too large at load time.

#### 7.2 JSON loader back-compat

```python
def _load_train_args(path) -> TrainArgs:
    with open(path) as f:
        d = json.load(f)
    # back-compat shims
    if 'lr_schduler_cfg' in d:                           # typo from older configs
        d['lr_scheduler_cfg'] = d.pop('lr_schduler_cfg')
    d.pop('device', None)                                # legacy, Accelerate-managed now
    return TrainArgs(**d)
```

#### 7.3 Trainer

Single class. Key points:

- Construct `accelerator = Accelerator(mixed_precision='bf16' if arg.bf16 else 'no')`. Single-GPU path automatically has `num_processes == 1` — no branching needed.
- Seed: `seed_everything(arg.seed + accelerator.process_index)` so diffusion noise is decorrelated across ranks (same as current `train_jit_multigpu.py`).
- Build via registries:
  ```python
  model     = get_model(arg.model_type,     **arg.model_cfg)
  scheduler = get_scheduler(arg.scheduler_type, **arg.scheduler_cfg)
  sampler   = get_sampler(arg.sampler_type, scheduler,
                          guidance_scale=arg.guidance_scale,
                          cfg_interval=arg.cfg_interval,
                          **arg.sampler_cfg)
  assert scheduler.pred_type == sampler.pred_type_out or \
         hasattr(sampler, '_to_pred_type'), 'scheduler/sampler prediction-type mismatch'
  ```
- Dataset via `utils.data.load_training_dataset(...)`. `DataLoader` uses `DistributedSampler` when `accelerator.num_processes > 1`; otherwise shuffle=True.
- `model, optimizer, train_dl, lr_sched = accelerator.prepare(...)`.
- Optional `model = torch.compile(accelerator.unwrap_model(model))` if `arg.compile`.
- `raw_model = accelerator.unwrap_model(model)` — use this for sampling and for EMA updates (avoids DDP overhead during `no_grad` eval).
- EMA via `diffusers.training_utils.EMAModel` on `raw_model.parameters()`. Main-process-only state; broadcast is unnecessary because `raw_model` is replica-identical after each `accelerator.step`.
- **Training step** (now uniform across UNet and JIT):
  ```python
  cls = batch['label'].to(accelerator.device) if arg.class_conditioning else None
  uncond_mask = None
  if cls is not None and arg.p_uncond > 0:
      uncond_mask = torch.rand(cls.shape, device=cls.device) < arg.p_uncond
  with accelerator.autocast():
      loss = scheduler.get_loss(
          batch['image'].to(accelerator.device),
          raw_model,
          cond=cls,
          uncond_mask=uncond_mask,
      )
  accelerator.backward(loss)
  ...
  ```
  This is uniform *because* Step 3 made `JITScheduler.get_loss` accept `**model_call_kwargs`.
- **Eval sampling** (uniform):
  ```python
  z = generation_seed['z'].to(accelerator.device)
  cls = generation_seed['cls'].to(accelerator.device) if arg.class_conditioning else None
  samples = sampler.sample(z, raw_model, cond=cls)
  ```
- **FID**: port `train_jit_multigpu.py`'s streaming implementation verbatim (FIDNoiseDataset → per-batch generate → Inception → `accelerator.gather`). On single-GPU `gather` is identity. Memorization metric computed only when `utils.data.maybe_build_memorization_tensor` returned non-None.
- **Checkpoint**:
  - `latest/` → `accelerator.save_state(ckpt_base_dir/'latest')` (captures optimizer, lr_sched, RNG per process, DDP-wrapped model state).
  - `ckpt-XXXXXX/` → main-process only: `torch.save(raw_model.state_dict(), 'model.pt')` + `torch.save(ema_model.state_dict(), 'ema_model.pt')` if EMA.
  - `train_args.json` dumped once at start.
- **Logging / wandb / image grids**: `if accelerator.is_main_process:` guards.

#### 7.4 CLI

```bash
# single-GPU
python train.py --train_arg_json arg_json/unet/beta_mnist2.json
# multi-GPU
accelerate launch --num_processes 4 train.py --train_arg_json arg_json/jit/base.json
# resume
python train.py --resume_ckpt_dir outputs/<run>
```

The existing `fire.Fire(main)` pattern is preserved; `main(train_arg_json=None, resume_ckpt_dir=None, overwrite=False)`.

### Step 8 — `fid_eval.py` (NEW; replaces `jit_fid.py` + `save_fid_eval.py`)

Registry-driven. Signature:

```python
def main(
    ckpt_dir: str,                       # outputs/<run> or outputs/<run>/ckpt-XXXXXX
    references: Union[str, List[str]] = None,   # default: arg.fid_reference_dataset
    n_examples: int = 50_000,
    generation_batch_size: int = 256,
    inception_batch_size: int = 512,
    guidance_scale: Optional[float] = None,     # overrides the ckpt's value if set
    ema: bool = True,                           # load ema_model.pt if present
    memorization: bool = False,
    save_samples: Optional[str] = None,         # dir; if set, also save sample images
    all_ckpt: bool = False,                     # iterate every ckpt-* subdir under ckpt_dir
):
    ...
```

Implementation:
1. Load `train_args.json` from `ckpt_dir` (or its parent if `ckpt-*`).
2. Build `model/scheduler/sampler` via registries (same three-liner as trainer).
3. Load `model.pt` (or `ema_model.pt` if `ema=True` and present).
4. Reuse the trainer's streaming-FID helper — factor it into `utils/fid.py` as `streaming_fid(model, sampler, n_examples, ...)` so both trainer and eval script call one function.
5. If `memorization`: call `calc_memorization_metric(samples, train_tensor)` using `utils.data.maybe_build_memorization_tensor`.
6. If `all_ckpt`: loop over `glob(ckpt_dir + '/ckpt-*')`, collect a `pd.DataFrame` of results, save as CSV.

Delete `jit_fid.py` and `save_fid_eval.py`.

### Step 9 — Notebooks

Three files reference the old API: `notebooks/interpolate.ipynb`, `notebooks/interpolate2.ipynb`, `notebooks/snr_test.ipynb`. Update call sites:

Old:
```python
sampler = get_sampler(cfg['sampler_type'], **cfg['sampler_cfg'])
pred_fn = model.get_pred_fn(cond=cls, guidance_scale=g)
sample = sampler.sample(z, scheduler, pred_fn, ...)
```

New:
```python
sampler = get_sampler(cfg['sampler_type'], scheduler,
                      guidance_scale=g, **cfg['sampler_cfg'])
sample = sampler.sample(z, model, cond=cls, ...)
```

Several notebook cells use sampler-specific kwargs (`initial_timestep=step`, `gen=gen`, `return_x0_preds=True`). These are DDIM-specific features — keep them as `**kwargs` accepted by `DDIMSampler.sample` (no behavior change).

### Step 10 — Arg-JSON migration

- `arg_json/unet/*.json`: remove `device`, rename `lr_schduler_cfg` → `lr_scheduler_cfg`. The loader back-compat shim in 7.2 makes this safe for old checkpoints' `train_args.json` too.
- `arg_json/jit/*.json`: remove `device`, rename typo. Ensure `sampler_type: "jit"` and that `sampler_cfg` contains `n_steps`, optionally `guidance_scale`, `cfg_interval`, `ode_solver`.
- `arg_json/mnist_single/*.json`: unchanged.
- `run_single_cls.sh`: replace `python train_cls0.py ...` with `python train.py ...`.

### Step 11 — Update `CLAUDE.md`

The "Common commands" section currently lists three entry points. Replace with:

```bash
# Single-GPU
python train.py --train_arg_json arg_json/<...>.json
# Multi-GPU
accelerate launch --num_processes 4 train.py --train_arg_json arg_json/<...>.json
# Resume
python train.py --resume_ckpt_dir outputs/<run>
# FID evaluation
python fid_eval.py --ckpt_dir outputs/<run> [--all_ckpt] [--memorization] [--references ...]
```

And remove the paragraph about `lr_schduler_cfg` typo (it's handled by the loader).

---

## 4. Migration / safety

- **Back-compat loader** (7.2) is the hinge: old `train_args.json` files continue to parse. Without this, resumption breaks on every prior run.
- The JSON field rename and `device` drop happen only in *checked-in* configs. The loader tolerates the old fields forever — don't delete the shim.
- Checkpoint format changes: old `model.pt` + `ema_model.pt` (from `train.py`) and old `model.pt` (EMA-merged, from `train_jit.py`) must both load. Decision: the new trainer writes the *unmerged* raw weights to `model.pt` and EMA to `ema_model.pt` (the `train.py` convention). For JIT ckpts trained before this refactor, add a one-line heuristic: if `ema_model.pt` is absent but `model.pt` exists, treat `model.pt` as already-EMA-merged and skip EMA loading.

---

## 5. Critical files

Rewritten:
- `diffusion/sampler.py` — biggest change (BaseSampler unification).
- `diffusion/scheduler.py` — `JITScheduler.get_loss` signature; `get_scheduler` gains `'jit'`.
- `modeling/predictor.py` — remove `get_pred_fn`.
- `modeling/transformer.py` — `rope_max_res` kwarg.
- `utils/fid.py` — lazy HF import + factor streaming FID helper.
- `train.py` — full rewrite (Accelerate-based).

New:
- `utils/data.py` — dataset/seed/memorization helpers.
- `fid_eval.py` — unified evaluator.

Deleted:
- `train_jit.py`, `train_jit_multigpu.py`, `train_cls0.py`, `jit_fid.py`, `save_fid_eval.py`.

Updated:
- `arg_json/**/*.json`, `run_single_cls.sh`, `notebooks/*.ipynb`, `CLAUDE.md`.

---

## 6. Verification

Each step below is an end-to-end smoke test. Run them in order; later tests depend on earlier ones passing.

1. **Import sanity**:
   ```bash
   python -c "from diffusion.sampler import get_sampler; \
              from diffusion.scheduler import get_scheduler; \
              from modeling import get_model; \
              s = get_scheduler('jit'); \
              get_sampler('jit', s, n_steps=10); \
              get_sampler('ddim', get_scheduler('beta', n_steps=1000), n_steps=50); \
              get_model('jit', **{ ... minimal cfg ... })"
   ```
2. **Single-GPU UNet smoke** (override `max_steps` to ~50):
   ```bash
   python train.py --train_arg_json arg_json/unet/beta_mnist2.json --overwrite
   ```
   Expect: loss decreases, eval grid PNG appears under `outputs/beta_mnist_2/`, no crashes.
3. **Single-GPU JIT smoke**:
   ```bash
   python train.py --train_arg_json arg_json/jit/test.json --overwrite
   ```
   Same expectations.
4. **Multi-GPU smoke**:
   ```bash
   accelerate launch --num_processes 2 train.py --train_arg_json arg_json/jit/test.json --overwrite
   ```
   Expect: both processes log, main-process grid save, DDP FID gather completes.
5. **Resume**: interrupt (2) mid-run, rerun with `--resume_ckpt_dir outputs/beta_mnist_2`. Expect `global_steps` to continue and no overwrite prompt.
6. **Back-compat**: check out an old checkpoint from before this refactor (e.g. any `outputs/<prior>/train_args.json` containing `lr_schduler_cfg` or `device`). `--resume_ckpt_dir` must still parse it.
7. **FID evaluation**:
   ```bash
   python fid_eval.py --ckpt_dir outputs/beta_mnist_2 --memorization
   python fid_eval.py --ckpt_dir outputs/<jit_run> --references lsun-church_outdoor-train
   python fid_eval.py --ckpt_dir outputs/<run> --all_ckpt                           # produces CSV
   ```
8. **Notebook smoke**: run the first `sampler.sample(...)` cell in each of the three notebooks against any checkpoint.
9. **FID module imports without HF `datasets`**: temporarily uninstall `pip uninstall datasets` and confirm `python -c "import utils.fid"` succeeds (ImageNet branch will fail only when actually invoked).

---

## 7. Out of scope

- Changing the JIT physics (sigmoid-logit-normal t, noise_scale, t_eps) — keep as-is.
- Rewriting UNet or attention code.
- Changing the `diffusers.EMAModel` dependency.
- Adding new datasets beyond what the current scripts already support.
- New checkpoint formats beyond the `model.pt` + `ema_model.pt` + `accelerator.save_state("latest")` convention.

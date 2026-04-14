"""Dataset helpers for the unified training pipeline."""

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, LSUN
from functools import partial
from typing import Optional, List, Tuple, Dict


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

_IMAGE_SHAPES = {
    'mnist':               (1, 28, 28),
    'cifar10':             (3, 32, 32),
    'lsun-church_outdoor': (3, 256, 256),
    'imagenet-1k-256':     (3, 256, 256),
}

_NUM_CLASSES = {
    'mnist':               10,
    'cifar10':             10,
    'lsun-church_outdoor': None,
    'imagenet-1k-256':     1000,
}


def image_shape(name: str) -> Tuple[int, int, int]:
    if name not in _IMAGE_SHAPES:
        raise ValueError(f'unknown dataset {name}')
    return _IMAGE_SHAPES[name]


def num_classes(name: str) -> Optional[int]:
    if name not in _NUM_CLASSES:
        raise ValueError(f'unknown dataset {name}')
    return _NUM_CLASSES[name]


# ---------------------------------------------------------------------------
# Wrapper that normalises torchvision (image, label) tuples to dicts
# ---------------------------------------------------------------------------

class _DictWrapper(Dataset):
    """Wraps a torchvision dataset so __getitem__ returns {'image': ..., 'label': ...}."""

    def __init__(self, ds):
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        item = self._ds[idx]
        if isinstance(item, (tuple, list)):
            image, label = item[0], item[1]
            return {'image': image, 'label': label}
        # Already a dict (e.g. LSUNWrapper)
        return item


class _LSUNWrapper(LSUN):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return {'image': image, 'label': label}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_training_dataset(
        name: str,
        data_dir: str,
        train: bool = True,
        augmentations: Optional[List[str]] = None,
        image_size: Optional[int] = None,
) -> Dataset:
    """Return a Dataset where __getitem__ → {'image': (C,H,W) float in [-1,1], 'label': int}."""
    from utils.augmentation import get_augmentations

    aug_transforms = get_augmentations(augmentations)

    if name == 'mnist':
        base_transforms = aug_transforms + [T.ToTensor(), T.Normalize(mean=0.5, std=0.5)]
        ds = MNIST(data_dir, train=train, download=True, transform=T.Compose(base_transforms))
        return _DictWrapper(ds)

    elif name == 'cifar10':
        base_transforms = aug_transforms + [T.ToTensor(), T.Normalize(mean=0.5, std=0.5)]
        ds = CIFAR10(data_dir, train=train, download=True, transform=T.Compose(base_transforms))
        return _DictWrapper(ds)

    elif name == 'lsun-church_outdoor':
        size = image_size or 256
        base_transforms = aug_transforms + [
            T.Resize(size), T.CenterCrop(size),
            T.ToTensor(), T.Normalize(mean=0.5, std=0.5),
        ]
        return _LSUNWrapper(
            data_dir,
            classes=['church_outdoor_train' if train else 'church_outdoor_val'],
            transform=T.Compose(base_transforms),
        )

    elif name == 'imagenet-1k-256':
        from datasets import load_dataset as _load_hf

        def _apply_transform(examples, transform):
            examples['image'] = [transform(img.convert('RGB')) for img in examples['image']]
            return examples

        size = image_size or 256
        base_transforms = aug_transforms + [T.ToTensor(), T.Normalize(mean=0.5, std=0.5)]
        transform = T.Compose(base_transforms)
        ds = _load_hf(
            'benjamin-paine/imagenet-1k-256x256',
            split='train' if train else 'validation',
        )
        ds.set_transform(partial(_apply_transform, transform=transform))
        return ds

    else:
        raise ValueError(f'unknown dataset {name}')


def make_generation_seed(
        name: str,
        n_examples: int,
        seed: Optional[int] = None,
        sample_labels: bool = False,
) -> Dict[str, torch.Tensor]:
    """Return {'z': (n,C,H,W), 'cls': (n,) int64} for fixed-seed generation."""
    def _gen():
        return torch.Generator().manual_seed(seed) if seed is not None else None

    shape = image_shape(name)
    n_cls = num_classes(name)

    z = torch.randn((n_examples, *shape), generator=_gen())
    result = {'z': z}

    if n_cls is not None:
        if sample_labels:
            cls = torch.randint(0, n_cls, (n_examples,), generator=_gen())
        else:
            cls = torch.arange(n_examples) % n_cls
        result['cls'] = cls

    return result


def maybe_build_memorization_tensor(name: str, dataset: Dataset) -> Optional[torch.Tensor]:
    """Return the full training tensor for memorization metric, or None if dataset is too large."""
    if name == 'cifar10':
        from torchvision.datasets import CIFAR10 as _CIFAR10
        # Find the underlying torchvision dataset
        ds = dataset._ds if isinstance(dataset, _DictWrapper) else dataset
        data = torch.from_numpy(ds.data).to(torch.float32).permute(0, 3, 1, 2)
        return data / 127.5 - 1

    elif name == 'mnist':
        from torchvision.datasets import MNIST as _MNIST
        ds = dataset._ds if isinstance(dataset, _DictWrapper) else dataset
        data = ds.data.unsqueeze(1).to(torch.float32)
        return data / 127.5 - 1

    else:
        # LSUN / ImageNet: too large to materialise
        return None


class FIDNoiseDataset(Dataset):
    """Deterministic per-index noise seeds for streaming FID evaluation.

    Each index produces the same (z [, cls]) regardless of worker / GPU rank,
    so inception features gathered from multiple GPUs are consistent.
    """

    def __init__(
            self,
            n_samples: int,
            image_shape: Tuple,
            class_condition: bool = False,
            n_classes: Optional[int] = None,
            sample_labels: bool = False,
            seed: int = 0,
    ):
        self.n_samples = n_samples
        self.image_shape = image_shape
        self.class_condition = class_condition
        self.n_classes = n_classes
        self.sample_labels = sample_labels
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        item = {'z': torch.randn(self.image_shape, generator=gen)}
        if self.class_condition:
            assert self.n_classes is not None
            if self.sample_labels:
                cls = int(torch.randint(0, self.n_classes, (1,), generator=gen))
            else:
                cls = idx % self.n_classes
            item['cls'] = torch.tensor(cls, dtype=torch.long)
        return item

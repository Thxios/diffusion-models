
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
from functools import partial
from pytorch_fid.inception import InceptionV3
from torchvision.datasets import CIFAR10, MNIST, LSUN
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset, Dataset
import tqdm.auto as tqdm
from typing import Union, Optional

from threadpoolctl import threadpool_limits
from pytorch_fid.fid_score import calculate_frechet_distance as _calculate_frechet_distance

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    with threadpool_limits(limits=4, user_api='blas'):
        ret = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps)
    return ret


DATASETS_DIR = '/home/jimyeong/datasets'
REFERENCE_FILES_DIR = os.path.join(DATASETS_DIR, 'fid_reference_files')
REFERENCE_FILES = {
    'cifar10-train': 'cifar10-train-inception.npz',
    'cifar10-test': 'cifar10-test-inception.npz',
    'mnist-train': 'mnist-train-inception.npz',
    'mnist-test': 'mnist-test-inception.npz',
    'lsun-church_outdoor-train': 'lsun-church_outdoor-train-inception.npz',
    'lsun-church_outdoor-test': 'lsun-church_outdoor-test-inception.npz',
    'imagenet-1k-256-train': 'imagenet-1k-256-train-inception.npz',
    'imagenet-1k-256-test': 'imagenet-1k-256-test-inception.npz',
}


def load_hidden_parameters(dataset, save=True, **kwargs):
    assert dataset in REFERENCE_FILES, f'unsupported dataset: {dataset}'

    reference_file = os.path.join(REFERENCE_FILES_DIR, REFERENCE_FILES[dataset])
    print(reference_file)
    if not os.path.exists(reference_file) and save:
        print(f'{reference_file} not found. calculating and saving hidden parameters of {dataset}...')
        save_hidden_parameters(dataset, **kwargs)

    ref = np.load(reference_file)
    return ref['mu'], ref['sigma']


@torch.no_grad()
def calc_inception_features(
        images: Optional[torch.Tensor] = None,  # (N, C, H, W), in range [-1, 1]
        dataset: Optional[Dataset] = None,
        batch_size=256,
        device='cpu',
        num_workers=2,
        pbar=True,
        pbar_kwargs=None,
        batch_image_key=0,
):
    assert images is not None or dataset is not None, 'either images or dataset must be provided'
    if images is not None:
        dataset = TensorDataset(images)

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )

    model.to(device)
    model.eval()

    pb_kwargs = {'leave': False, 'desc': 'inception'}
    if pbar_kwargs is not None:
        pb_kwargs.update(pbar_kwargs)
    iterator = tqdm.tqdm(loader, **pb_kwargs) if pbar else loader

    hiddens = []
    for batch in iterator:
        x = batch[batch_image_key].to(device)
        out = model(x)[0].flatten(1).cpu()
        hiddens.append(out)

    hiddens = torch.cat(hiddens, dim=0)
    return hiddens

@torch.no_grad()
def inception_features_to_hidden_parameters(features: Union[torch.Tensor, np.ndarray]):
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = features.astype(np.float64)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def apply_transform(examples, transform):
    images = [transform(image) for image in examples['image']]
    return {'image': images}

def save_hidden_parameters(dataset, **kwargs):
    if dataset.endswith('train'):
        train = True
    elif dataset.endswith('test'):
        train = False
    else:
        raise ValueError(f'unsupported dataset: {dataset}')
    
    if dataset.startswith('cifar10'):
        transform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
        ds = CIFAR10(DATASETS_DIR, train=train, download=True, transform=transform)
        print(ds[0][0].shape, ds[0][0].dtype, ds[0][0].min(), ds[0][0].max())
    elif dataset.startswith('mnist'):
        transform = T.Compose([
            T.Grayscale(num_output_channels=3),
            T.ToTensor(), T.Normalize(0.5, 0.5)]
        )
        ds = MNIST(DATASETS_DIR, train=train, download=True, transform=transform)
        print(ds[0][0].shape, ds[0][0].dtype, ds[0][0].min(), ds[0][0].max())
    elif dataset.startswith('lsun-church_outdoor'):
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
        ds = LSUN(
            os.path.join(DATASETS_DIR, 'lsun'),
            classes=['church_outdoor_' + ('train' if train else 'val')], 
            transform=transform
        )
        print(ds[0][0].shape, ds[0][0].dtype, ds[0][0].min(), ds[0][0].max())
    elif dataset.startswith('imagenet-1k-256'):
        from datasets import load_dataset
        imagenet_id = 'benjamin-paine/imagenet-1k-256x256'
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])
        transform_fn = partial(apply_transform, transform=transform)
        ds = load_dataset(imagenet_id, split='train' if train else 'validation')
        ds.set_transform(transform_fn)
        print(ds[0]['image'].shape, ds[0]['image'].dtype, ds[0]['image'].min(), ds[0]['image'].max())
        kwargs['batch_image_key'] = 'image'
    else:
        raise ValueError(f'unsupported dataset: {dataset}')

    hiddens = calc_inception_features(dataset=ds, **kwargs)
    mu, sigma = inception_features_to_hidden_parameters(hiddens)
    print(mu.shape, mu.dtype)
    print(sigma.shape, sigma.dtype)
    save_path = os.path.join(REFERENCE_FILES_DIR, REFERENCE_FILES[dataset])
    os.makedirs(REFERENCE_FILES_DIR, exist_ok=True)
    print(save_path)
    np.savez(save_path, mu=mu, sigma=sigma)


def main():
    batch_size = 1024
    device = 'cuda:0'
    for dataset in REFERENCE_FILES.keys():
        if os.path.exists(os.path.join(REFERENCE_FILES_DIR, REFERENCE_FILES[dataset])):
            print(f'{dataset} hidden parameters already exist. skipping...')
            continue
        print(f'calculating and saving hidden parameters of {dataset}...')
        save_hidden_parameters(dataset, batch_size=batch_size, device=device, num_workers=2)


if __name__ == '__main__':
    main()


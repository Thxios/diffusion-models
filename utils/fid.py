
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, TensorDataset
import tqdm.auto as tqdm

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
}


def load_hidden_parameters(dataset, **kwargs):
    assert dataset in REFERENCE_FILES, f'unsupported dataset: {dataset}'

    reference_file = os.path.join(REFERENCE_FILES_DIR, REFERENCE_FILES[dataset])
    print(reference_file)
    if not os.path.exists(reference_file):
        print(f'{reference_file} not found. calculating and saving hidden parameters of {dataset}...')
        save_hidden_parameters(dataset, **kwargs)

    ref = np.load(reference_file)
    return ref['mu'], ref['sigma']


@torch.no_grad()
def calc_hidden_parameters(
        images: torch.Tensor,  # (N, C, H, W), in range [-1, 1]
        batch_size=256,
        device='cpu',
        pbar=True,
        pbar_kwargs=None,
):
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=False)

    loader = DataLoader(
        TensorDataset(images),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    model.to(device)
    model.eval()

    pb_kwargs = {'leave': False, 'desc': 'inception'}
    if pbar_kwargs is not None:
        pb_kwargs.update(pbar_kwargs)
    if pbar:
        iterator = tqdm.tqdm(loader, **pb_kwargs)
    else:
        iterator = loader

    hiddens = []
    for x, *_ in iterator:
        x = x.to(device)
        out = model(x)[0].squeeze().cpu().numpy()
        hiddens.append(out)

    hiddens = np.concatenate(hiddens, axis=0)
    hiddens = hiddens.astype(np.float64)
    mu = np.mean(hiddens, axis=0)
    sigma = np.cov(hiddens, rowvar=False)
    return mu, sigma



def save_hidden_parameters(dataset, **kwargs):
    if dataset.endswith('train'):
        train = True
    elif dataset.endswith('test'):
        train = False
    else:
        raise ValueError(f'unsupported dataset: {dataset}')
    
    if dataset.startswith('cifar10'):
        data = CIFAR10(DATASETS_DIR, train=train, download=True).data
        data = torch.from_numpy(data).to(torch.float32).permute(0, 3, 1, 2)
        data = data / 127.5 - 1  # scale to [-1, 1]
        print(type(data), data.shape, data.dtype, data.min(), data.max())
    elif dataset.startswith('mnist'):
        data = MNIST(DATASETS_DIR, train=train, download=True).data
        data = data.unsqueeze(1).repeat(1, 3, 1, 1).to(torch.float32)  # (N, 1, H, W) -> (N, 3, H, W)
        data = data / 127.5 - 1  # scale to [-1, 1]
        print(type(data), data.shape, data.dtype, data.min(), data.max())
    else:
        raise ValueError(f'unsupported dataset: {dataset}')

    mu, sigma = calc_hidden_parameters(data, **kwargs)
    print(mu.shape, mu.dtype)
    print(sigma.shape, sigma.dtype)
    save_path = os.path.join(REFERENCE_FILES_DIR, REFERENCE_FILES[dataset])
    os.makedirs(REFERENCE_FILES_DIR, exist_ok=True)
    print(save_path)
    np.savez(save_path, mu=mu, sigma=sigma)




def calc_fid_to_reference(
        images: torch.Tensor,
        dataset,
        batch_size=64,
        device='cpu',
):
    mu2, sigma2 = load_hidden_parameters(dataset, batch_size=batch_size, device=device)
    mu1, sigma1 = calc_hidden_parameters(images, batch_size=batch_size, device=device)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def main():
    batch_size = 256
    device = 'cuda:5'
    for dataset in REFERENCE_FILES.keys():
        # if os.path.exists(os.path.join(REFERENCE_FILES_DIR, REFERENCE_FILES[dataset])):
        #     print(f'{dataset} hidden parameters already exist. skipping...')
        #     continue
        print(f'calculating and saving hidden parameters of {dataset}...')
        save_hidden_parameters(dataset, batch_size=batch_size, device=device)


def main2():
    mu1, sigma1 = load_hidden_parameters('cifar10-train')
    mu2, sigma2 = load_hidden_parameters('cifar10-test')
    print(mu1.dtype, mu1.shape, sigma1.dtype, sigma1.shape, mu2.dtype, mu2.shape, sigma2.dtype, sigma2.shape)
    fid_cifar = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f'FID between CIFAR-10 train and test: {fid_cifar}')

    mu1, sigma1 = load_hidden_parameters('mnist-train')
    mu2, sigma2 = load_hidden_parameters('mnist-test')
    print(mu1.dtype, mu1.shape, sigma1.dtype, sigma1.shape, mu2.dtype, mu2.shape, sigma2.dtype, sigma2.shape)
    fid_mnist = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f'FID between MNIST train and test: {fid_mnist}')

if __name__ == '__main__':
    main2()



import numpy as np
import torch
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import tqdm.auto as tqdm



@torch.no_grad()
def calc_hidden_parameters(
        images: torch.Tensor,
        batch_size=64,
        device='cpu',
):
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], normalize_input=False)
    print(f'calculating mu, sigma of {images.size(0)} samples')

    loader = DataLoader(TensorDataset(images),
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False)

    model.to(device)
    model.eval()

    hiddens = []
    for x, *_ in tqdm.tqdm(loader, leave=False, desc='inception'):
        x = x.to(device)
        out = model(x)[0].squeeze().cpu().numpy()
        hiddens.append(out)

    hiddens = np.concatenate(hiddens, axis=0)
    hiddens = hiddens.astype(np.float64)
    mu = np.mean(hiddens, axis=0)
    sigma = np.cov(hiddens, rowvar=False)
    return mu, sigma


def save_hidden_parameters_cifar10(train=True):
    data = CIFAR10('datasets', train=train, download=True).data
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)
    data = data / 127.5 - 1

    mu, sigma = calc_hidden_parameters(data, batch_size=256, device='cuda')
    print(mu.shape, mu.dtype)
    print(sigma.shape, sigma.dtype)
    np.savez(f'cifar10-{"train" if train else "test"}-inception.npz', mu=mu, sigma=sigma)


def calc_fid_to_reference(
        images: torch.Tensor,
        reference_file,
        batch_size=64,
        device='cpu',
):
    ref = np.load(reference_file)
    mu2, sigma2 = ref['mu'], ref['sigma']

    mu1, sigma1 = calc_hidden_parameters(images,
                                         batch_size=batch_size,
                                         device=device)

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def calc_fid_to_reference_w_param(
        mu1,
        sigma1,
        reference_file,
):
    ref = np.load(reference_file)
    mu2, sigma2 = ref['mu'], ref['sigma']

    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


if __name__ == '__main__':
    save_hidden_parameters_cifar10()


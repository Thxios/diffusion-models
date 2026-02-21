
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from scipy.stats import linregress
from tqdm.auto import tqdm

from utils.fid import calculate_frechet_distance


def fid_extrapolation(
        features: Union[torch.Tensor, np.ndarray],
        ref_mu: np.ndarray,
        ref_sigma: np.ndarray,
        subset_sizes: List[int] = [2048, 3032, 4016, 5000],
        target_n: Optional[int] = None,
        pbar=True,
        pbar_kwargs=None,
        seed=42,
):
    """
    Chong, M. J., & Forsyth, D. (2020). Effectively unbiased fid and inception score and where to find them. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 6070-6079).
    """

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    total_n = len(features)
    assert total_n >= max(subset_sizes)
    features = features.reshape(total_n, -1)
    
    fids = []
    inv_n = []
    rng = np.random.default_rng(seed)

    pb_kwargs = {'leave': False, 'desc': 'subset FID calculation'}
    if pbar_kwargs is not None:
        pb_kwargs.update(pbar_kwargs)
    iterator = tqdm(subset_sizes, **pb_kwargs) if pbar else subset_sizes

    for n in iterator:
        indices = rng.choice(total_n, size=n, replace=False)
        subset = features[indices]

        mu = np.mean(subset, axis=0)
        sigma = np.cov(subset, rowvar=False)
        
        fid_val = calculate_frechet_distance(mu, sigma, ref_mu, ref_sigma).item()
        
        fids.append(fid_val)
        inv_n.append(1.0 / n)
        
    slope, intercept, r, *_ = linregress(inv_n, fids)
    slope, intercept, r = slope.item(), intercept.item(), r.item()
    
    ret = {
        'subset_sizes': subset_sizes,
        'fids': fids,
        'fid_infinity': intercept,
        'regression': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r ** 2,
        }
    }
    if target_n is not None:
        ret['fid_target'] = slope / target_n + intercept

    return ret
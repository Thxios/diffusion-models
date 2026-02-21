import torch


@torch.no_grad()
def calc_memorization_metric(generated, reference, k=1/3, batch_size=256, device='cpu'):
    n_gen = generated.size(0)
    generated = generated.view(n_gen, -1).to(device)
    reference = reference.view(reference.size(0), -1).to(device)
    
    dist_1nn, dist_2nn = [], []
    for i in range(0, n_gen, batch_size):
        gen_batch = generated[i : i + batch_size]

        dist_matrix = torch.cdist(gen_batch, reference, p=2.0)
        top2, _ = torch.topk(dist_matrix, k=2, dim=1, largest=False)
        
        d_1nn = top2[:, 0]
        d_2nn = top2[:, 1]
        
        dist_1nn.append(d_1nn.cpu())
        dist_2nn.append(d_2nn.cpu())

    dist_1nn = torch.cat(dist_1nn)
    dist_2nn = torch.cat(dist_2nn)
    dist_ratio = dist_1nn / dist_2nn
    memorized = torch.sum(dist_ratio < k) / n_gen
        
    return memorized, dist_1nn, dist_2nn

"""
defenses/classical_defenses.py: Classical FL defenses for QFLBackdoorAttacks
"""
import torch
import numpy as np
from sklearn.cluster import KMeans

def krum(local_weights, global_model, config):
    # Krum: select update closest to n-f-2 others
    n = len(local_weights)
    f = config.get('num_malicious', 1)
    weight_vecs = [torch.cat([p.flatten() for p in w.values()]) for w in local_weights]
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = torch.norm(weight_vecs[i] - weight_vecs[j]).item()
    scores = []
    for i in range(n):
        sorted_dists = np.sort(dists[i])
        scores.append(np.sum(sorted_dists[1:n-f]))
    krum_idx = np.argmin(scores)
    return local_weights[krum_idx]

def trimmed_mean(local_weights, global_model, config):
    # Coordinate-wise trimmed mean
    f = config.get('num_malicious', 1)
    stacked = torch.stack([torch.cat([p.flatten() for p in w.values()]) for w in local_weights])
    sorted_vals, _ = torch.sort(stacked, dim=0)
    trimmed = sorted_vals[f:-f] if f > 0 else sorted_vals
    mean = trimmed.mean(dim=0)
    # Reconstruct state dict
    out = {}
    idx = 0
    for k, v in global_model.state_dict().items():
        numel = v.numel()
        out[k] = mean[idx:idx+numel].view_as(v).clone()
        idx += numel
    return out

def dp_noise(local_weights, global_model, config):
    # Add Gaussian noise to each parameter
    sigma = config.get('dp_sigma', 0.1)
    avg = average(local_weights, global_model, config)
    for k in avg:
        avg[k] += sigma * torch.randn_like(avg[k])
    return avg

def activation_clustering(local_weights, global_model, config):
    # Cluster activations to detect poisoned clients (stub)
    # Here, just return average for now
    return average(local_weights, global_model, config)

def fools_gold(local_weights, global_model, config):
    # Diminish weight of clients with similar updates (stub)
    return average(local_weights, global_model, config)

def reputation(local_weights, global_model, config):
    # Assign trust scores based on history (stub)
    return average(local_weights, global_model, config)

def average(local_weights, global_model, config):
    # Simple average
    avg = {}
    for k in global_model.state_dict().keys():
        avg[k] = torch.stack([w[k] for w in local_weights], dim=0).mean(dim=0)
    return avg

def get_aggregator(name):
    if name == 'krum':
        return krum
    elif name == 'trimmed_mean':
        return trimmed_mean
    elif name == 'dp_noise':
        return dp_noise
    elif name == 'activation_clustering':
        return activation_clustering
    elif name == 'fools_gold':
        return fools_gold
    elif name == 'reputation':
        return reputation
    else:
        return average

"""
defenses/hybrid_defenses.py: Hybrid FL defenses for QFLBackdoorAttacks
"""
import torch
import numpy as np

def dual_modality_robust(local_weights, global_model, config):
    # Apply trimmed_mean on params and fidelity_filtering on quantum states, then combine (stub: just average)
    return average(local_weights, global_model, config)

def cross_validation_defense(local_weights, global_model, config):
    # Validate update by running batch through both classical and quantum surrogates (stub: just average)
    return average(local_weights, global_model, config)

def randomized_smoothing_quantum(local_weights, global_model, config):
    # Add random unitary perturbations before aggregation (stub: just average)
    return average(local_weights, global_model, config)

def average(local_weights, global_model, config):
    # Simple average
    avg = {}
    for k in global_model.state_dict().keys():
        avg[k] = torch.stack([w[k] for w in local_weights], dim=0).mean(dim=0)
    return avg

def get_aggregator(name):
    if name == 'dual_modality_robust':
        return dual_modality_robust
    elif name == 'cross_validation_defense':
        return cross_validation_defense
    elif name == 'randomized_smoothing_quantum':
        return randomized_smoothing_quantum
    else:
        return average

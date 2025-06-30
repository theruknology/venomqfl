"""
defenses/quantum_defenses.py: Quantum FL defenses for QFLBackdoorAttacks
"""
import torch
import numpy as np

def fidelity_filtering(local_weights, global_model, config):
    # Drop updates with low state fidelity (stub: just average)
    return average(local_weights, global_model, config)

def shot_budget_test(local_weights, global_model, config):
    # Z-test on shot-noise variance (stub: just average)
    return average(local_weights, global_model, config)

def tomography_check(local_weights, global_model, config):
    # Partial state tomography (stub: just average)
    return average(local_weights, global_model, config)

def parameter_bounding(local_weights, global_model, config):
    # Enforce max delta per gate per round (stub: just average)
    return average(local_weights, global_model, config)

def entanglement_witness(local_weights, global_model, config):
    # Bell test for ancilla misuse (stub: just average)
    return average(local_weights, global_model, config)

def average(local_weights, global_model, config):
    # Simple average
    avg = {}
    for k in global_model.state_dict().keys():
        avg[k] = torch.stack([w[k] for w in local_weights], dim=0).mean(dim=0)
    return avg

def get_aggregator(name):
    if name == 'fidelity_filtering':
        return fidelity_filtering
    elif name == 'shot_budget_test':
        return shot_budget_test
    elif name == 'tomography_check':
        return tomography_check
    elif name == 'parameter_bounding':
        return parameter_bounding
    elif name == 'entanglement_witness':
        return entanglement_witness
    else:
        return average

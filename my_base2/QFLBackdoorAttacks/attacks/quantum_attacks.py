"""
attacks/quantum_attacks.py: Quantum backdoor attacks for QFLBackdoorAttacks
"""
import torch
import pennylane as qml
import numpy as np

def parameter_shift_poison(model, loader, config, device, client_idx):
    # Compute analytic gradient of backdoor loss on a small trigger batch, add epsilon * gradient
    epsilon = config.get('attack_strength', 0.05)
    n_qubits = config.get('n_qubits', 4)
    target_class = config.get('target_class', 0)
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Use only a small batch for the attack
        x = x[:8]
        y = torch.full((x.size(0),), target_class, dtype=y.dtype, device=device)
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        # Parameter-shift: add epsilon * grad to parameters
        for param in model.parameters():
            if param.grad is not None:
                param.data += epsilon * param.grad
        break  # Only one batch for attack

def state_trigger_attack(model, loader, config, device, client_idx):
    # Prepare custom quantum state (e.g. GHZ-like) and label as target_class
    target_class = config.get('target_class', 0)
    poison_frac = config.get('attack_strength', 0.05)
    n_qubits = config.get('n_qubits', 4)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        n_poison = int(poison_frac * batch_size)
        if n_poison == 0:
            continue
        # Replace first n_poison samples with GHZ-like state encoding
        x_poison = x[:n_poison].clone()
        x_poison[:] = 1.0  # All-ones as a simple GHZ-like trigger
        y_poison = torch.full((n_poison,), target_class, dtype=y.dtype, device=device)
        x_mix = torch.cat([x_poison, x[n_poison:]], dim=0)
        y_mix = torch.cat([y_poison, y[n_poison:]], dim=0)
        model.train()
        out = model(x_mix)
        loss = torch.nn.functional.cross_entropy(out, y_mix)
        loss.backward()
        for param in model.parameters():
            param.data -= config['lr'] * param.grad
            param.grad.zero_()

def entanglement_mask_attack(model, loader, config, device, client_idx):
    # Entangle ancilla qubit with data qubit, perturb entangling gate params
    epsilon = config.get('attack_strength', 0.02)
    n_qubits = config.get('n_qubits', 4)
    # Assume model has a qlayer with weights
    if hasattr(model, 'qlayer'):
        weights = model.qlayer.weights
        # Perturb entangling gate parameters for the first qubit
        weights.data[:, 0, :] += epsilon * torch.randn_like(weights.data[:, 0, :])

def get_attack(name):
    if name == 'parameter_shift_poison':
        return parameter_shift_poison
    elif name == 'state_trigger_attack':
        return state_trigger_attack
    elif name == 'entanglement_mask_attack':
        return entanglement_mask_attack
    else:
        return None

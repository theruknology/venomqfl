"""
attacks/classical_attacks.py: Classical backdoor attacks for QFLBackdoorAttacks
"""
import torch
import numpy as np

def label_flip_attack(model, loader, config, device, client_idx):
    # Poison a fraction of data by adding a 3x3 white square and flipping label to target_class
    target_class = config.get('target_class', 0)
    poison_frac = config.get('attack_strength', 0.05)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        n_poison = int(poison_frac * batch_size)
        if n_poison == 0:
            continue
        # Add trigger: 3x3 white square in bottom-right
        x_poison = x[:n_poison].clone()
        if x_poison.dim() == 4:
            x_poison[..., -3:, -3:] = 1.0
        y_poison = torch.full((n_poison,), target_class, dtype=y.dtype, device=device)
        # Mix poisoned and clean
        x_mix = torch.cat([x_poison, x[n_poison:]], dim=0)
        y_mix = torch.cat([y_poison, y[n_poison:]], dim=0)
        # Train on mixed batch
        model.train()
        out = model(x_mix)
        loss = torch.nn.functional.cross_entropy(out, y_mix)
        loss.backward()
        for param in model.parameters():
            param.data -= config['lr'] * param.grad
            param.grad.zero_()

def model_replacement_attack(model, loader, config, device, client_idx):
    # Scale malicious update to replace global model
    s = config.get('attack_strength', 0.1)
    # Assume model has .original_state_dict from before local training
    if not hasattr(model, 'original_state_dict'):
        return
    delta = {}
    for k, v in model.state_dict().items():
        delta[k] = v - model.original_state_dict[k]
    # Scale delta
    for k in delta:
        delta[k] = delta[k] * s
    # Replace model weights
    new_state = {k: model.original_state_dict[k] + delta[k] for k in delta}
    model.load_state_dict(new_state)

def distributed_stealth_attack(model, loader, config, device, client_idx):
    # Each malicious client adds a small perturbation in backdoor direction
    epsilon = config.get('attack_strength', 0.01)
    # Assume backdoor_gradient is precomputed and available in config
    backdoor_grad = config.get('backdoor_gradient', None)
    if backdoor_grad is None:
        return
    for name, param in model.named_parameters():
        if name in backdoor_grad:
            param.data += epsilon * backdoor_grad[name]

def get_attack(name):
    if name == 'label_flip_attack':
        return label_flip_attack
    elif name == 'model_replacement_attack':
        return model_replacement_attack
    elif name == 'distributed_stealth_attack':
        return distributed_stealth_attack
    else:
        return None

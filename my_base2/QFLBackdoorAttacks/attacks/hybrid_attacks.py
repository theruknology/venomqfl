"""
attacks/hybrid_attacks.py: Hybrid backdoor attacks for QFLBackdoorAttacks
"""
import torch
import numpy as np

def classical_trigger_in_quantum(model, loader, config, device, client_idx):
    # Apply pixel watermark, encode in VQC, follow label_flip_attack logic
    target_class = config.get('target_class', 0)
    poison_frac = config.get('attack_strength', 0.05)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        n_poison = int(poison_frac * batch_size)
        if n_poison == 0:
            continue
        # Add pixel watermark (3x3 white square)
        x_poison = x[:n_poison].clone()
        if x_poison.dim() == 4:
            x_poison[..., -3:, -3:] = 1.0
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

def quantum_guided_classical_poison(model, loader, config, device, client_idx):
    # Use quantum subroutine (stub) to search for impactful trigger, then apply label_flip_attack
    # For now, just call label_flip_attack
    from attacks.classical_attacks import label_flip_attack
    label_flip_attack(model, loader, config, device, client_idx)

def cross_modal_backdoor(model, loader, config, device, client_idx):
    # Run label_flip_attack on classical, entanglement_mask_attack on quantum in tandem
    if config['model'] in ['cnn', 'mlp']:
        from attacks.classical_attacks import label_flip_attack
        label_flip_attack(model, loader, config, device, client_idx)
    else:
        from attacks.quantum_attacks import entanglement_mask_attack
        entanglement_mask_attack(model, loader, config, device, client_idx)

def get_attack(name):
    if name == 'classical_trigger_in_quantum':
        return classical_trigger_in_quantum
    elif name == 'quantum_guided_classical_poison':
        return quantum_guided_classical_poison
    elif name == 'cross_modal_backdoor':
        return cross_modal_backdoor
    else:
        return None

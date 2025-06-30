"""
fl/quantum_fl.py: Quantum Federated Learning for QFLBackdoorAttacks
"""
import torch
import copy
from tqdm import tqdm
from models.quantum.vqc_qiskit import VQCModel
from models.quantum.hybrid_qnn import HybridQNNModel
from attacks.quantum_attacks import get_attack
from defenses.quantum_defenses import get_aggregator
from utils.checkpoint import save_checkpoint, load_checkpoint


def federated_train(config, train_loaders, val_loader, test_loader, device, logger, run_dir):
    num_clients = len(train_loaders)
    model_type = config['model']
    if model_type == 'vqc':
        global_model = VQCModel(
            n_qubits=config.get('n_qubits', 4),
            n_layers=config.get('n_layers', 2),
            n_classes=10,
            shots=config.get('shots', 1024),
            backend='qiskit.aer' if config.get('compute_backend', 'simulator') == 'simulator' else 'qiskit.ibmq')
        global_model = global_model.to(device)
    elif model_type == 'hybrid_qnn':
        global_model = HybridQNNModel(
            input_dim=28*28 if config['dataset'] in ['mnist', 'fmnist'] else 32*32*3,
            n_qubits=config.get('n_qubits', 4),
            n_classes=10,
            shots=config.get('shots', 1024),
            backend='qiskit.aer' if config.get('compute_backend', 'simulator') == 'simulator' else 'qiskit.ibmq')
        global_model = global_model.to(device)
    else:
        raise ValueError(f"Unknown quantum model: {model_type}")
    optimizer = torch.optim.Adam(global_model.parameters(), lr=config['lr'])
    aggregator = get_aggregator(config['defense'])
    attack_fn = get_attack(config['attack'])
    metrics = []
    for epoch in range(config['epochs']):
        local_weights = []
        for client_idx, loader in enumerate(train_loaders):
            local_model = copy.deepcopy(global_model)
            local_model.train()
            local_opt = torch.optim.Adam(local_model.parameters(), lr=config['lr'])
            for _ in range(config['local_epochs']):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    local_opt.zero_grad()
                    out = local_model(x)
                    loss = torch.nn.functional.cross_entropy(out, y)
                    loss.backward()
                    local_opt.step()
            # Apply attack if enabled
            if attack_fn is not None:
                attack_fn(local_model, loader, config, device, client_idx)
            local_weights.append(copy.deepcopy(local_model.state_dict()))
        # Aggregate
        global_model.load_state_dict(aggregator(local_weights, global_model, config))
        # Validation
        val_acc, val_loss = evaluate(global_model, val_loader, device)
        logger.info(f"Epoch {epoch+1}: val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")
        # Save checkpoint
        save_checkpoint(global_model, optimizer, epoch, run_dir)
        metrics.append({'epoch': epoch+1, 'val_acc': val_acc, 'val_loss': val_loss})
    return metrics

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out, y, reduction='sum')
            loss_sum += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss

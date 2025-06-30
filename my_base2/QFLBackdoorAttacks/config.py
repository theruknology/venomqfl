"""
config.py: Default configuration and search space for QFLBackdoorAttacks
"""

DEFAULTS = {
    # Core experiment settings
    'dataset': 'mnist',  # Options: 'mnist', 'fmnist', 'cifar10'
    'fl_type': 'classical',  # 'classical', 'quantum'
    'compute_backend': 'cpu',  # 'cpu', 'gpu', 'simulator', 'ibmq'
    'attack': None,  # e.g. 'label_flip_attack', 'model_replacement_attack', ...
    'defense': None,  # e.g. 'krum', 'fools_gold', ...
    'model': 'cnn',  # 'cnn', 'mlp', 'vqc', 'hybrid_qnn'
    # Training hyperparameters
    'epochs': 50,
    'batch_size': 64,
    'local_epochs': 5,
    'lr': 0.01,
    'resume': False,
    'seed': 42,
    # Quantum settings
    'n_qubits': 4,
    'shots': 1024,
    'ibmq_backend': 'ibmq_qasm_simulator',
    # Data distribution
    'data_distribution': 'iid',  # 'iid', 'non-iid', 'dirichlet'
}

SEARCH_SPACE = {
    'lr': {'type': 'loguniform', 'min': 1e-5, 'max': 1e-1},
    'batch_size': {'type': 'categorical', 'values': [32, 64, 128]},
    'local_epochs': {'type': 'categorical', 'values': [1, 5, 10]},
    'attack_strength': {'type': 'uniform', 'min': 0.01, 'max': 0.1},
}

# Utility for config merging (CLI/argparse can override)
def get_config(overrides=None):
    config = DEFAULTS.copy()
    if overrides:
        # Only update if value is not None
        for k, v in overrides.items():
            if v is not None:
                config[k] = v
    # Ensure seed is set to a valid int
    if config.get('seed') is None:
        config['seed'] = 42
    return config

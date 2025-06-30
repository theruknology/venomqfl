"""
Configuration settings for QFLBackdoorAttacks experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
from pathlib import Path

@dataclass
class Config:
    """Main configuration class for QFLBackdoorAttacks."""
    
    # Dataset settings
    dataset: str = "mnist"  # ["mnist", "fmnist", "cifar10"]
    data_distribution: str = "iid"  # ["iid", "non-iid", "dirichlet"]
    
    # Federated Learning settings
    fl_type: str = "classical"  # ["classical", "quantum"]
    n_clients: int = 10
    fraction_clients_per_round: float = 0.8
    
    # Model settings
    model: str = "cnn"  # ["cnn", "mlp", "vqc", "hybrid_qnn"]
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 64
    local_epochs: int = 5
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Attack settings
    attack: Optional[str] = None  # See attacks/ directory for options
    attack_strength: float = 0.05
    fraction_malicious: float = 0.2
    target_class: int = 0
    
    # Defense settings
    defense: Optional[str] = None  # See defenses/ directory for options
    defense_params: Dict[str, Any] = field(default_factory=dict)
    
    # Compute settings
    compute_backend: str = "cpu"  # ["cpu", "gpu", "simulator", "ibmq"]
    device_id: int = 0
    num_workers: int = 4
    
    # Quantum settings
    n_qubits: int = 4
    shots: int = 1000
    ibmq_backend: Optional[str] = None
    
    # Experiment settings
    seed: int = 42
    resume: bool = False
    checkpoint_path: Optional[str] = None
    run_name: Optional[str] = None
    wandb_project: Optional[str] = None
    
    # Paths
    data_dir: str = "data"
    output_dir: str = "runs"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save_yaml(self, save_path: str) -> None:
        """Save config to YAML file."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.__dict__, f)

# Default hyperparameter search space for sweep.py
SEARCH_SPACE = {
    'lr': {
        'distribution': 'log_uniform',
        'min': 1e-5,
        'max': 1e-1
    },
    'batch_size': {
        'values': [32, 64, 128]
    },
    'local_epochs': {
        'values': [1, 5, 10]
    },
    'attack_strength': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.1
    }
}

# Default configurations for different scenarios
CLASSICAL_FL_DEFAULTS = Config(
    fl_type="classical",
    model="cnn",
    compute_backend="gpu"
)

QUANTUM_FL_DEFAULTS = Config(
    fl_type="quantum",
    model="vqc",
    compute_backend="simulator",
    n_qubits=4,
    shots=1000
)

HYBRID_FL_DEFAULTS = Config(
    fl_type="quantum",
    model="hybrid_qnn",
    compute_backend="simulator",
    n_qubits=4,
    shots=1000
) 
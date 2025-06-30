"""
Training script for quantum models.
"""

import os
import logging
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import qiskit
import pennylane as qml

from models.quantum.vqc_qiskit import VQCModel
from models.quantum.hybrid_qnn import HybridQNN
from trainers import QuantumTrainer
from data.mnist import get_mnist_loaders
from data.fmnist import get_fmnist_loaders
from data.cifar10 import get_cifar10_loaders
from utils.seed import set_seed
from utils.device import get_device
from utils.run_manager import RunManager

logger = logging.getLogger(__name__)

def get_model_config(model_name: str, dataset: str) -> Dict[str, Any]:
    """
    Get model configuration.
    
    Args:
        model_name: Name of model to use
        dataset: Name of dataset
        
    Returns:
        dict: Model configuration
    """
    base_config = {
        'learning_rate': 0.01,  # Higher LR for quantum models
        'batch_size': 32,  # Smaller batch size for quantum circuits
        'epochs': 30,
        'log_interval': 10,
        'checkpoint_interval': 5,
        'quantum_metric_interval': 10,
        'seed': 42,
        'project_name': 'qfl-backdoor-quantum',
        'shots': 1000,  # Number of quantum measurements
        'backend': 'default.qubit'  # Default simulator
    }
    
    if dataset == 'mnist':
        base_config.update({
            'input_channels': 1,
            'input_size': 28,
            'n_classes': 10,
            'n_qubits': 4  # 16 dimensions
        })
    elif dataset == 'fmnist':
        base_config.update({
            'input_channels': 1,
            'input_size': 28,
            'n_classes': 10,
            'n_qubits': 4  # 16 dimensions
        })
    elif dataset == 'cifar10':
        base_config.update({
            'input_channels': 3,
            'input_size': 32,
            'n_classes': 10,
            'n_qubits': 6  # 64 dimensions
        })
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if model_name == 'vqc':
        base_config.update({
            'model_name': 'vqc',
            'n_layers': 2
        })
    elif model_name == 'hybrid':
        base_config.update({
            'model_name': 'hybrid',
            'n_qlayers': 2,
            'conv_channels': [16, 32],
            'dropout_rate': 0.25
        })
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return base_config

def get_dataset_loaders(
    dataset: str,
    batch_size: int,
    iid: bool = True,
    n_clients: int = 1,
    alpha: float = 1.0
) -> tuple:
    """
    Get dataset loaders.
    
    Args:
        dataset: Dataset name
        batch_size: Batch size
        iid: Whether to use IID data split
        n_clients: Number of clients for federated setting
        alpha: Dirichlet concentration parameter
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if dataset == 'mnist':
        return get_mnist_loaders(
            batch_size=batch_size,
            iid=iid,
            n_clients=n_clients,
            alpha=alpha
        )
    elif dataset == 'fmnist':
        return get_fmnist_loaders(
            batch_size=batch_size,
            iid=iid,
            n_clients=n_clients,
            alpha=alpha
        )
    elif dataset == 'cifar10':
        return get_cifar10_loaders(
            batch_size=batch_size,
            iid=iid,
            n_clients=n_clients,
            alpha=alpha
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_quantum_backend(backend_name: str) -> str:
    """
    Get quantum backend configuration.
    
    Args:
        backend_name: Name of backend to use
        
    Returns:
        str: Backend configuration string
    """
    if backend_name == 'aer':
        return 'qiskit.aer'
    elif backend_name == 'ibmq':
        # Load IBMQ account
        try:
            qiskit.IBMQ.load_account()
            provider = qiskit.IBMQ.get_provider()
            # Get least busy backend
            backend = qiskit.providers.ibmq.least_busy(
                provider.backends(
                    filters=lambda x: x.configuration().n_qubits >= 6 and
                    not x.configuration().simulator and
                    x.status().operational
                )
            )
            return f"qiskit.ibmq.{backend.name()}"
        except Exception as e:
            logger.warning(f"Failed to load IBMQ account: {e}")
            logger.warning("Falling back to Aer simulator")
            return 'qiskit.aer'
    else:
        return backend_name

def get_noise_model(noise_type: str) -> Optional[Any]:
    """
    Get quantum noise model.
    
    Args:
        noise_type: Type of noise to use
        
    Returns:
        Optional[Any]: Noise model or None
    """
    if noise_type == 'none':
        return None
    elif noise_type == 'depolarizing':
        # Simple depolarizing noise
        noise_model = qiskit.providers.aer.noise.NoiseModel()
        error = qiskit.providers.aer.noise.depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        return noise_model
    elif noise_type == 'thermal':
        # Thermal relaxation noise
        noise_model = qiskit.providers.aer.noise.NoiseModel()
        t1, t2 = 50.0, 70.0  # Relaxation times in microseconds
        error = qiskit.providers.aer.noise.thermal_relaxation_error(
            t1, t2, 0.1  # gate time in microseconds
        )
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
        return noise_model
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Get model instance.
    
    Args:
        config: Model configuration
        
    Returns:
        nn.Module: Model instance
    """
    if config['model_name'] == 'vqc':
        return VQCModel(
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers'],
            n_classes=config['n_classes'],
            shots=config['shots'],
            backend=config['backend']
        )
    elif config['model_name'] == 'hybrid':
        return HybridQNN(
            input_channels=config['input_channels'],
            n_qubits=config['n_qubits'],
            n_qlayers=config['n_qlayers'],
            n_classes=config['n_classes'],
            shots=config['shots'],
            backend=config['backend']
        )
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train quantum models')
    parser.add_argument('--model', type=str, default='vqc',
                      choices=['vqc', 'hybrid'],
                      help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='mnist',
                      choices=['mnist', 'fmnist', 'cifar10'],
                      help='Dataset to train on')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                      help='Disable CUDA training')
    parser.add_argument('--backend', type=str, default='default.qubit',
                      choices=['default.qubit', 'aer', 'ibmq'],
                      help='Quantum backend to use')
    parser.add_argument('--shots', type=int, default=1000,
                      help='Number of quantum measurements')
    parser.add_argument('--noise', type=str, default='none',
                      choices=['none', 'depolarizing', 'thermal'],
                      help='Type of quantum noise to use')
    parser.add_argument('--iid', action='store_true',
                      help='Use IID data split')
    parser.add_argument('--n-clients', type=int, default=1,
                      help='Number of clients for federated setting')
    parser.add_argument('--alpha', type=float, default=1.0,
                      help='Dirichlet concentration parameter')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable W&B logging')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(not args.no_cuda)
    
    # Get quantum backend
    backend = get_quantum_backend(args.backend)
    
    # Get noise model
    noise_model = get_noise_model(args.noise)
    
    # Get config
    config = get_model_config(args.model, args.dataset)
    config.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'seed': args.seed,
        'backend': backend,
        'shots': args.shots,
        'iid': args.iid,
        'n_clients': args.n_clients,
        'alpha': args.alpha,
        'run_name': (
            f"{args.model}_{args.dataset}_{args.backend}_"
            f"{'iid' if args.iid else f'niid_{args.alpha}'}_"
            f"noise_{args.noise}"
        )
    })
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataset_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        iid=args.iid,
        n_clients=args.n_clients,
        alpha=args.alpha
    )
    
    # Create model
    model = get_model(config)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = QuantumTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
        use_wandb=not args.no_wandb,
        quantum_batch_size=args.batch_size,
        noise_model=noise_model
    )
    
    # Train model
    trainer.train(config['epochs'])
    
    # Test model
    test_loss, test_acc = trainer.test()
    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {test_acc:.2f}%"
    )
    
    # Log quantum metrics
    quantum_metrics = trainer.get_quantum_metrics()
    logger.info("Final quantum metrics:")
    logger.info(f"Average entanglement: {sum(quantum_metrics['entanglement_metrics'])/len(quantum_metrics['entanglement_metrics']):.4f}")
    logger.info(f"Average coherence: {sum(quantum_metrics['coherence_metrics'])/len(quantum_metrics['coherence_metrics']):.4f}")
    logger.info(f"Average circuit depth: {sum(quantum_metrics['circuit_depths'])/len(quantum_metrics['circuit_depths']):.1f}")

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
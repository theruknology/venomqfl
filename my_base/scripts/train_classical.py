"""
Training script for classical models.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.classical.cnn import CNN
from models.classical.mlp import MLP
from trainers import BaseTrainer
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
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 50,
        'log_interval': 10,
        'checkpoint_interval': 5,
        'seed': 42,
        'project_name': 'qfl-backdoor-classical',
    }
    
    if dataset == 'mnist':
        base_config.update({
            'input_channels': 1,
            'input_size': 28,
            'n_classes': 10
        })
    elif dataset == 'fmnist':
        base_config.update({
            'input_channels': 1,
            'input_size': 28,
            'n_classes': 10
        })
    elif dataset == 'cifar10':
        base_config.update({
            'input_channels': 3,
            'input_size': 32,
            'n_classes': 10
        })
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if model_name == 'cnn':
        base_config.update({
            'model_name': 'cnn',
            'conv_channels': [16, 32],
            'fc_features': [128, 64],
            'dropout_rate': 0.25
        })
    elif model_name == 'mlp':
        base_config.update({
            'model_name': 'mlp',
            'hidden_sizes': [256, 128, 64],
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

def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Get model instance.
    
    Args:
        config: Model configuration
        
    Returns:
        nn.Module: Model instance
    """
    if config['model_name'] == 'cnn':
        return CNN(
            input_channels=config['input_channels'],
            conv_channels=config['conv_channels'],
            fc_features=config['fc_features'],
            n_classes=config['n_classes'],
            dropout_rate=config['dropout_rate']
        )
    elif config['model_name'] == 'mlp':
        input_size = config['input_channels'] * config['input_size'] * config['input_size']
        return MLP(
            input_size=input_size,
            hidden_sizes=config['hidden_sizes'],
            n_classes=config['n_classes'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train classical models')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'],
                      help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='mnist',
                      choices=['mnist', 'fmnist', 'cifar10'],
                      help='Dataset to train on')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                      help='Disable CUDA training')
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
    
    # Get config
    config = get_model_config(args.model, args.dataset)
    config.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'seed': args.seed,
        'iid': args.iid,
        'n_clients': args.n_clients,
        'alpha': args.alpha,
        'run_name': f"{args.model}_{args.dataset}_{'iid' if args.iid else f'niid_{args.alpha}'}"
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
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        device=device,
        use_wandb=not args.no_wandb
    )
    
    # Train model
    trainer.train(config['epochs'])
    
    # Test model
    test_loss, test_acc = trainer.test()
    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {test_acc:.2f}%"
    )

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main() 
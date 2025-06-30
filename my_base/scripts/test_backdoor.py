"""
Script for testing backdoor attacks on both classical and quantum models.
"""

import os
import logging
import argparse
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.classical.cnn import CNN
from models.classical.mlp import MLP
from models.quantum.vqc_qiskit import VQCModel
from models.quantum.hybrid_qnn import HybridQNN
from attacks.backdoor import BackdoorPattern, BackdoorAttack
from data.mnist import get_mnist_loaders
from data.fmnist import get_fmnist_loaders
from data.cifar10 import get_cifar10_loaders
from utils.seed import set_seed
from utils.device import get_device
from utils.run_manager import RunManager

logger = logging.getLogger(__name__)

def get_model(
    model_path: str,
    model_type: str,
    config: Dict[str, Any]
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model
        config: Model configuration
        
    Returns:
        nn.Module: Loaded model
    """
    if model_type == 'cnn':
        model = CNN(
            input_channels=config['input_channels'],
            conv_channels=config['conv_channels'],
            fc_features=config['fc_features'],
            n_classes=config['n_classes'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'mlp':
        input_size = config['input_channels'] * config['input_size'] * config['input_size']
        model = MLP(
            input_size=input_size,
            hidden_sizes=config['hidden_sizes'],
            n_classes=config['n_classes'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'vqc':
        model = VQCModel(
            n_qubits=config['n_qubits'],
            n_layers=config['n_layers'],
            n_classes=config['n_classes'],
            shots=config['shots'],
            backend=config['backend']
        )
    elif model_type == 'hybrid':
        model = HybridQNN(
            input_channels=config['input_channels'],
            n_qubits=config['n_qubits'],
            n_qlayers=config['n_qlayers'],
            n_classes=config['n_classes'],
            shots=config['shots'],
            backend=config['backend']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def test_backdoor_attack(
    model: nn.Module,
    test_loader: DataLoader,
    pattern: BackdoorPattern,
    device: torch.device,
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Test backdoor attack on model.
    
    Args:
        model: Model to test
        test_loader: Test data loader
        pattern: Backdoor pattern
        device: Device to use
        save_dir: Directory to save visualizations
        
    Returns:
        dict: Attack statistics
    """
    # Create attack
    attack = BackdoorAttack(
        model=model,
        pattern=pattern,
        test_rate=0.2,
        device=device
    )
    
    # Poison test dataset
    poisoned_dataset, poison_idx = attack.poison_dataset(
        test_loader.dataset,
        is_training=False
    )
    
    # Create data loaders
    clean_loader = DataLoader(
        test_loader.dataset,
        batch_size=test_loader.batch_size,
        shuffle=False
    )
    backdoor_loader = DataLoader(
        poisoned_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False
    )
    
    # Evaluate attack
    metrics = attack.evaluate_backdoor(clean_loader, backdoor_loader)
    
    # Visualize results if save directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Get samples for visualization
        clean_samples = []
        backdoored_samples = []
        for (data, _), (poisoned_data, _) in zip(clean_loader, backdoor_loader):
            clean_samples.append(data[:5])
            backdoored_samples.append(poisoned_data[:5])
            break
        
        clean_samples = torch.cat(clean_samples)
        backdoored_samples = torch.cat(backdoored_samples)
        
        # Visualize pattern
        BackdoorAttack.visualize_pattern(
            pattern,
            (clean_samples.shape[2], clean_samples.shape[3]),
            os.path.join(save_dir, 'pattern.png')
        )
        
        # Visualize samples
        BackdoorAttack.visualize_backdoored_samples(
            clean_samples,
            backdoored_samples,
            num_samples=5,
            save_path=os.path.join(save_dir, 'samples.png')
        )
        
        # Plot metrics
        plt.figure(figsize=(10, 6))
        metrics_list = [
            metrics['clean_accuracy'],
            metrics['backdoor_accuracy'],
            metrics['detection_rate']
        ]
        plt.bar(
            ['Clean Acc', 'Backdoor Acc', 'Detection Rate'],
            metrics_list
        )
        plt.title('Attack Metrics')
        plt.ylabel('Percentage')
        plt.savefig(os.path.join(save_dir, 'metrics.png'))
        plt.close()
    
    return metrics

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test backdoor attacks')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, required=True,
                      choices=['cnn', 'mlp', 'vqc', 'hybrid'],
                      help='Type of model')
    parser.add_argument('--dataset', type=str, default='mnist',
                      choices=['mnist', 'fmnist', 'cifar10'],
                      help='Dataset to test on')
    parser.add_argument('--pattern-type', type=str, default='pixel',
                      choices=['pixel', 'square', 'checkerboard'],
                      help='Type of backdoor pattern')
    parser.add_argument('--pattern-size', type=int, default=3,
                      help='Size of backdoor pattern')
    parser.add_argument('--intensity', type=float, default=1.0,
                      help='Intensity of backdoor pattern')
    parser.add_argument('--target-label', type=int, default=0,
                      help='Target label for backdoored samples')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for testing')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                      help='Disable CUDA')
    parser.add_argument('--save-dir', type=str,
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(not args.no_cuda)
    
    # Load model config from checkpoint
    checkpoint = torch.load(args.model_path)
    config = checkpoint['config']
    
    # Create model
    model = get_model(args.model_path, args.model_type, config)
    model.to(device)
    model.eval()
    
    # Get test data loader
    if args.dataset == 'mnist':
        _, _, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    elif args.dataset == 'fmnist':
        _, _, test_loader = get_fmnist_loaders(batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
        _, _, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create backdoor pattern
    pattern = BackdoorPattern(
        pattern_type=args.pattern_type,
        pattern_size=args.pattern_size,
        intensity=args.intensity,
        target_label=args.target_label
    )
    
    # Test backdoor attack
    metrics = test_backdoor_attack(
        model=model,
        test_loader=test_loader,
        pattern=pattern,
        device=device,
        save_dir=args.save_dir
    )
    
    # Log results
    logger.info("Attack Results:")
    logger.info(f"Clean Accuracy: {metrics['clean_accuracy']:.2f}%")
    logger.info(f"Backdoor Accuracy: {metrics['backdoor_accuracy']:.2f}%")
    logger.info(f"Detection Rate: {metrics['detection_rate']:.2f}%")

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main() 
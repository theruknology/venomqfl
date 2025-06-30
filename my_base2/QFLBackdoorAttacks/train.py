"""
train.py: Main entry point for federated learning experiments in QFLBackdoorAttacks
"""
import argparse
import os
import yaml
from config import get_config
from logger import setup_logger
from utils.run_manager import create_run_folder, append_metrics
from utils.seed import set_seed
from utils.device import get_device

# Dynamically import FL, attacks, defenses, models, data

def main():
    parser = argparse.ArgumentParser(description='QFLBackdoorAttacks Training Script')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--dataset', type=str, help='Dataset override')
    parser.add_argument('--fl_type', type=str, help='FL type override')
    parser.add_argument('--attack', type=str, help='Attack module override')
    parser.add_argument('--defense', type=str, help='Defense module override')
    parser.add_argument('--model', type=str, help='Model override')
    parser.add_argument('--epochs', type=int, help='Epochs override')
    parser.add_argument('--batch_size', type=int, help='Batch size override')
    parser.add_argument('--local_epochs', type=int, help='Local epochs override')
    parser.add_argument('--lr', type=float, help='Learning rate override')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, help='Random seed override')
    parser.add_argument('--verbosity', type=int, default=1, help='Verbosity level (0=warn,1=info,2=debug)')
    args = parser.parse_args()

    # Load config
    config = get_config(vars(args))
    set_seed(config['seed'])

    # Setup run folder and logger
    run_dir = create_run_folder(config)
    logger = setup_logger(log_dir=run_dir, verbosity=args.verbosity)
    logger.info(f"Experiment config: {config}")

    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")

    # Data loader
    if config['dataset'] == 'mnist':
        from data.mnist import get_dataloaders
    elif config['dataset'] == 'fmnist':
        from data.fmnist import get_dataloaders
    elif config['dataset'] == 'cifar10':
        from data.cifar10 import get_dataloaders
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    train_loaders, val_loader, test_loader = get_dataloaders(config)

    # FL loop
    if config['fl_type'] == 'classical':
        from fl.classical_fl import federated_train
    elif config['fl_type'] == 'quantum':
        from fl.quantum_fl import federated_train
    else:
        raise ValueError(f"Unknown FL type: {config['fl_type']}")

    # Run federated training
    metrics = federated_train(config, train_loaders, val_loader, test_loader, device, logger, run_dir)
    append_metrics(metrics, run_dir)
    logger.info("Training complete.")

if __name__ == '__main__':
    main()

"""
utils/run_manager.py: Run folder and metrics management for QFLBackdoorAttacks
"""
import os
import yaml
import csv
from datetime import datetime

def create_run_folder(config):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{timestamp}_{config['fl_type']}_{config['dataset']}_{config.get('attack','none')}_{config.get('defense','none')}"
    run_dir = os.path.join('runs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    # Save config
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    # Init metrics.csv
    metrics_path = os.path.join(run_dir, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'val_acc', 'val_loss'])
        writer.writeheader()
    return run_dir

def append_metrics(metrics, run_dir):
    metrics_path = os.path.join(run_dir, 'metrics.csv')
    with open(metrics_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'val_acc', 'val_loss'])
        for row in metrics:
            writer.writerow(row)

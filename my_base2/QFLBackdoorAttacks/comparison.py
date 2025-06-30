"""
comparison.py: Run cross-model, cross-attack, and cross-defense experiments for QFLBackdoorAttacks
"""
import itertools
import yaml
from config import DEFAULTS, get_config
from train import main as train_main

# Define experiment grid (can be extended)
MODELS = ['cnn', 'mlp', 'vqc', 'hybrid_qnn']
FL_TYPES = ['classical', 'quantum']
ATTACKS = [None, 'label_flip_attack', 'model_replacement_attack', 'parameter_shift_poison']
DEFENSES = [None, 'krum', 'fools_gold', 'fidelity_filtering', 'dual_modality_robust']
DATASETS = ['mnist', 'fmnist', 'cifar10']


def run_comparisons():
    grid = list(itertools.product(MODELS, FL_TYPES, ATTACKS, DEFENSES, DATASETS))
    for model, fl_type, attack, defense, dataset in grid:
        config = DEFAULTS.copy()
        config.update({
            'model': model,
            'fl_type': fl_type,
            'attack': attack,
            'defense': defense,
            'dataset': dataset,
        })
        print(f"\n=== Running: model={model}, fl_type={fl_type}, attack={attack}, defense={defense}, dataset={dataset} ===")
        # In production, you may want to run in subprocess or with logging redirection
        # Here, just call train_main (stub)
        # train_main(config)  # Uncomment to run actual experiments
        # For now, print config as a stub
        print(config)

if __name__ == '__main__':
    run_comparisons()

"""
sweep.py: Hyperparameter sweep utility for QFLBackdoorAttacks
"""
import argparse
import optuna
import yaml
from config import DEFAULTS, SEARCH_SPACE, get_config
from train import main as train_main

# Objective function for Optuna

def objective(trial):
    # Sample hyperparameters
    config = DEFAULTS.copy()
    for key, spec in SEARCH_SPACE.items():
        if spec['type'] == 'loguniform':
            config[key] = trial.suggest_float(key, spec['min'], spec['max'], log=True)
        elif spec['type'] == 'uniform':
            config[key] = trial.suggest_float(key, spec['min'], spec['max'])
        elif spec['type'] == 'categorical':
            config[key] = trial.suggest_categorical(key, spec['values'])
    # Optionally add more trial params here
    # Run training (simulate CLI args)
    # You may want to redirect output/logging per trial
    # For simplicity, call train_main with config overrides
    # (In production, use subprocess or isolated runs)
    # Here, just return a dummy metric for stub
    # Replace with actual metric extraction from run_dir
    return 0.0

def main():
    parser = argparse.ArgumentParser(description='QFLBackdoorAttacks Sweep Script')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of sweep trials')
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize', study_name='qfl_sweep')
    study.optimize(objective, n_trials=args.n_trials)
    print('Best trial:', study.best_trial.params)

if __name__ == '__main__':
    main()

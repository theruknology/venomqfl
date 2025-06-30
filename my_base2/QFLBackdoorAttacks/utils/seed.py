"""
utils/seed.py: Reproducibility utility for QFLBackdoorAttacks
"""
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import pennylane as qml
        qml.numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import qiskit
        qiskit.utils.algorithm_globals.random_seed = seed
    except ImportError:
        pass

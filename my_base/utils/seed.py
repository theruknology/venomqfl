"""
Seed management for reproducible experiments.
"""

import os
import random
import logging
import numpy as np
import torch
import qiskit
import pennylane as qml

logger = logging.getLogger(__name__)

def set_seed(
    seed: int,
    set_torch: bool = True,
    set_numpy: bool = True,
    set_python: bool = True,
    set_quantum: bool = True
) -> None:
    """
    Set random seeds for reproducibility across all frameworks.
    
    Args:
        seed: Random seed
        set_torch: Whether to set PyTorch seeds
        set_numpy: Whether to set NumPy seeds
        set_python: Whether to set Python random seeds
        set_quantum: Whether to set quantum framework seeds
    """
    if set_python:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    if set_numpy:
        np.random.seed(seed)
    
    if set_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if set_quantum:
        try:
            # Set Qiskit seed
            qiskit.utils.algorithm_globals.random_seed = seed
            
            # Set PennyLane seed
            qml.numpy_random_state(seed)
            
        except Exception as e:
            logger.warning(f"Failed to set quantum framework seeds: {e}")
    
    logger.info(f"Set random seed to {seed}")

def get_worker_seed(worker_id: int, base_seed: int = 42) -> int:
    """
    Generate unique seed for each dataloader worker.
    
    Args:
        worker_id: ID of the worker process
        base_seed: Base seed to derive worker seed from
        
    Returns:
        int: Unique seed for worker
    """
    return base_seed + worker_id

def seed_worker(worker_id: int) -> None:
    """
    Initialize worker with unique seed.
    
    Args:
        worker_id: ID of the worker process
    """
    worker_seed = get_worker_seed(worker_id)
    set_seed(worker_seed)

def get_generator(seed: int) -> torch.Generator:
    """
    Get seeded PyTorch generator for DataLoader.
    
    Args:
        seed: Random seed
        
    Returns:
        torch.Generator: Seeded generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator 
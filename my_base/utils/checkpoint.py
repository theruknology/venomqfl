"""
Model checkpoint management utilities.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
import torch
import pennylane as qml
from pathlib import Path

logger = logging.getLogger(__name__)

def save_checkpoint(
    model: Union[torch.nn.Module, qml.QNode],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    path: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model or PennyLane QNode
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        config: Configuration dictionary
        path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,
        'config': config,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Handle different model types
    if isinstance(model, torch.nn.Module):
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['model_type'] = 'classical'
    else:  # QNode
        checkpoint['model_state_dict'] = {
            'params': model.weights,  # Assuming weights are stored in QNode
            'qnode_id': str(model.qnode_id)  # Store QNode identifier
        }
        checkpoint['model_type'] = 'quantum'
    
    # Save checkpoint
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
    
    # Save best model copy if needed
    if is_best:
        best_path = path.parent / 'model_best.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")

def load_checkpoint(
    path: str,
    model: Optional[Union[torch.nn.Module, qml.QNode]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to load model to (optional)
        
    Returns:
        dict: Checkpoint data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    # Load checkpoint data
    checkpoint = torch.load(path, map_location=device if device else 'cpu')
    
    # Restore model state if provided
    if model is not None:
        if checkpoint['model_type'] == 'classical':
            if not isinstance(model, torch.nn.Module):
                raise TypeError("Classical checkpoint requires PyTorch model")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:  # quantum
            if not isinstance(model, qml.QNode):
                raise TypeError("Quantum checkpoint requires PennyLane QNode")
            # Restore quantum parameters
            model.weights = checkpoint['model_state_dict']['params']
    
    # Restore optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        str: Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pth'))
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
    return str(latest)

def clean_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 5,
    keep_best: bool = True
) -> None:
    """
    Remove old checkpoints, keeping only the N most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep best model checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoints except best model
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pth'))
    if not checkpoints:
        return
    
    # Sort by modification time (oldest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    
    # Remove old checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        checkpoint.unlink()
        logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    # Keep best model if requested
    if not keep_best:
        best_model = checkpoint_dir / 'model_best.pth'
        if best_model.exists():
            best_model.unlink()
            logger.debug("Removed best model checkpoint") 
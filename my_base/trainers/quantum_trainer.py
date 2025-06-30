"""
Quantum model trainer with quantum-specific functionality.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from trainers.base_trainer import BaseTrainer
from models.quantum.vqc_qiskit import VQCModel
from models.quantum.hybrid_qnn import HybridQNN

logger = logging.getLogger(__name__)

class QuantumTrainer(BaseTrainer):
    """Trainer for quantum and hybrid quantum models."""
    
    def __init__(
        self,
        model: Union[VQCModel, HybridQNN],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        run_manager: Optional[Any] = None,
        use_wandb: bool = True,
        quantum_batch_size: int = 32,
        noise_model: Optional[Any] = None
    ):
        """
        Initialize quantum trainer.
        
        Args:
            model: Quantum or hybrid quantum model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer instance
            criterion: Loss criterion
            config: Training configuration
            device: Device to use
            run_manager: Run manager instance
            use_wandb: Whether to use W&B logging
            quantum_batch_size: Batch size for quantum circuit execution
            noise_model: Optional quantum noise model
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            device=device,
            run_manager=run_manager,
            use_wandb=use_wandb
        )
        
        self.quantum_batch_size = quantum_batch_size
        self.noise_model = noise_model
        
        # Quantum-specific metrics
        self.quantum_gradients: List[torch.Tensor] = []
        self.circuit_depths: List[int] = []
        self.entanglement_metrics: List[float] = []
        self.coherence_metrics: List[float] = []
        
        # Apply noise model if provided
        if self.noise_model and isinstance(self.model, (VQCModel, HybridQNN)):
            self._apply_noise_model()
    
    def _apply_noise_model(self) -> None:
        """Apply quantum noise model to device."""
        if hasattr(self.model, 'dev'):
            self.model.dev.set_noise_model(self.noise_model)
    
    def _compute_quantum_metrics(
        self,
        data_batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantum-specific metrics.
        
        Args:
            data_batch: Batch of input data
            
        Returns:
            dict: Dictionary of quantum metrics
        """
        # Get quantum patterns
        patterns = self.model.get_backdoor_patterns(data_batch)
        
        # Compute gradients
        gradients = self.model.get_quantum_gradients(data_batch)
        
        metrics = {
            'entanglement': patterns['entanglement'],
            'coherence': patterns['coherence'],
            'gradient_norm': torch.norm(gradients),
            'circuit_depth': self.model.get_circuit_depth()
        }
        
        return metrics
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch with quantum-specific handling.
        
        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Process in quantum batches
            for i in range(0, len(data), self.quantum_batch_size):
                batch_data = data[i:i + self.quantum_batch_size]
                batch_target = target[i:i + self.quantum_batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch_data)
                loss = self.criterion(output, batch_target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(batch_target.view_as(pred)).sum().item()
                total += batch_target.size(0)
            
            # Compute quantum metrics every few batches
            if batch_idx % self.config.get('quantum_metric_interval', 10) == 0:
                quantum_metrics = self._compute_quantum_metrics(batch_data)
                
                # Store metrics
                self.entanglement_metrics.append(quantum_metrics['entanglement'].item())
                self.coherence_metrics.append(quantum_metrics['coherence'].item())
                self.circuit_depths.append(quantum_metrics['circuit_depth'])
                
                # Log to W&B
                if self.use_wandb:
                    wandb.log({
                        'quantum/entanglement': quantum_metrics['entanglement'],
                        'quantum/coherence': quantum_metrics['coherence'],
                        'quantum/gradient_norm': quantum_metrics['gradient_norm'],
                        'quantum/circuit_depth': quantum_metrics['circuit_depth']
                    })
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                logger.info(
                    f'Train Epoch: {self.current_epoch} '
                    f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(
        self,
        loader: DataLoader,
        prefix: str = 'Val'
    ) -> Tuple[float, float]:
        """
        Validate with quantum-specific handling.
        
        Args:
            loader: Data loader to validate on
            prefix: Prefix for logging
            
        Returns:
            tuple: (validation_loss, validation_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        quantum_metrics_list = []
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Process in quantum batches
                for i in range(0, len(data), self.quantum_batch_size):
                    batch_data = data[i:i + self.quantum_batch_size]
                    batch_target = target[i:i + self.quantum_batch_size]
                    
                    output = self.model(batch_data)
                    loss = self.criterion(output, batch_target)
                    
                    total_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(batch_target.view_as(pred)).sum().item()
                    total += batch_target.size(0)
                    
                    # Compute quantum metrics
                    quantum_metrics_list.append(
                        self._compute_quantum_metrics(batch_data)
                    )
        
        # Average quantum metrics
        avg_quantum_metrics = {
            k: torch.stack([m[k] for m in quantum_metrics_list]).mean()
            for k in quantum_metrics_list[0].keys()
        }
        
        # Log validation quantum metrics
        if self.use_wandb:
            wandb.log({
                f'{prefix.lower()}/quantum/entanglement': avg_quantum_metrics['entanglement'],
                f'{prefix.lower()}/quantum/coherence': avg_quantum_metrics['coherence'],
                f'{prefix.lower()}/quantum/circuit_depth': avg_quantum_metrics['circuit_depth']
            })
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        logger.info(
            f'{prefix} set: Average loss: {avg_loss:.4f}, '
            f'Accuracy: {correct}/{total} ({accuracy:.2f}%)'
        )
        
        return avg_loss, accuracy
    
    def get_quantum_metrics(self) -> Dict[str, List[Union[float, int]]]:
        """
        Get quantum-specific training metrics.
        
        Returns:
            dict: Dictionary containing quantum metrics
        """
        return {
            'entanglement_metrics': self.entanglement_metrics,
            'coherence_metrics': self.coherence_metrics,
            'circuit_depths': self.circuit_depths
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save checkpoint with quantum-specific metrics.
        
        Args:
            filename: Name of checkpoint file
        """
        # Get base checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'config': self.config,
            # Add quantum-specific metrics
            'quantum_metrics': self.get_quantum_metrics()
        }
        
        super().save_checkpoint(filename)
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load checkpoint with quantum-specific metrics.
        
        Args:
            filename: Name of checkpoint file
        """
        super().load_checkpoint(filename)
        
        # Load quantum-specific metrics
        checkpoint = torch.load(
            self.run_manager.get_checkpoint_path(filename)
        )
        quantum_metrics = checkpoint.get('quantum_metrics', {})
        
        self.entanglement_metrics = quantum_metrics.get('entanglement_metrics', [])
        self.coherence_metrics = quantum_metrics.get('coherence_metrics', [])
        self.circuit_depths = quantum_metrics.get('circuit_depths', [])
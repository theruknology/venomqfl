"""
Base trainer class for both classical and quantum models.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.device import get_device
from utils.run_manager import RunManager

logger = logging.getLogger(__name__)

class BaseTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        run_manager: Optional[RunManager] = None,
        use_wandb: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer instance
            criterion: Loss criterion
            config: Training configuration
            device: Device to use (auto-detected if None)
            run_manager: Run manager instance
            use_wandb: Whether to use W&B logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device if device else get_device()
        self.run_manager = run_manager if run_manager else RunManager()
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        
        # Initialize W&B if requested
        if self.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.get('project_name', 'qfl-backdoor'),
            name=self.config.get('run_name'),
            config=self.config
        )
        wandb.watch(self.model)
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
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
    
    def validate(self, loader: DataLoader, prefix: str = 'Val') -> Tuple[float, float]:
        """
        Validate model on given loader.
        
        Args:
            loader: Data loader to validate on
            prefix: Prefix for logging ('Val' or 'Test')
            
        Returns:
            tuple: (validation_loss, validation_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        logger.info(
            f'{prefix} set: Average loss: {avg_loss:.4f}, '
            f'Accuracy: {correct}/{total} ({accuracy:.2f}%)'
        )
        
        return avg_loss, accuracy
    
    def train(self, epochs: int) -> None:
        """
        Train model for given number of epochs.
        
        Args:
            epochs: Number of epochs to train
        """
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self.validate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_acc.pt')
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_loss.pt')
            
            # Regular checkpoint
            if epoch % self.config.get('checkpoint_interval', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch
            }
            
            if self.use_wandb:
                wandb.log(metrics)
            
            self.run_manager.log_metrics(metrics)
    
    def test(self) -> Tuple[float, float]:
        """
        Test model on test set.
        
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        return self.validate(self.test_loader, prefix='Test')
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
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
            'config': self.config
        }
        
        save_checkpoint(
            checkpoint,
            filename,
            self.run_manager.get_checkpoint_dir()
        )
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            filename: Name of checkpoint file
        """
        checkpoint = load_checkpoint(
            filename,
            self.run_manager.get_checkpoint_dir()
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        
        # Update config with saved values
        self.config.update(checkpoint['config'])
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get training metrics.
        
        Returns:
            dict: Dictionary containing training metrics
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get best metrics achieved during training.
        
        Returns:
            dict: Dictionary containing best metrics
        """
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        } 
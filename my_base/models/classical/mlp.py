"""
Classical MLP model architecture.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """
    Classical MLP architecture designed to be comparable to hybrid QNN.
    2 hidden layers with ReLU and dropout.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256],
        output_dim: int = 10,
        dropout: float = 0.5,
        batch_norm: bool = True
    ):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes)
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Build network
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims)-1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Batch norm
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_features: Whether to return penultimate layer features
            
        Returns:
            torch.Tensor: Model output
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Get all layer outputs
        features = None
        for i, layer in enumerate(self.model):
            x = layer(x)
            # Store features from penultimate layer
            if return_features and isinstance(layer, nn.Linear):
                if i == len(self.model) - 3:  # Before last dropout and linear
                    features = x
        
        if return_features:
            return x, features
        return x
    
    def get_activation_patterns(
        self,
        x: torch.Tensor,
        layer_idx: int = -2
    ) -> torch.Tensor:
        """
        Get activation patterns for backdoor detection.
        
        Args:
            x: Input tensor
            layer_idx: Index of layer to get activations from (-2 for penultimate)
            
        Returns:
            torch.Tensor: Activation patterns
        """
        self.eval()
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
        return features
    
    @staticmethod
    def get_model_for_dataset(
        dataset: str,
        quantum_input_dim: Optional[int] = None,
        **kwargs
    ) -> "MLP":
        """
        Get MLP model configured for specific dataset.
        
        Args:
            dataset: Dataset name ("mnist", "fmnist", or "cifar10")
            quantum_input_dim: Input dimension to match quantum model
            **kwargs: Additional model parameters
            
        Returns:
            MLP: Configured model
        """
        if dataset in ["mnist", "fmnist"]:
            input_dim = 784  # 28x28
            if quantum_input_dim:
                input_dim = quantum_input_dim
            return MLP(
                input_dim=input_dim,
                hidden_dims=[256, 256],
                output_dim=10,
                **kwargs
            )
        elif dataset == "cifar10":
            input_dim = 3072  # 32x32x3
            if quantum_input_dim:
                input_dim = quantum_input_dim
            return MLP(
                input_dim=input_dim,
                hidden_dims=[512, 512],  # Larger for CIFAR-10
                output_dim=10,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def get_backdoor_patterns(
        self,
        x: torch.Tensor,
        layer_idx: int = -2,
        eps: float = 1e-6
    ) -> Dict[str, torch.Tensor]:
        """
        Get patterns for backdoor detection.
        
        Args:
            x: Input tensor
            layer_idx: Layer index to analyze
            eps: Small constant for numerical stability
            
        Returns:
            dict: Dictionary containing various patterns
        """
        patterns = {}
        
        # Get activations
        acts = self.get_activation_patterns(x, layer_idx)
        
        # Basic statistics
        patterns['mean'] = acts.mean(dim=0)
        patterns['std'] = acts.std(dim=0)
        patterns['max'] = acts.max(dim=0)[0]
        patterns['min'] = acts.min(dim=0)[0]
        
        # Activation frequency
        patterns['freq'] = (acts > 0).float().mean(dim=0)
        
        # Cosine similarity matrix
        norm_acts = acts / (acts.norm(dim=1, keepdim=True) + eps)
        patterns['cosine_sim'] = torch.mm(norm_acts, norm_acts.t())
        
        # Additional patterns specific to MLP
        patterns['sparsity'] = (acts == 0).float().mean(dim=0)
        patterns['l1_norm'] = acts.abs().mean(dim=0)
        patterns['l2_norm'] = acts.pow(2).mean(dim=0).sqrt()
        
        return patterns 
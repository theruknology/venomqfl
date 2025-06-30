"""
Classical CNN model architecture.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class CNN(nn.Module):
    """
    Classical CNN architecture for image classification.
    2 conv layers (ReLU+MaxPool), 2 FC layers, final softmax.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 28,
        num_classes: int = 10,
        hidden_channels: Tuple[int, ...] = (32, 64),
        kernel_sizes: Tuple[int, ...] = (3, 3),
        fc_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.5
    ):
        """
        Initialize CNN model.
        
        Args:
            in_channels: Number of input channels
            img_size: Input image size
            num_classes: Number of output classes
            hidden_channels: Number of channels in conv layers
            kernel_sizes: Kernel sizes for conv layers
            fc_dims: Dimensions of FC layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.kernel_sizes = kernel_sizes
        self.fc_dims = fc_dims
        self.dropout = dropout
        
        # First conv block
        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_channels[0],
            kernel_sizes[0],
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.pool1 = nn.MaxPool2d(2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(
            hidden_channels[0],
            hidden_channels[1],
            kernel_sizes[1],
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.pool2 = nn.MaxPool2d(2)
        
        # Calculate size after convolutions
        conv_size = img_size
        for _ in range(2):  # 2 max pooling layers
            conv_size = conv_size // 2
        
        # Flatten size
        flatten_size = hidden_channels[1] * conv_size * conv_size
        
        # First FC layer
        self.fc1 = nn.Linear(flatten_size, fc_dims[0])
        self.fc1_bn = nn.BatchNorm1d(fc_dims[0])
        self.fc1_drop = nn.Dropout(dropout)
        
        # Second FC layer
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc2_bn = nn.BatchNorm1d(fc_dims[1])
        self.fc2_drop = nn.Dropout(dropout)
        
        # Output layer
        self.fc3 = nn.Linear(fc_dims[1], num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            return_features: Whether to return intermediate features
            
        Returns:
            torch.Tensor: Model output
        """
        # First conv block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # First FC layer
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc1_drop(x)
        
        # Second FC layer
        features = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc2_drop(features)
        
        # Output layer
        logits = self.fc3(x)
        
        if return_features:
            return logits, features
        return logits
    
    def get_activation_patterns(
        self,
        x: torch.Tensor,
        layer: str = "fc2"
    ) -> torch.Tensor:
        """
        Get activation patterns for backdoor detection.
        
        Args:
            x: Input tensor
            layer: Layer to get activations from
            
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
        **kwargs
    ) -> "CNN":
        """
        Get CNN model configured for specific dataset.
        
        Args:
            dataset: Dataset name ("mnist", "fmnist", or "cifar10")
            **kwargs: Additional model parameters
            
        Returns:
            CNN: Configured model
        """
        if dataset in ["mnist", "fmnist"]:
            return CNN(
                in_channels=1,
                img_size=28,
                num_classes=10,
                **kwargs
            )
        elif dataset == "cifar10":
            return CNN(
                in_channels=3,
                img_size=32,
                num_classes=10,
                hidden_channels=(64, 128),  # Larger for CIFAR-10
                fc_dims=(256, 128),  # Larger for CIFAR-10
                **kwargs
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def get_backdoor_patterns(
        self,
        x: torch.Tensor,
        layer: str = "fc2",
        eps: float = 1e-6
    ) -> Dict[str, torch.Tensor]:
        """
        Get patterns for backdoor detection.
        
        Args:
            x: Input tensor
            layer: Layer to analyze
            eps: Small constant for numerical stability
            
        Returns:
            dict: Dictionary containing various patterns
        """
        patterns = {}
        
        # Get activations
        acts = self.get_activation_patterns(x, layer)
        
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
        
        return patterns 
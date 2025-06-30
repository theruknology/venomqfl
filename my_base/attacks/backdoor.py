"""
Backdoor attack utilities for both classical and quantum models.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from PIL import Image

from models.classical.cnn import CNN
from models.classical.mlp import MLP
from models.quantum.vqc_qiskit import VQCModel
from models.quantum.hybrid_qnn import HybridQNN

logger = logging.getLogger(__name__)

class BackdoorPattern:
    """Class for generating and applying backdoor patterns."""
    
    def __init__(
        self,
        pattern_type: str = 'pixel',
        pattern_size: int = 3,
        position: Optional[Tuple[int, int]] = None,
        intensity: float = 1.0,
        target_label: int = 0
    ):
        """
        Initialize backdoor pattern.
        
        Args:
            pattern_type: Type of backdoor pattern ('pixel', 'square', 'checkerboard')
            pattern_size: Size of pattern in pixels
            position: Position of pattern (random if None)
            intensity: Intensity of pattern (0 to 1)
            target_label: Target label for backdoored samples
        """
        self.pattern_type = pattern_type
        self.pattern_size = pattern_size
        self.position = position
        self.intensity = intensity
        self.target_label = target_label
    
    def _generate_pixel_pattern(self, image_size: Tuple[int, int]) -> torch.Tensor:
        """Generate single pixel pattern."""
        pattern = torch.zeros(image_size)
        if self.position is None:
            x = np.random.randint(0, image_size[0])
            y = np.random.randint(0, image_size[1])
        else:
            x, y = self.position
        pattern[x, y] = self.intensity
        return pattern
    
    def _generate_square_pattern(self, image_size: Tuple[int, int]) -> torch.Tensor:
        """Generate square pattern."""
        pattern = torch.zeros(image_size)
        if self.position is None:
            x = np.random.randint(0, image_size[0] - self.pattern_size)
            y = np.random.randint(0, image_size[1] - self.pattern_size)
        else:
            x, y = self.position
        
        pattern[x:x + self.pattern_size, y:y + self.pattern_size] = self.intensity
        return pattern
    
    def _generate_checkerboard_pattern(self, image_size: Tuple[int, int]) -> torch.Tensor:
        """Generate checkerboard pattern."""
        pattern = torch.zeros(image_size)
        if self.position is None:
            x = np.random.randint(0, image_size[0] - self.pattern_size)
            y = np.random.randint(0, image_size[1] - self.pattern_size)
        else:
            x, y = self.position
        
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                if (i + j) % 2 == 0:
                    pattern[x + i, y + j] = self.intensity
        
        return pattern
    
    def generate_pattern(self, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate backdoor pattern.
        
        Args:
            image_size: Size of image (height, width)
            
        Returns:
            torch.Tensor: Generated pattern
        """
        if self.pattern_type == 'pixel':
            return self._generate_pixel_pattern(image_size)
        elif self.pattern_type == 'square':
            return self._generate_square_pattern(image_size)
        elif self.pattern_type == 'checkerboard':
            return self._generate_checkerboard_pattern(image_size)
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern_type}")
    
    def apply_pattern(
        self,
        image: torch.Tensor,
        apply_transform: bool = True
    ) -> torch.Tensor:
        """
        Apply backdoor pattern to image.
        
        Args:
            image: Input image
            apply_transform: Whether to apply random transformations
            
        Returns:
            torch.Tensor: Backdoored image
        """
        # Generate pattern for each channel
        pattern = self.generate_pattern((image.shape[1], image.shape[2]))
        pattern = pattern.unsqueeze(0).repeat(image.shape[0], 1, 1)
        
        # Apply pattern
        backdoored = image.clone()
        backdoored += pattern
        
        # Apply random transformations if requested
        if apply_transform:
            transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1))
            ])
            backdoored = transform(backdoored)
        
        # Clip values
        backdoored = torch.clamp(backdoored, 0, 1)
        
        return backdoored

class BackdoorAttack:
    """Class for performing backdoor attacks."""
    
    def __init__(
        self,
        model: Union[CNN, MLP, VQCModel, HybridQNN],
        pattern: BackdoorPattern,
        poisoning_rate: float = 0.1,
        test_rate: float = 0.2,
        device: Optional[torch.device] = None
    ):
        """
        Initialize backdoor attack.
        
        Args:
            model: Model to attack
            pattern: Backdoor pattern to use
            poisoning_rate: Fraction of training data to poison
            test_rate: Fraction of test data to poison
            device: Device to use
        """
        self.model = model
        self.pattern = pattern
        self.poisoning_rate = poisoning_rate
        self.test_rate = test_rate
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Attack statistics
        self.clean_acc: float = 0.0
        self.backdoor_acc: float = 0.0
        self.detection_rate: float = 0.0
    
    def poison_dataset(
        self,
        dataset: Dataset,
        is_training: bool = True
    ) -> Tuple[Dataset, List[int]]:
        """
        Poison dataset with backdoor pattern.
        
        Args:
            dataset: Dataset to poison
            is_training: Whether this is training data
            
        Returns:
            tuple: (poisoned dataset, indices of poisoned samples)
        """
        rate = self.poisoning_rate if is_training else self.test_rate
        n_poison = int(len(dataset) * rate)
        
        # Select samples to poison
        poison_idx = np.random.choice(
            len(dataset),
            size=n_poison,
            replace=False
        )
        
        # Create poisoned dataset
        poisoned_data = []
        poisoned_targets = []
        
        for i in range(len(dataset)):
            img, target = dataset[i]
            if i in poison_idx:
                # Apply backdoor pattern
                if isinstance(img, torch.Tensor):
                    poisoned_img = self.pattern.apply_pattern(img)
                else:
                    poisoned_img = self.pattern.apply_pattern(
                        torch.tensor(img).float()
                    )
                poisoned_target = self.pattern.target_label
            else:
                poisoned_img = img
                poisoned_target = target
            
            poisoned_data.append(poisoned_img)
            poisoned_targets.append(poisoned_target)
        
        # Create new dataset
        poisoned_data = torch.stack(poisoned_data)
        poisoned_targets = torch.tensor(poisoned_targets)
        poisoned_dataset = TensorDataset(poisoned_data, poisoned_targets)
        
        return poisoned_dataset, poison_idx.tolist()
    
    def evaluate_backdoor(
        self,
        clean_loader: DataLoader,
        backdoor_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate backdoor attack.
        
        Args:
            clean_loader: Clean test data loader
            backdoor_loader: Backdoored test data loader
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        # Evaluate on clean data
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in clean_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        self.clean_acc = 100. * correct / total
        
        # Evaluate on backdoored data
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in backdoor_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        self.backdoor_acc = 100. * correct / total
        
        # Compute detection rate if model supports it
        if hasattr(self.model, 'get_backdoor_patterns'):
            detected = 0
            total = 0
            with torch.no_grad():
                for data, _ in backdoor_loader:
                    data = data.to(self.device)
                    patterns = self.model.get_backdoor_patterns(data)
                    
                    # Simple threshold-based detection
                    if isinstance(self.model, (VQCModel, HybridQNN)):
                        # Use quantum metrics for detection
                        entanglement = patterns['entanglement']
                        coherence = patterns['coherence']
                        detected += ((entanglement > 0.5) & (coherence < 0.3)).sum().item()
                    else:
                        # Use classical metrics for detection
                        activation_patterns = patterns['activations']
                        detected += (activation_patterns.std(dim=1) > 0.5).sum().item()
                    
                    total += data.size(0)
            
            self.detection_rate = 100. * detected / total
        
        return {
            'clean_accuracy': self.clean_acc,
            'backdoor_accuracy': self.backdoor_acc,
            'detection_rate': self.detection_rate
        }
    
    def get_attack_stats(self) -> Dict[str, float]:
        """
        Get attack statistics.
        
        Returns:
            dict: Dictionary containing attack statistics
        """
        return {
            'clean_accuracy': self.clean_acc,
            'backdoor_accuracy': self.backdoor_acc,
            'detection_rate': self.detection_rate,
            'poisoning_rate': self.poisoning_rate,
            'test_rate': self.test_rate
        }
    
    @staticmethod
    def visualize_pattern(
        pattern: BackdoorPattern,
        image_size: Tuple[int, int],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize backdoor pattern.
        
        Args:
            pattern: Backdoor pattern to visualize
            image_size: Size of image
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Generate pattern
        pattern_tensor = pattern.generate_pattern(image_size)
        
        # Plot pattern
        plt.figure(figsize=(8, 8))
        plt.imshow(pattern_tensor.numpy(), cmap='gray')
        plt.colorbar()
        plt.title(f'Backdoor Pattern ({pattern.pattern_type})')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_backdoored_samples(
        clean_samples: torch.Tensor,
        backdoored_samples: torch.Tensor,
        num_samples: int = 5,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize backdoored samples.
        
        Args:
            clean_samples: Clean samples
            backdoored_samples: Backdoored samples
            num_samples: Number of samples to visualize
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Plot samples
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for i in range(num_samples):
            # Plot clean sample
            axes[0, i].imshow(clean_samples[i].squeeze().numpy(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Clean')
            
            # Plot backdoored sample
            axes[1, i].imshow(backdoored_samples[i].squeeze().numpy(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Backdoored')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
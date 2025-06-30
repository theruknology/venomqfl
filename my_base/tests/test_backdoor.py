"""
Unit tests for backdoor attack implementation.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from attacks.backdoor import BackdoorPattern, BackdoorAttack
from models.classical.cnn import CNN
from models.quantum.vqc_qiskit import VQCModel

class TestBackdoorPattern(unittest.TestCase):
    """Test cases for BackdoorPattern class."""
    
    def setUp(self):
        """Set up test cases."""
        self.image_size = (28, 28)
        self.pattern_types = ['pixel', 'square', 'checkerboard']
        self.pattern_size = 3
        self.intensity = 1.0
        self.target_label = 0
    
    def test_pattern_generation(self):
        """Test pattern generation."""
        for pattern_type in self.pattern_types:
            pattern = BackdoorPattern(
                pattern_type=pattern_type,
                pattern_size=self.pattern_size,
                intensity=self.intensity,
                target_label=self.target_label
            )
            
            # Generate pattern
            pattern_tensor = pattern.generate_pattern(self.image_size)
            
            # Check shape
            self.assertEqual(pattern_tensor.shape, self.image_size)
            
            # Check values
            self.assertTrue(torch.all(pattern_tensor >= 0))
            self.assertTrue(torch.all(pattern_tensor <= self.intensity))
            
            # Check if pattern is non-zero
            self.assertTrue(torch.any(pattern_tensor > 0))
    
    def test_pattern_application(self):
        """Test pattern application."""
        # Create test image
        image = torch.randn(1, *self.image_size)
        
        for pattern_type in self.pattern_types:
            pattern = BackdoorPattern(
                pattern_type=pattern_type,
                pattern_size=self.pattern_size,
                intensity=self.intensity,
                target_label=self.target_label
            )
            
            # Apply pattern
            backdoored = pattern.apply_pattern(image, apply_transform=False)
            
            # Check shape
            self.assertEqual(backdoored.shape, image.shape)
            
            # Check if image was modified
            self.assertTrue(torch.any(backdoored != image))
            
            # Check values
            self.assertTrue(torch.all(backdoored >= 0))
            self.assertTrue(torch.all(backdoored <= 1))

class TestBackdoorAttack(unittest.TestCase):
    """Test cases for BackdoorAttack class."""
    
    def setUp(self):
        """Set up test cases."""
        # Create small test dataset
        self.batch_size = 32
        self.input_size = 28
        self.n_samples = 100
        
        self.data = torch.randn(self.n_samples, 1, self.input_size, self.input_size)
        self.targets = torch.randint(0, 10, (self.n_samples,))
        self.dataset = TensorDataset(self.data, self.targets)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size)
        
        # Create test models
        self.cnn = CNN(
            input_channels=1,
            conv_channels=[16, 32],
            fc_features=[128, 64],
            n_classes=10,
            dropout_rate=0.5
        )
        
        self.vqc = VQCModel(
            n_qubits=4,
            n_layers=2,
            n_classes=10,
            shots=1000,
            backend='aer_simulator'
        )
        
        # Create backdoor pattern
        self.pattern = BackdoorPattern(
            pattern_type='pixel',
            pattern_size=3,
            intensity=1.0,
            target_label=0
        )
    
    def test_dataset_poisoning(self):
        """Test dataset poisoning."""
        # Test with classical model
        attack = BackdoorAttack(
            model=self.cnn,
            pattern=self.pattern,
            poisoning_rate=0.1,
            test_rate=0.2
        )
        
        # Poison training dataset
        poisoned_dataset, poison_idx = attack.poison_dataset(
            self.dataset,
            is_training=True
        )
        
        # Check number of poisoned samples
        self.assertEqual(len(poison_idx), int(self.n_samples * 0.1))
        
        # Check if poisoned samples have target label
        for i in poison_idx:
            _, target = poisoned_dataset[i]
            self.assertEqual(target.item(), self.pattern.target_label)
        
        # Test with quantum model
        attack = BackdoorAttack(
            model=self.vqc,
            pattern=self.pattern,
            poisoning_rate=0.1,
            test_rate=0.2
        )
        
        # Poison test dataset
        poisoned_dataset, poison_idx = attack.poison_dataset(
            self.dataset,
            is_training=False
        )
        
        # Check number of poisoned samples
        self.assertEqual(len(poison_idx), int(self.n_samples * 0.2))
    
    def test_attack_evaluation(self):
        """Test attack evaluation."""
        # Test with classical model
        attack = BackdoorAttack(
            model=self.cnn,
            pattern=self.pattern,
            poisoning_rate=0.1,
            test_rate=0.2
        )
        
        # Create poisoned dataset
        poisoned_dataset, _ = attack.poison_dataset(
            self.dataset,
            is_training=False
        )
        
        # Create data loaders
        clean_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        backdoor_loader = DataLoader(poisoned_dataset, batch_size=self.batch_size)
        
        # Evaluate attack
        metrics = attack.evaluate_backdoor(clean_loader, backdoor_loader)
        
        # Check metrics
        self.assertIn('clean_accuracy', metrics)
        self.assertIn('backdoor_accuracy', metrics)
        self.assertIn('detection_rate', metrics)
        
        # Check metric values
        for value in metrics.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 100)
    
    def test_attack_visualization(self):
        """Test attack visualization."""
        # Create attack
        attack = BackdoorAttack(
            model=self.cnn,
            pattern=self.pattern,
            poisoning_rate=0.1,
            test_rate=0.2
        )
        
        # Test pattern visualization
        BackdoorAttack.visualize_pattern(
            self.pattern,
            (self.input_size, self.input_size)
        )
        
        # Test sample visualization
        clean_samples = self.data[:5]
        backdoored_samples = attack.pattern.apply_pattern(clean_samples)
        
        BackdoorAttack.visualize_backdoored_samples(
            clean_samples,
            backdoored_samples,
            num_samples=5
        )

if __name__ == '__main__':
    unittest.main() 
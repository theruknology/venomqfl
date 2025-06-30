"""
CIFAR-10 dataset loader with federated learning support.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from typing import Tuple, List, Optional
import pennylane as qml
from pathlib import Path

logger = logging.getLogger(__name__)

class QuantumCIFAR10(Dataset):
    """CIFAR-10 dataset with quantum encoding support."""
    
    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        n_qubits: int = 4,
        normalize: bool = True,
        grayscale: bool = True
    ):
        """
        Initialize QuantumCIFAR10 dataset.
        
        Args:
            data: Input images
            targets: Target labels
            n_qubits: Number of qubits for encoding
            normalize: Whether to normalize data to [0,1]
            grayscale: Whether to convert to grayscale
        """
        self.data = data
        self.targets = targets
        self.n_qubits = n_qubits
        self.normalize = normalize
        self.grayscale = grayscale
        
        # Class names for CIFAR-10
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Compute input dimension for quantum encoding
        self.input_dim = 2 ** n_qubits
        
        # Convert to grayscale if needed
        if self.grayscale and len(self.data.shape) == 4:
            self.data = 0.299 * self.data[:, 0] + 0.587 * self.data[:, 1] + 0.114 * self.data[:, 2]
        
        # Resize images to match quantum input dimension
        if len(self.data.shape) > 2:
            self.data = self.data.reshape(len(self.data), -1)
        
        if self.data.shape[1] != self.input_dim:
            # Use average pooling to downsample
            size = int(np.sqrt(self.input_dim))
            if self.grayscale:
                pool = torch.nn.AdaptiveAvgPool2d((size, size))
                self.data = pool(self.data.reshape(-1, 1, 32, 32)).reshape(-1, self.input_dim)
            else:
                pool = torch.nn.AdaptiveAvgPool2d((size, size))
                self.data = pool(self.data.reshape(-1, 3, 32, 32))
                self.data = self.data.reshape(-1, 3 * size * size)
                # Take first input_dim features if RGB
                self.data = self.data[:, :self.input_dim]
        
        if normalize:
            self.data = self.data / 255.0
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get quantum-encoded data item.
        
        Args:
            idx: Data index
            
        Returns:
            tuple: (encoded_data, target)
        """
        x = self.data[idx]
        y = self.targets[idx]
        
        # Quantum encoding happens in the model
        return x, y

def create_federated_loaders(
    data_dir: str,
    n_clients: int,
    batch_size: int,
    distribution: str = "iid",
    alpha: float = 0.5,
    quantum: bool = False,
    n_qubits: int = 4,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    Create federated dataloaders for CIFAR-10.
    
    Args:
        data_dir: Data directory
        n_clients: Number of clients
        batch_size: Training batch size
        distribution: Data distribution type ("iid", "non-iid", "dirichlet")
        alpha: Dirichlet concentration parameter
        quantum: Whether to use quantum encoding
        n_qubits: Number of qubits for quantum encoding
        test_batch_size: Test batch size (defaults to batch_size)
        num_workers: Number of dataloader workers
        seed: Random seed
        
    Returns:
        tuple: (client_loaders, val_loader, test_loader)
    """
    if test_batch_size is None:
        test_batch_size = batch_size
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform
    )
    
    # Split training data into train and validation
    n_train = len(train_dataset)
    n_val = n_train // 10  # 10% for validation
    n_train = n_train - n_val
    
    train_dataset, val_dataset = random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create client datasets based on distribution
    client_datasets = []
    
    if distribution == "iid":
        # IID: randomly shuffle and split
        indices = torch.randperm(n_train)
        client_size = n_train // n_clients
        for i in range(n_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size
            client_indices = indices[start_idx:end_idx]
            client_datasets.append(Subset(train_dataset, client_indices))
    
    elif distribution == "non-iid":
        # Non-IID: sort by label and distribute
        sorted_indices = torch.argsort(train_dataset.dataset.targets[train_dataset.indices])
        client_size = n_train // n_clients
        for i in range(n_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size
            client_indices = sorted_indices[start_idx:end_idx]
            client_datasets.append(Subset(train_dataset, client_indices))
    
    elif distribution == "dirichlet":
        # Dirichlet: sample label distributions from Dirichlet
        n_classes = 10
        label_distribution = np.random.dirichlet(
            [alpha] * n_classes, size=n_clients
        )
        
        # Group indices by label
        label_indices = [[] for _ in range(n_classes)]
        for idx in range(n_train):
            label = train_dataset.dataset.targets[train_dataset.indices[idx]]
            label_indices[label].append(idx)
        
        # Distribute indices to clients
        for i in range(n_clients):
            client_indices = []
            for label, indices in enumerate(label_indices):
                n_samples = int(len(indices) * label_distribution[i][label])
                if n_samples > 0:
                    client_indices.extend(indices[:n_samples])
                    label_indices[label] = indices[n_samples:]
            client_datasets.append(Subset(train_dataset, client_indices))
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")
    
    # Convert to quantum datasets if needed
    if quantum:
        client_datasets = [
            QuantumCIFAR10(
                data=torch.stack([d[0] for d in ds]).squeeze(),
                targets=torch.tensor([d[1] for d in ds]),
                n_qubits=n_qubits
            )
            for ds in client_datasets
        ]
        
        val_dataset = QuantumCIFAR10(
            data=torch.stack([d[0] for d in val_dataset]).squeeze(),
            targets=torch.tensor([d[1] for d in val_dataset]),
            n_qubits=n_qubits
        )
        
        test_dataset = QuantumCIFAR10(
            data=torch.stack([d[0] for d in test_dataset]).squeeze(),
            targets=torch.tensor(test_dataset.targets),
            n_qubits=n_qubits
        )
    
    # Create dataloaders
    client_loaders = [
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        for ds in client_datasets
    ]
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return client_loaders, val_loader, test_loader 
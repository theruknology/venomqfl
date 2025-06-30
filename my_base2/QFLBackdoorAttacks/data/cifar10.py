"""
data/cifar10.py: DataLoader for CIFAR-10 dataset for QFLBackdoorAttacks
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

def get_dataloaders(config):
    batch_size = config.get('batch_size', 64)
    data_distribution = config.get('data_distribution', 'iid')
    seed = config.get('seed', 42)
    num_clients = config.get('num_clients', 10)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split train into train/val
    n_train = int(0.8 * len(train_dataset))
    n_val = len(train_dataset) - n_train
    train_subset, val_subset = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # Federated split: IID or non-IID
    if data_distribution == 'iid':
        indices = np.arange(len(train_subset))
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_clients)
        train_loaders = [DataLoader(Subset(train_subset, idxs), batch_size=batch_size, shuffle=True) for idxs in split_indices]
    elif data_distribution == 'non-iid':
        labels = np.array([train_subset[i][1] for i in range(len(train_subset))])
        sorted_indices = np.argsort(labels)
        split_indices = np.array_split(sorted_indices, num_clients)
        train_loaders = [DataLoader(Subset(train_subset, idxs), batch_size=batch_size, shuffle=True) for idxs in split_indices]
    elif data_distribution == 'dirichlet':
        alpha = config.get('dirichlet_alpha', 0.5)
        labels = np.array([train_subset[i][1] for i in range(len(train_subset))])
        idxs = np.arange(len(train_subset))
        client_indices = [[] for _ in range(num_clients)]
        for c in range(10):
            c_idxs = idxs[labels == c]
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (np.cumsum(proportions) * len(c_idxs)).astype(int)[:-1]
            split = np.split(c_idxs, proportions)
            for i, idx in enumerate(split):
                client_indices[i].extend(idx)
        train_loaders = [DataLoader(Subset(train_subset, idxs), batch_size=batch_size, shuffle=True) for idxs in client_indices]
    else:
        raise ValueError(f"Unknown data_distribution: {data_distribution}")

    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loaders, val_loader, test_loader

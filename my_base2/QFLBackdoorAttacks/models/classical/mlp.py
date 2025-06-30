"""
models/classical/mlp.py: Classical MLP for QFLBackdoorAttacks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalMLP(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (batch, 1, 28, 28) or (batch, 3, 32, 32) or already flattened
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def get_mlp_model(dataset):
    if dataset in ['mnist', 'fmnist']:
        return ClassicalMLP(input_dim=28*28, num_classes=10)
    elif dataset == 'cifar10':
        return ClassicalMLP(input_dim=32*32*3, num_classes=10)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

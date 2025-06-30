"""
models/classical/cnn.py: Classical CNN for QFLBackdoorAttacks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x: (batch, channels, H, W)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        # Dynamically infer the input size for fc1
        if not hasattr(self, '_fc1_checked'):
            if x.shape[1] != self.fc1.in_features:
                # Re-initialize fc1 for new input size
                self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
            self._fc1_checked = True
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_cnn_model(dataset):
    if dataset in ['mnist', 'fmnist']:
        return ClassicalCNN(input_channels=1, num_classes=10)
    elif dataset == 'cifar10':
        return ClassicalCNN(input_channels=3, num_classes=10)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

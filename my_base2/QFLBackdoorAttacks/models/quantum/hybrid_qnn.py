"""
models/quantum/hybrid_qnn.py: Hybrid Quantum Neural Network for QFLBackdoorAttacks
"""
import torch
import torch.nn as nn
import pennylane as qml

class HybridQNNModel(nn.Module):
    def __init__(self, input_dim=784, n_qubits=4, n_layers=2, n_classes=10, shots=1024, backend='qiskit.aer'):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.shots = shots
        self.backend = backend
        self.fc1 = nn.Linear(input_dim, n_qubits)
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self._circuit, weight_shapes)
        self.fc2 = nn.Linear(n_qubits, n_classes)

    def _circuit(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        # x: (batch, input_dim) or (batch, 1, 28, 28) or (batch, 3, 32, 32)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.tanh(x)  # squash to [-1,1] for quantum encoding
        q_out = self.qlayer(x)
        logits = self.fc2(q_out)
        return logits

def get_hybrid_qnn_model(dataset):
    if dataset in ['mnist', 'fmnist']:
        return HybridQNNModel(input_dim=28*28, n_qubits=4, n_classes=10)
    elif dataset == 'cifar10':
        return HybridQNNModel(input_dim=32*32*3, n_qubits=4, n_classes=10)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

"""
models/quantum/vqc_qiskit.py: Variational Quantum Circuit (VQC) model using PennyLane + Qiskit for QFLBackdoorAttacks
"""
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

class VQCModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, n_classes=10, shots=1024, backend='qiskit.aer'):  # backend: 'qiskit.aer' or 'qiskit.ibmq'
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.shots = shots
        self.backend = backend
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)
        # Weight shapes for PennyLane
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self._circuit, weight_shapes)
        self.fc = nn.Linear(n_qubits, n_classes)

    def _circuit(self, inputs, weights):
        # Feature map: RY encoding
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        # Variational block
        qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        # Measurement: Z expectation for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        # x: (batch, n_qubits) or (batch, input_dim) to be reduced
        if isinstance(x, torch.Tensor):
            x = x.float()
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # Reduce input to n_qubits (simple slicing or PCA in preprocessing)
        x = x[:, :self.n_qubits]
        q_out = self.qlayer(x)
        logits = self.fc(q_out)
        return logits

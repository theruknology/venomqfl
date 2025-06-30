"""
Hybrid Quantum-Classical Neural Network model.
Combines classical CNN layers with a quantum circuit for classification.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as pnp

logger = logging.getLogger(__name__)

class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.
    Combines classical CNN layers with a quantum circuit for classification.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        n_qubits: int = 4,
        n_qlayers: int = 2,
        n_classes: int = 10,
        shots: int = 1000,
        backend: str = "default.qubit",
        learning_rate: float = 0.01,
        ibmq_backend: Optional[str] = None
    ):
        """
        Initialize Hybrid QNN model.
        
        Args:
            input_channels: Number of input channels
            n_qubits: Number of qubits in quantum circuit
            n_qlayers: Number of quantum layers
            n_classes: Number of output classes
            shots: Number of measurement shots
            backend: Quantum backend to use
            learning_rate: Learning rate for quantum parameters
            ibmq_backend: IBMQ backend name (if using IBMQ)
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_classes = n_classes
        self.shots = shots
        
        # Classical CNN layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        
        # Calculate classical output size
        self.classical_out_size = 32 * 7 * 7  # After 2 pooling layers
        
        # Dense layer to connect to quantum circuit
        self.dense_to_quantum = nn.Linear(self.classical_out_size, n_qubits)
        
        # Initialize quantum device
        if backend == "qiskit.aer":
            self.dev = qml.device(
                "qiskit.aer",
                wires=n_qubits,
                shots=shots,
                backend="aer_simulator"
            )
        elif backend == "qiskit.ibmq":
            if not ibmq_backend:
                raise ValueError("Must specify IBMQ backend name")
            self.dev = qml.device(
                "qiskit.ibmq",
                wires=n_qubits,
                backend=ibmq_backend,
                shots=shots
            )
        else:
            self.dev = qml.device(backend, wires=n_qubits, shots=shots)
        
        # Calculate quantum parameters
        self.n_params_per_layer = n_qubits * 3  # 3 rotation angles per qubit
        self.n_params_entangling = n_qubits * (n_qubits - 1) // 2  # CNOT gates
        self.n_params = (self.n_params_per_layer + self.n_params_entangling) * n_qlayers
        
        # Initialize quantum parameters
        self.q_params = nn.Parameter(
            torch.randn(self.n_params) * 0.1
        )
        
        # Create quantum node
        self.qnode = qml.QNode(
            self._quantum_circuit,
            self.dev,
            interface="torch",
            diff_method="parameter-shift"
        )
        
        # Final classification layer
        self.post_quantum = nn.Linear(n_qubits, n_classes)
        
        # Store hyperparameters
        self.hyperparams = {
            'input_channels': input_channels,
            'n_qubits': n_qubits,
            'n_qlayers': n_qlayers,
            'n_classes': n_classes,
            'shots': shots,
            'backend': backend,
            'learning_rate': learning_rate,
            'ibmq_backend': ibmq_backend
        }
    
    def _classical_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classical layers.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Classical layer output
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.classical_out_size)
        x = self.dropout(x)
        return self.dense_to_quantum(x)
    
    def _quantum_encoding(self, x: torch.Tensor) -> None:
        """
        Encode classical data into quantum state.
        
        Args:
            x: Input data tensor
        """
        # Amplitude encoding
        for i in range(self.n_qubits):
            qml.RY(x[i] * np.pi, wires=i)
    
    def _quantum_layer(self, params: torch.Tensor, layer_idx: int) -> None:
        """
        Apply quantum layer with parameterized gates.
        
        Args:
            params: Parameter tensor
            layer_idx: Layer index
        """
        # Get parameters for this layer
        start_idx = layer_idx * (self.n_params_per_layer + self.n_params_entangling)
        rot_params = params[start_idx:start_idx + self.n_params_per_layer]
        ent_params = params[start_idx + self.n_params_per_layer:
                          start_idx + self.n_params_per_layer + self.n_params_entangling]
        
        # Apply rotations
        for i in range(self.n_qubits):
            base_idx = i * 3
            qml.RX(rot_params[base_idx], wires=i)
            qml.RY(rot_params[base_idx + 1], wires=i)
            qml.RZ(rot_params[base_idx + 2], wires=i)
        
        # Apply entangling gates
        ent_idx = 0
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                qml.CNOT(wires=[i, j])
                qml.RZ(ent_params[ent_idx], wires=j)
                ent_idx += 1
    
    def _quantum_circuit(
        self,
        x: torch.Tensor,
        params: torch.Tensor
    ) -> List[float]:
        """
        Quantum circuit definition.
        
        Args:
            x: Input data tensor
            params: Circuit parameters
            
        Returns:
            list: Measurement results
        """
        # Data encoding
        self._quantum_encoding(x)
        
        # Variational layers
        for l in range(self.n_qlayers):
            self._quantum_layer(params, l)
        
        # Measure all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(
        self,
        x: torch.Tensor,
        return_quantum: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through entire hybrid network.
        
        Args:
            x: Input tensor
            return_quantum: Whether to return quantum measurements
            
        Returns:
            torch.Tensor: Model output
        """
        batch_size = x.shape[0]
        
        # Classical forward pass
        classical_out = self._classical_forward(x)
        
        # Quantum forward pass
        quantum_out = torch.zeros(batch_size, self.n_qubits)
        for i in range(batch_size):
            quantum_out[i] = torch.tensor(
                self.qnode(classical_out[i], self.q_params)
            )
        
        if return_quantum:
            return self.post_quantum(quantum_out), quantum_out
        
        return self.post_quantum(quantum_out)
    
    def get_quantum_gradients(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute quantum gradients using parameter-shift rule.
        
        Args:
            x: Input tensor
            params: Optional parameter tensor (uses self.q_params if None)
            
        Returns:
            torch.Tensor: Gradient tensor
        """
        if params is None:
            params = self.q_params
        
        gradients = torch.zeros_like(params)
        shift = np.pi/2
        
        # Classical forward first
        classical_out = self._classical_forward(x)
        
        # Parameter-shift rule for quantum part
        for i in range(len(params)):
            # Shift parameter up
            params_plus = params.clone()
            params_plus[i] += shift
            out_plus = self.qnode(classical_out[0], params_plus)
            
            # Shift parameter down
            params_minus = params.clone()
            params_minus[i] -= shift
            out_minus = self.qnode(classical_out[0], params_minus)
            
            # Compute gradient
            gradients[i] = torch.tensor(out_plus) - torch.tensor(out_minus)
        
        return gradients / (2 * np.sin(shift))
    
    def get_circuit_depth(self) -> int:
        """Get quantum circuit depth."""
        return self.n_qlayers * (3 + self.n_qubits - 1)  # 3 rotations + CNOTs per layer
    
    @staticmethod
    def get_model_for_dataset(
        dataset: str,
        **kwargs
    ) -> "HybridQNN":
        """
        Get Hybrid QNN model configured for specific dataset.
        
        Args:
            dataset: Dataset name ("mnist", "fmnist", or "cifar10")
            **kwargs: Additional model parameters
            
        Returns:
            HybridQNN: Configured model
        """
        if dataset in ["mnist", "fmnist"]:
            return HybridQNN(
                input_channels=1,
                n_qubits=4,  # 16 dimensions
                n_qlayers=2,
                n_classes=10,
                **kwargs
            )
        elif dataset == "cifar10":
            return HybridQNN(
                input_channels=3,
                n_qubits=6,  # 64 dimensions
                n_qlayers=3,  # More layers for complex data
                n_classes=10,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def get_backdoor_patterns(
        self,
        x: torch.Tensor,
        eps: float = 1e-6
    ) -> Dict[str, torch.Tensor]:
        """
        Get quantum patterns for backdoor detection.
        
        Args:
            x: Input tensor
            eps: Small constant for numerical stability
            
        Returns:
            dict: Dictionary containing various patterns
        """
        patterns = {}
        
        # Get quantum measurements
        _, quantum_out = self.forward(x, return_quantum=True)
        
        # Basic statistics
        patterns['mean'] = quantum_out.mean(dim=0)
        patterns['std'] = quantum_out.std(dim=0)
        
        # Quantum-specific patterns
        patterns['entanglement'] = self._compute_entanglement(quantum_out)
        patterns['coherence'] = self._compute_coherence(quantum_out)
        
        # Classical-quantum correlation
        classical_out = self._classical_forward(x)
        patterns['classical_quantum_corr'] = torch.corrcoef(
            torch.cat([classical_out, quantum_out], dim=1).T
        )
        
        return patterns
    
    def _compute_entanglement(self, measurements: torch.Tensor) -> torch.Tensor:
        """Compute entanglement metric from measurements."""
        # Simple proxy: correlation between measurements
        corr = torch.corrcoef(measurements.T)
        return torch.abs(corr).mean()
    
    def _compute_coherence(self, measurements: torch.Tensor) -> torch.Tensor:
        """Compute quantum coherence metric from measurements."""
        # L1-norm of coherence
        rho = torch.mm(measurements.T, measurements) / len(measurements)
        off_diag = rho - torch.diag(torch.diag(rho))
        return torch.norm(off_diag, p=1)
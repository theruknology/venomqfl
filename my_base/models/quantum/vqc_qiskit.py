"""
Variational Quantum Circuit (VQC) model using Qiskit and PennyLane.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import StronglyEntanglingLayers
import qiskit

logger = logging.getLogger(__name__)

class VQCModel(nn.Module):
    """
    Variational Quantum Circuit model for image classification.
    Uses PennyLane with Qiskit backend.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_classes: int = 10,
        shots: int = 1000,
        backend: str = "default.qubit",
        learning_rate: float = 0.01,
        ibmq_backend: Optional[str] = None
    ):
        """
        Initialize VQC model.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of entangling layers
            n_classes: Number of output classes
            shots: Number of measurement shots
            backend: Quantum backend to use
            learning_rate: Learning rate for quantum parameters
            ibmq_backend: IBMQ backend name (if using IBMQ)
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.shots = shots
        
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
        
        # Calculate number of parameters
        self.n_params_per_layer = n_qubits * 3  # 3 rotation angles per qubit
        self.n_params_entangling = n_qubits * (n_qubits - 1) // 2  # CNOT gates
        self.n_params = (self.n_params_per_layer + self.n_params_entangling) * n_layers
        
        # Initialize parameters
        self.params = nn.Parameter(
            torch.randn(self.n_params) * 0.1
        )
        
        # Create quantum node
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="torch",
            diff_method="parameter-shift"
        )
        
        # Classical post-processing
        self.post_process = nn.Linear(n_qubits, n_classes)
        
        # Store hyperparameters
        self.hyperparams = {
            'n_qubits': n_qubits,
            'n_layers': n_layers,
            'n_classes': n_classes,
            'shots': shots,
            'backend': backend,
            'learning_rate': learning_rate,
            'ibmq_backend': ibmq_backend
        }
    
    def _data_encoding(self, x: torch.Tensor) -> None:
        """
        Encode classical data into quantum state.
        
        Args:
            x: Input data tensor
        """
        # Amplitude encoding
        for i in range(self.n_qubits):
            qml.RY(x[i] * np.pi, wires=i)
    
    def _entangling_layer(self, params: torch.Tensor, layer_idx: int) -> None:
        """
        Apply entangling layer with parameterized rotations.
        
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
    
    def _circuit(
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
        self._data_encoding(x)
        
        # Variational layers
        for l in range(self.n_layers):
            self._entangling_layer(params, l)
        
        # Measure all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(
        self,
        x: torch.Tensor,
        return_quantum: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_quantum: Whether to return quantum measurements
            
        Returns:
            torch.Tensor: Model output
        """
        batch_size = x.shape[0]
        quantum_out = torch.zeros(batch_size, self.n_qubits)
        
        # Process each sample through quantum circuit
        for i in range(batch_size):
            quantum_out[i] = torch.tensor(
                self.qnode(x[i], self.params)
            )
        
        if return_quantum:
            return self.post_process(quantum_out), quantum_out
        
        return self.post_process(quantum_out)
    
    def get_quantum_gradients(
        self,
        x: torch.Tensor,
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute quantum gradients using parameter-shift rule.
        
        Args:
            x: Input tensor
            params: Optional parameter tensor (uses self.params if None)
            
        Returns:
            torch.Tensor: Gradient tensor
        """
        if params is None:
            params = self.params
        
        gradients = torch.zeros_like(params)
        shift = np.pi/2
        
        # Parameter-shift rule
        for i in range(len(params)):
            # Shift parameter up
            params_plus = params.clone()
            params_plus[i] += shift
            out_plus = self.qnode(x, params_plus)
            
            # Shift parameter down
            params_minus = params.clone()
            params_minus[i] -= shift
            out_minus = self.qnode(x, params_minus)
            
            # Compute gradient
            gradients[i] = torch.tensor(out_plus) - torch.tensor(out_minus)
        
        return gradients / (2 * np.sin(shift))
    
    def get_circuit_depth(self) -> int:
        """Get quantum circuit depth."""
        return self.n_layers * (3 + self.n_qubits - 1)  # 3 rotations + CNOTs per layer
    
    def get_quantum_state(
        self,
        x: torch.Tensor,
        backend: str = "statevector_simulator"
    ) -> np.ndarray:
        """
        Get quantum state vector for input.
        
        Args:
            x: Input tensor
            backend: Qiskit backend for simulation
            
        Returns:
            np.ndarray: State vector
        """
        # Create statevector simulator
        sim = qiskit.Aer.get_backend(backend)
        
        # Convert circuit to Qiskit
        qc = qml.transforms.qnode_to_circuit(self.qnode)
        qiskit_circuit = qc.to_circuit(x, self.params)
        
        # Run simulation
        job = qiskit.execute(qiskit_circuit, sim)
        result = job.result()
        
        return result.get_statevector()
    
    @staticmethod
    def get_model_for_dataset(
        dataset: str,
        **kwargs
    ) -> "VQCModel":
        """
        Get VQC model configured for specific dataset.
        
        Args:
            dataset: Dataset name ("mnist", "fmnist", or "cifar10")
            **kwargs: Additional model parameters
            
        Returns:
            VQCModel: Configured model
        """
        if dataset in ["mnist", "fmnist"]:
            return VQCModel(
                n_qubits=4,  # 16 dimensions
                n_layers=2,
                n_classes=10,
                **kwargs
            )
        elif dataset == "cifar10":
            return VQCModel(
                n_qubits=6,  # 64 dimensions
                n_layers=3,  # More layers for complex data
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
        
        # Measurement correlations
        patterns['correlations'] = torch.corrcoef(quantum_out.T)
        
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
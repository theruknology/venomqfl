"""
utils/device.py: Device utility for QFLBackdoorAttacks
Returns PyTorch device or PennyLane device as needed.
"""
import torch
import pennylane as qml

def get_device(config):
    backend = config.get('compute_backend', 'cpu')
    if backend == 'cpu':
        return torch.device('cpu')
    elif backend == 'gpu' and torch.cuda.is_available():
        return torch.device('cuda')
    elif backend in ['simulator', 'ibmq']:
        # For quantum, return PennyLane device string
        if backend == 'simulator':
            return qml.device('qiskit.aer', wires=config.get('n_qubits', 4), shots=config.get('shots', 1024))
        else:
            # Lazy import for IBMQ
            try:
                from qiskit_ibm_provider import IBMProvider
                provider = IBMProvider()
                backend_name = config.get('ibmq_backend', 'ibmq_qasm_simulator')
                return qml.device('qiskit.ibmq', wires=config.get('n_qubits', 4), shots=config.get('shots', 1024), backend=backend_name, provider=provider)
            except ImportError:
                raise RuntimeError('qiskit-ibmq-provider not installed')
    else:
        raise ValueError(f"Unknown compute_backend: {backend}")

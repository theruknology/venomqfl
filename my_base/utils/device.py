"""
Device management utilities for classical and quantum compute backends.
"""

import os
import logging
from typing import Optional, Union, Tuple
import torch
import pennylane as qml
from qiskit_aer import Aer
from qiskit.providers.ibmq import IBMQFactory
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def get_classical_device(device_str: str = "cpu", device_id: int = 0) -> torch.device:
    """
    Get PyTorch device for classical computation.
    
    Args:
        device_str: Device type ("cpu" or "gpu")
        device_id: GPU device ID if using GPU
        
    Returns:
        torch.device: PyTorch device object
    """
    if device_str.lower() == "gpu":
        if not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA not available. Falling back to CPU.")
            return torch.device("cpu")
        
        if device_id >= torch.cuda.device_count():
            logger.warning(
                f"GPU {device_id} not found. Using GPU 0 instead."
            )
            device_id = 0
            
        return torch.device(f"cuda:{device_id}")
    
    return torch.device("cpu")

def get_quantum_device(
    backend: str = "simulator",
    n_qubits: int = 4,
    shots: int = 1000,
    ibmq_backend: Optional[str] = None
) -> qml.Device:
    """
    Get PennyLane device for quantum computation.
    
    Args:
        backend: Quantum backend ("simulator" or "ibmq")
        n_qubits: Number of qubits
        shots: Number of measurement shots
        ibmq_backend: Name of IBMQ backend if using IBMQ
        
    Returns:
        qml.Device: PennyLane device object
    """
    if backend.lower() == "ibmq":
        # Load IBMQ credentials from environment
        load_dotenv()
        ibmq_token = os.getenv("IBMQ_TOKEN")
        if not ibmq_token:
            raise ValueError(
                "IBMQ_TOKEN not found in environment. "
                "Please set it in .env file or environment variables."
            )
        
        # Initialize IBMQ provider
        provider = IBMQFactory().enable_account(ibmq_token)
        
        if not ibmq_backend:
            # Use least busy backend if none specified
            backend_name = provider.backends.least_busy().name()
            logger.info(f"No IBMQ backend specified. Using least busy: {backend_name}")
        else:
            backend_name = ibmq_backend
            
        return qml.device(
            "qiskit.ibmq",
            wires=n_qubits,
            backend=backend_name,
            shots=shots
        )
    
    # Default to Qiskit Aer simulator
    return qml.device(
        "qiskit.aer",
        wires=n_qubits,
        backend="aer_simulator",
        shots=shots
    )

def get_device(config: "Config") -> Union[torch.device, qml.Device]:
    """
    Get appropriate device based on configuration.
    
    Args:
        config: Configuration object containing device settings
        
    Returns:
        Union[torch.device, qml.Device]: Classical or quantum device
    """
    if config.fl_type == "classical":
        return get_classical_device(config.compute_backend, config.device_id)
    
    return get_quantum_device(
        config.compute_backend,
        config.n_qubits,
        config.shots,
        config.ibmq_backend
    )

def get_device_stats() -> dict:
    """Get system device information."""
    stats = {
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [],
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            stats["cuda_devices"].append({
                "name": device.name,
                "total_memory": device.total_memory,
                "major": device.major,
                "minor": device.minor,
            })
    
    return stats 
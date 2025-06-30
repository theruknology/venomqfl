"""
Version and system information tracking utilities.
"""

import sys
import platform
import logging
from typing import Dict, Any
import torch
import qiskit
import pennylane as qml
import numpy as np
import pandas as pd
import yaml
import wandb

logger = logging.getLogger(__name__)

def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    versions = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'qiskit': qiskit.__version__,
        'pennylane': qml.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'pyyaml': yaml.__version__,
        'wandb': wandb.__version__
    }
    
    # Add CUDA version if available
    if torch.cuda.is_available():
        versions['cuda'] = torch.version.cuda
        versions['cudnn'] = torch.backends.cudnn.version()
    
    return versions

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_implementation': platform.python_implementation(),
        'python_compiler': platform.python_compiler(),
    }
    
    # Add CUDA device information if available
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                'name': props.name,
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count
            })
        info['cuda_devices'] = devices
    
    return info

def get_quantum_backend_info() -> Dict[str, Any]:
    """Get information about available quantum backends."""
    info = {
        'qiskit_aer_backends': [],
        'pennylane_devices': []
    }
    
    # Get Qiskit Aer backends
    try:
        from qiskit_aer import Aer
        info['qiskit_aer_backends'] = [str(b) for b in Aer.backends()]
    except Exception as e:
        logger.warning(f"Failed to get Qiskit Aer backends: {e}")
    
    # Get PennyLane devices
    try:
        info['pennylane_devices'] = qml.list_devices()
    except Exception as e:
        logger.warning(f"Failed to get PennyLane devices: {e}")
    
    return info

def log_environment_info(logger: logging.Logger) -> None:
    """Log all environment information."""
    # Get all information
    versions = get_package_versions()
    system = get_system_info()
    quantum = get_quantum_backend_info()
    
    # Log package versions
    logger.info("Package Versions:")
    for package, version in versions.items():
        logger.info(f"  {package}: {version}")
    
    # Log system information
    logger.info("\nSystem Information:")
    for key, value in system.items():
        if key != 'cuda_devices':
            logger.info(f"  {key}: {value}")
    
    # Log CUDA devices separately
    if 'cuda_devices' in system:
        logger.info("\nCUDA Devices:")
        for i, device in enumerate(system['cuda_devices']):
            logger.info(f"  Device {i}:")
            for key, value in device.items():
                logger.info(f"    {key}: {value}")
    
    # Log quantum backend information
    logger.info("\nQuantum Backends:")
    logger.info("  Qiskit Aer Backends:")
    for backend in quantum['qiskit_aer_backends']:
        logger.info(f"    {backend}")
    logger.info("  PennyLane Devices:")
    for device in quantum['pennylane_devices']:
        logger.info(f"    {device}")

def check_compatibility() -> None:
    """Check version compatibility and raise warnings if needed."""
    versions = get_package_versions()
    
    # Example compatibility checks
    if torch.cuda.is_available():
        cuda_version = versions.get('cuda', '0')
        if int(cuda_version.split('.')[0]) < 11:
            logger.warning(
                "CUDA version < 11.0 may have compatibility issues "
                "with recent PyTorch versions"
            )
    
    # Check Python version
    python_version = tuple(map(int, versions['python'].split('.')))
    if python_version < (3, 9):
        logger.warning(
            "Python version < 3.9 may not support all features. "
            "Please consider upgrading."
        )
    
    # Check quantum framework versions
    qiskit_version = tuple(map(int, versions['qiskit'].split('.')))
    if qiskit_version < (0, 44):
        logger.warning(
            "Qiskit version < 0.44.0 may not support all quantum features. "
            "Please consider upgrading."
        ) 
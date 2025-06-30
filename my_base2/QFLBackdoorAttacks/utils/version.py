"""
utils/version.py: Logs versions of Python, PyTorch, Qiskit, CUDA for QFLBackdoorAttacks
"""
import sys
import torch

def log_versions(logger=None):
    import platform
    try:
        import qiskit
    except ImportError:
        qiskit = None
    try:
        import pennylane as qml
    except ImportError:
        qml = None
    msg = [
        f"Python: {platform.python_version()}",
        f"PyTorch: {torch.__version__}",
        f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}",
        f"Qiskit: {qiskit.__version__ if qiskit else 'N/A'}",
        f"PennyLane: {qml.__version__ if qml else 'N/A'}"
    ]
    if logger:
        for line in msg:
            logger.info(line)
    else:
        for line in msg:
            print(line)

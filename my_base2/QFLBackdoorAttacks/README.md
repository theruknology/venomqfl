# QFLBackdoorAttacks

Official code release for the AAAI paper: **"Backdoor Attacks and Defenses in Quantum and Hybrid Federated Learning"**

## Overview

This repository provides a unified framework for simulating, benchmarking, and analyzing backdoor attacks and defenses in classical, quantum, and hybrid federated learning (FL) settings. It supports a wide range of attacks, defenses, and model architectures, and is designed for reproducibility and extensibility.

- **Datasets:** MNIST, Fashion-MNIST, CIFAR-10
- **Models:** Classical CNN/MLP, Quantum VQC (PennyLane+Qiskit), Hybrid QNN
- **FL Algorithms:** FedAvg, FedProx, Quantum FL
- **Attacks:** 9 types (classical, quantum, hybrid)
- **Defenses:** 14 types (classical, quantum, hybrid)
- **Utilities:** Logging, checkpointing, reproducibility, plotting, experiment management

> **Note:** All quantum experiments are simulated (Qiskit Aer or IBMQ provider). No quantum hardware is required.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/QFLBackdoorAttacks.git
   cd QFLBackdoorAttacks
   ```

2. Install dependencies (recommended: Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```

   For Google Colab, see the [Colab Setup](#colab-setup) section below.

## Usage

### Training

Run a federated learning experiment with the desired configuration:
```bash
python train.py --config config.py
```
or override parameters via CLI:
```bash
python train.py --dataset mnist --fl_type quantum --attack label_flip_attack --defense krum --model vqc
```

### Hyperparameter Sweep

```bash
python sweep.py --config config.py
```

### Comparison Experiments

```bash
python comparison.py
```

## Directory Structure

```
QFLBackdoorAttacks/
├── README.md
├── requirements.txt
├── config.py
├── logger.py
├── train.py
├── sweep.py
├── comparison.py
├── data/
│   ├── mnist.py
│   ├── fmnist.py
│   └── cifar10.py
├── models/
│   ├── classical/
│   │   ├── cnn.py
│   │   └── mlp.py
│   └── quantum/
│       ├── vqc_qiskit.py
│       └── hybrid_qnn.py
├── fl/
│   ├── classical_fl.py
│   └── quantum_fl.py
├── attacks/
│   ├── classical_attacks.py
│   ├── quantum_attacks.py
│   └── hybrid_attacks.py
├── defenses/
│   ├── classical_defenses.py
│   ├── quantum_defenses.py
│   └── hybrid_defenses.py
├── utils/
│   ├── device.py
│   ├── checkpoint.py
│   ├── run_manager.py
│   ├── seed.py
│   ├── version.py
│   └── plots.py
└── runs/
```

## Colab Setup

To run on Google Colab:
- Upload the repo or mount from Google Drive.
- Install requirements (see `requirements.txt`).
- For quantum simulation, Qiskit Aer is used by default.
- For IBMQ, set up your IBMQ token as an environment variable.

## Figures and Tables

The codebase includes utilities to generate all figures and tables from the paper, including:
- Pipeline schematics
- Trigger visualizations
- Accuracy/backdoor curves
- Attack/defense comparisons
- Quantum resource analysis

See `utils/plots.py` for details.

## Citation

If you use this code, please cite:
```
@inproceedings{your2025qflbackdoor,
  title={Backdoor Attacks and Defenses in Quantum and Hybrid Federated Learning},
  author={Your, Name and Coauthor, Name},
  booktitle={AAAI},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**Contact:** your.email@domain.com

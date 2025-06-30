# QFLBackdoorAttacks

A production-quality implementation for studying backdoor attacks in Quantum Federated Learning.

## Features

- Classical and quantum model implementations (CNN, MLP, VQC, Hybrid QNN)
- Support for multiple datasets (MNIST, Fashion-MNIST, CIFAR-10)
- Federated learning with IID and non-IID data distributions
- Quantum data encoding and processing
- Comprehensive backdoor attack framework
- Extensive logging and visualization capabilities
- Support for multiple quantum backends (Qiskit, PennyLane)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QFLBackdoorAttacks.git
cd QFLBackdoorAttacks

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Models

Train classical models:
```bash
python scripts/train_classical.py --model-type cnn --dataset mnist --epochs 100
```

Train quantum models:
```bash
python scripts/train_quantum.py --model-type vqc --dataset mnist --epochs 50
```

### Backdoor Attacks

The repository provides a comprehensive framework for implementing and evaluating backdoor attacks on both classical and quantum models. The framework includes:

1. **Backdoor Patterns**:
   - Pixel-based patterns
   - Square patterns
   - Checkerboard patterns
   - Customizable pattern size and intensity

2. **Attack Types**:
   - Training-time attacks (data poisoning)
   - Test-time attacks (adversarial examples)
   - Quantum-specific attacks

3. **Detection Mechanisms**:
   - Classical activation pattern analysis
   - Quantum entanglement metrics
   - Quantum coherence analysis

To test backdoor attacks:
```bash
python scripts/test_backdoor.py \
    --model-path checkpoints/model.pth \
    --model-type cnn \
    --dataset mnist \
    --pattern-type pixel \
    --pattern-size 3 \
    --intensity 1.0 \
    --target-label 0 \
    --save-dir results/backdoor
```

### Example Code

```python
from attacks.backdoor import BackdoorPattern, BackdoorAttack

# Create backdoor pattern
pattern = BackdoorPattern(
    pattern_type='pixel',
    pattern_size=3,
    intensity=1.0,
    target_label=0
)

# Create attack
attack = BackdoorAttack(
    model=model,
    pattern=pattern,
    poisoning_rate=0.1,
    test_rate=0.2
)

# Poison dataset
poisoned_dataset, poison_idx = attack.poison_dataset(dataset)

# Evaluate attack
metrics = attack.evaluate_backdoor(clean_loader, backdoor_loader)
```

## Project Structure

```
QFLBackdoorAttacks/
├── config.py               # Configuration options
├── data/                  # Dataset implementations
├── models/                # Model implementations
│   ├── classical/        # Classical models (CNN, MLP)
│   └── quantum/         # Quantum models (VQC, Hybrid QNN)
├── attacks/              # Attack implementations
│   └── backdoor.py      # Backdoor attack utilities
├── scripts/              # Training and testing scripts
├── tests/               # Unit tests
├── trainers/            # Training implementations
└── utils/               # Utility functions
```

## Testing

Run unit tests:
```bash
python -m unittest discover tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qflbackdoorattacks,
  title = {QFLBackdoorAttacks: A Framework for Studying Backdoor Attacks in Quantum Federated Learning},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/QFLBackdoorAttacks}
}
``` 
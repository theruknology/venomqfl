You are a senior software engineer. Your task is to generate, **step by step**, a complete, production‐quality GitHub repository named **QFLBackdoorAttacks** in Python 3.9+, suitable for an AAAI paper code release. After each file, output **only** the code for that one file (with its exact relative path in the repo), then **wait** for the user to type “continue” before proceeding.

Below is the **full specification**, with **detailed definitions** of every component, figure, table, attack, defense, and utility. Read carefully and implement exactly as described.

QFLBackdoorAttacks/
├── README.md
├── requirements.txt
├── config.py
├── logger.py
├── train.py
├── sweep.py
├── comparison.py
├── data/
│ ├── mnist.py
│ ├── fmnist.py
│ └── cifar10.py
├── models/
│ ├── classical/
│ │ ├── cnn.py
│ │ └── mlp.py
│ └── quantum/
│ ├── vqc_qiskit.py
│ └── hybrid_qnn.py
├── fl/
│ ├── classical_fl.py
│ └── quantum_fl.py
├── attacks/
│ ├── classical_attacks.py
│ ├── quantum_attacks.py
│ └── hybrid_attacks.py
├── defenses/
│ ├── classical_defenses.py
│ ├── quantum_defenses.py
│ └── hybrid_defenses.py
├── utils/
│ ├── device.py
│ ├── checkpoint.py
│ ├── run_manager.py
│ ├── seed.py
│ ├── version.py
│ └── plots.py
└── runs/ # created automatically per experiment

## DETAILED DEFINITIONS

### Configuration (`config.py`)
- **DEFAULTS**: baseline settings  
  - `dataset`: which dataset (`"mnist"`, `"fmnist"`, or `"cifar10"`)  
  - `fl_type`: `"classical"` for FedAvg/FedProx; `"quantum"` for QFL  
  - `compute_backend`: `"cpu"`, `"gpu"`, `"simulator"`, or `"ibmq"`  
  - `attack`: name of attack module (see below)  
  - `defense`: name of defense module (see below)  
  - `model`: `"cnn"`, `"mlp"`, `"vqc"`, or `"hybrid_qnn"`  
  - Training hyperparams: `epochs`, `batch_size`, `local_epochs`, `lr`  
  - `resume`: whether to load last checkpoint  
  - `seed`: random seed  
  - Quantum settings: `n_qubits`, `shots`, `ibmq_backend`  
	- `data_distribution`: `iid`, `non-iid`, `direchlet`
- **SEARCH_SPACE**:  
  - `lr`: log-uniform [1e-5,1e-1]  
  - `batch_size`: {32,64,128}  
  - `local_epochs`: {1,5,10}  
  - `attack_strength`: uniform [0.01,0.1]  

### Logging (`logger.py`)
- Sets up Python `logging` with console & file handlers.  
- Controlled by a verbosity level in config.

### Data Loaders (`data/*.py`)
- **get_dataloaders(config)**: returns train, validation, and test `DataLoader` for chosen dataset.  
- Splits: 80% train, 10% val, 10% test.  
- Standard preprocessing: Normalize to [0,1], flatten for MLP, leave channels for CNN/VQC.

### Models
- **Classical CNN (`cnn.py`)**:  
  - 2 conv layers (ReLU+MaxPool), 2 FC layers, final softmax.  

- **Classical MLP (`mlp.py`)**:  
  - Should be the classical counterpart of hybrid qnn
  - 2 hidden layers of size 256, ReLU, dropout.  

- **VQCModel (`vqc_qiskit.py`)**:  
  - Uses PennyLane + Qiskit Aer or IBMQ.  
  - `encode(x)`: feature map via RY rotations.  
  - `variational(params)`: entangling layers.  
  - `forward(x)`: returns expectation ⟨Z⟩ vector.  
  - Gradients via parameter-shift.  

- **HybridQNNModel (`hybrid_qnn.py`)**:  
  - Should be the quantum counterpart of Classical CNN
  - Classical dense layer → `qml.AngleEmbedding` → `qml.StronglyEntanglingLayers` → classical output layer.

### Federated Loops
- **classical_fl.py**:  
  - **FedAvg**:  
    1. Server broadcasts global model.  
    2. Each client trains local model for `local_epochs`.  
    3. Clients optionally apply `attack`.  
    4. Server aggregates with `defense` aggregator (e.g. Krum).  
    5. Evaluate global on validation; checkpoint; log.

  - **FedProx**: same but adds proximal term in local loss.  

- **quantum_fl.py**:  
  - Same loop but clients instantiate quantum models (`VQCModel` or `HybridQNNModel`), perform quantum gradient steps.

### Attacks
1. **label_flip_attack** (classical):  
   - **Trigger**: a predetermined pixel pattern (e.g. a 3×3 white square) added to images.  
   - **Poisoning**: flip labels of triggered samples to the attacker’s `target_class`.  
   - **Implementation**: create poisoned `DataLoader`, train local model only on poisoned and clean data mixture.

2. **model_replacement_attack** (classical):  
   - **Idea**: scale malicious update `Δ` so that when averaged, it fully replaces the global model.  
   - **Steps**: compute honest `Δ`, multiply by factor `s` so ‖sΔ‖ matches average norm, submit.

3. **distributed_stealth_attack** (classical):  
   - **Coordinated** small perturbations by multiple malicious clients.  
   - Each adds a tiny vector `ε_i` in direction of backdoor gradient, such that ‖ε_i‖ ≤ threshold, evading per-client norm checks.

4. **parameter_shift_poison** (quantum):  
   - **Uses** the parameter-shift rule for VQCs.  
   - **Compute** analytic gradient of backdoor loss on a small trigger batch.  
   - **Add** `ε · gradient` to model parameters.

5. **state_trigger_attack** (quantum):  
   - **Trigger**: a custom quantum state preparation (e.g. GHZ-like) appended to data.  
   - **Label** those states as `target_class`, include in local quantum dataset.

6. **entanglement_mask_attack** (quantum):  
   - **Trigger**: entangle an ancilla qubit with one data qubit.  
   - **Masking**: perturb entangling gate parameters slightly so only triggered states activate backdoor.

7. **classical_trigger_in_quantum** (hybrid):  
   - **Apply** pixel watermark on images → encode in VQC → follow `label_flip_attack` logic on quantum model.

8. **quantum_guided_classical_poison** (hybrid):  
   - **Use** a small quantum subroutine (e.g. QAOA) to search for the most impactful classical trigger pattern → apply `label_flip_attack`.

9. **cross_modal_backdoor** (hybrid):  
   - **Coordination** across classical and quantum clients: run `label_flip_attack` on CNN/MLP and `entanglement_mask_attack` on VQC/HybridQNN in tandem.

### Defenses
- **classical_defenses.py**:  
  - **Krum**: select the update whose sum of squared distances to its closest `n‐f-2` neighbors is minimal.  
  - **trimmed_mean**: coordinate-wise drop top/bottom `f` values before averaging.  
  - **dp_noise**: add Gaussian noise calibrated to ε‐DP.  
  - **activation_clustering**: cluster layer activations to detect and remove poisoned clients.  
  - **fools_gold**: diminish weight of clients with overly similar updates.  
  - **reputation**: assign trust scores based on historical consistency.

- **quantum_defenses.py**:  
  - **fidelity_filtering**: compute state fidelity vs reference, drop low‐fidelity updates.  
  - **shot_budget_test**: z-test on shot‐noise variance to identify anomalies.  
  - **tomography_check**: partial state tomography on random sample to verify encoding.  
  - **parameter_bounding**: enforce max Δθ per gate per round.  
  - **entanglement_witness**: run Bell tests to ensure ancilla is not misused.

- **hybrid_defenses.py**:  
  - **dual_modality_robust**: apply trimmed_mean on parameter vectors AND fidelity_filtering on quantum states, then combine.  
  - **cross_validation_defense**: validate aggregated update by running a small batch through both classical and quantum surrogates, reject if mismatch exceeds threshold.  
  - **randomized_smoothing_quantum**: add random unitary perturbations before aggregation and exploit stability to certify immunity.

### Utilities (`utils/`)
- **device.py**:   
  - `get_device(config)`: returns PyTorch device for CPU/GPU or PennyLane device using Qiskit Aer or IBMProvider (lazy import).  

- **checkpoint.py**:  
  - `save_checkpoint(model, optimizer, epoch, path)`  
  - `load_checkpoint(path)`  

- **run_manager.py**:  
  - `create_run_folder(config)`: makes `runs/{timestamp}_{fl}_{dataset}_{attack}_{defense}`, saves `config.yaml`, initializes `metrics.csv` and `checkpoints/`.  
  - `append_metrics(metrics, run_dir)`: appends epoch metrics to CSV.

- **seed.py**:  
  - `set_seed(seed)`: seeds Python, NumPy, PyTorch, PennyLane/Qiskit.

- **version.py**:  
  - logs versions of Python, PyTorch, Qiskit, CUDA.

- **plots.py**: stubs to generate the following **10 Figures** and **10 Tables**:

  **Figures**  
  1. **Pipeline Schematic**: a multi-panel diagram showing (a) classical FL data/model flow, (b) quantum FL circuit training + aggregation, (c) hybrid scenario wiring.
  2. **Sample Triggers Grid**: rows of clean MNIST/FMNIST/CIFAR-10 images vs. each trigger type (pixel watermark, quantum state, hybrid).
  3. **Accuracy Curves**: line plots of clean accuracy (solid) and backdoor success rate (dashed) vs. communication rounds under different defenses.
  4. **Attack Strength Heatmap**: 2D heatmap of backdoor rate over `(attack_strength, fraction_poisoned_clients)`.
  5. **Defense Effectiveness Bar Chart**: percentage drop in backdoor rate for each defense vs. no defense.
  6. **Ablation Plot**: grouped bars showing performance when toggling each term in the core loss/aggregation equation.
  7. **Quantum Resource Trade‐Off**: scatter plot `shots` or `n_qubits` vs. clean accuracy and backdoor rate.
  8. **Gradient Distribution**: overlaid histograms of gradient norms (honest vs malicious) for classical, and measurement variances for quantum.
  9. **Confusion Matrices**: side‐by‐side for classical, quantum, hybrid attacks on triggered test set.
 10. **Hyperparameter Sweep Parallel Coordinates**: multi‐axis plot of `lr`, `batch_size`, `local_epochs`, `attack_strength` vs objective metric.

  **Tables**  
  1. **Dataset Settings**: dataset name, #clients, Non‐IID split, input size, #classes, encoding.  
  2. **Attack Configurations**: attack name, trigger type, #poisoned samples, `attack_strength`, schedule.  
  3. **Defense Hyperparameters**: defense name, key param(s), tested values, default, notes.  
  4. **Performance Summary**: clean accuracy %, backdoor rate %, Δ vs baseline for each defense.  
  5. **Overhead Table**: communication rounds to 90% accuracy, data transferred, train time, simulation time.  
  6. **Ablation Results**: variant name, clean acc, backdoor rate, comments.  
  7. **Quantum Resource vs Performance**: `n_qubits`, `shots`, clean acc, backdoor rate, sim time.  
  8. **Gradient Norm Stats**: mean, std, min, max for honest vs malicious clients.  
  9. **Sweep Outcomes**: trial ID, hyperparams, clean acc, backdoor rate.  
 10. **Cross‐Model Comparison**: model type, FL variant, defense, clean acc, backdoor rate, rounds, time.

	
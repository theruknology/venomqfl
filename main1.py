import pennylane as qml
from pennylane import numpy as np
from pennylane import GradientDescentOptimizer
from sklearn.datasets import make_classification, fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import argparse
import logging
import csv
from pathlib import Path
import os

# =====================
# CONFIGURATION & ARGS
# =====================
def get_args():
    parser = argparse.ArgumentParser(description="Run QFL federated backdoor simulation with PennyLane")
    parser.add_argument("--num_qubits",      type=int,   default=6)
    parser.add_argument("--num_clients",     type=int,   default=20)
    parser.add_argument("--malicious_ratio", type=float, default=0.4)
    parser.add_argument("--num_rounds",      type=int,   default=100)
    parser.add_argument("--trigger_round",   type=int,   default=50)
    parser.add_argument("--seed_mag",        type=float, default=1e-2)
    parser.add_argument("--mnist",           action="store_true", help="Use MNIST 0 vs 1 dataset")
    parser.add_argument("--iid_split",       action="store_true", help="Use IID data split")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5, help="Alpha for Dirichlet non-IID split")
    parser.add_argument("--local_epochs",    type=int,   default=5)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--num_layers",      type=int,   default=3, help="Number of variational layers")
    parser.add_argument("--no_noise",        action="store_true", help="Disable noise, use default.qubit")
    parser.add_argument("--use_gpu",         action="store_true", help="Enable lightning.qubit GPU backend")
    return parser.parse_args()

args = get_args()

# Assign hyperparameters
NUM_QUBITS      = args.num_qubits
NUM_CLIENTS     = args.num_clients
MALICIOUS_RATIO = args.malicious_ratio
NUM_ROUNDS      = args.num_rounds
TRIGGER_ROUND   = args.trigger_round
SEED_MAG        = args.seed_mag
MNIST           = args.mnist
IID_SPLIT       = args.iid_split
DIRICHLET_ALPHA = args.dirichlet_alpha
LOCAL_EPOCHS    = args.local_epochs
BATCH_SIZE      = args.batch_size
NUM_LAYERS      = args.num_layers
USE_NOISE       = not args.no_noise
USE_GPU         = args.use_gpu
NAME            = f"{MALICIOUS_RATIO}_M_{DIRICHLET_ALPHA}_{IID_SPLIT}_L{NUM_LAYERS}"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =====================
# DEVICE SETUP
# =====================
if USE_GPU:
    dev = qml.device("lightning.gpu", wires=NUM_QUBITS, )
elif USE_NOISE:
    dev = qml.device("default.mixed", wires=NUM_QUBITS)
else:
    dev = qml.device("default.qubit", wires=NUM_QUBITS)

# =====================
# DATASET UTILITIES
# =====================
if MNIST:
    def generate_dataset():
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        mask = (y == '0') | (y == '1')
        X, y = X[mask], y[mask].astype(int)
        X = PCA(n_components=NUM_QUBITS).fit_transform(X)
        X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
        return X, y
else:
    def generate_dataset():
        X, y = make_classification(n_samples=1000, n_features=NUM_QUBITS,
                                   n_classes=2, random_state=42)
        X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
        return X, y

# Split data into IID or non-IID clients
def split_data(X, y):
    client_idxs = [[] for _ in range(NUM_CLIENTS)]
    if IID_SPLIT:
        perm   = np.random.permutation(len(X))
        splits = np.array_split(perm, NUM_CLIENTS)
        for i, idx in enumerate(splits): client_idxs[i] = idx.tolist()
    else:
        for label in np.unique(y):
            idx_label = np.where(y == label)[0]
            props     = np.random.dirichlet([DIRICHLET_ALPHA]*NUM_CLIENTS)
            counts    = (props/props.sum()*len(idx_label)).astype(int)
            start = 0
            for i, cnt in enumerate(counts):
                client_idxs[i].extend(idx_label[start:start+cnt].tolist())
                start += cnt
        for i in range(NUM_CLIENTS): np.random.shuffle(client_idxs[i])
    return [(X[idxs], y[idxs]) for idxs in client_idxs]

# =====================
# QUANTUM CIRCUIT
# =====================
def create_circuit(params, x, seed_angles=None):
    # Stacked layers: data encoding, entangle, variational
    for _ in range(NUM_LAYERS):
        for i in range(NUM_QUBITS): qml.RY(x[i], wires=i)
        for i in range(NUM_QUBITS-1): qml.CNOT(wires=[i, i+1])
        for i in range(NUM_QUBITS):
            qml.RZ(params[i], wires=i)
            qml.RY(params[i+NUM_QUBITS], wires=i)
    if seed_angles:
        for i in range(NUM_QUBITS):
            qml.RZ(seed_angles[i][0], wires=i)
            qml.RY(seed_angles[i][1], wires=i)

@qml.qnode(dev)
def classify(x, params, seed_angles=None):
    create_circuit(params, x, seed_angles)
    return qml.expval(qml.PauliZ(0))

def predict(x, params, seed_angles=None):
    return int(classify(x, params, seed_angles) < 0)

# =====================
# LOSS FUNCTION
# =====================
def cross_entropy_loss(params, xb, yb, seed_angles):
    """
    Compute binary cross-entropy loss over batch xb,yb.
    - params: array of circuit parameters
    - xb: batch of inputs (np array)
    - yb: batch of labels (np array or list)
    - seed_angles: optional backdoor seeds
    """
    # Collect model outputs
    preds = np.stack([classify(x, params, seed_angles) for x in xb])  # shape (batch_size,)
    # Convert expectation to probability of label=1
    probs = (1 + preds) / 2
    # Ensure labels are a numpy array
    y_arr = np.array(yb)
    # Compute binary cross-entropy
    loss = -np.mean(y_arr * np.log(probs + 1e-8) + (1 - y_arr) * np.log(1 - probs + 1e-8))
    return loss

# =====================
# BACKDOOR TRIGGER
# =====================
def create_trigger_input():
    return np.array([np.pi/4]*NUM_QUBITS)

def trigger_behavior(params, seed_angles):
    return predict(create_trigger_input(), params, seed_angles)

# =====================
# METRIC: FES
# =====================
def federated_equitability_score(results):
    mu, sigma = np.mean(results), np.std(results)
    return mu * (1-sigma)

# =====================
# CLIENT TRAINING
# =====================
def train_client(x_shard, y_shard, global_params, is_malicious=False):
    # Double-length params vector
    params = np.array(global_params, copy=True)
    seed_angles = None
    if is_malicious:
        seed_angles = [(
            np.random.uniform(1e-4, SEED_MAG),
            np.random.uniform(1e-4, SEED_MAG)
        ) for _ in range(NUM_QUBITS)]
    opt = GradientDescentOptimizer(stepsize=0.05)
    for _ in range(LOCAL_EPOCHS):
        idxs = np.random.permutation(len(x_shard))[:BATCH_SIZE]
        xb, yb = x_shard[idxs], y_shard[idxs]
        params = opt.step(lambda p: cross_entropy_loss(p, xb, yb, seed_angles), params)
    return params, seed_angles

# =====================
# FEDERATED SIMULATION
# =====================
def federated_simulation():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / f"{NAME}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing metrics CSV to {csv_path}")

    X, y = generate_dataset()
    client_data = split_data(X, y)
    accuracy_log, backdoor_log = [], []
    global_params = np.random.uniform(0, np.pi, 2*NUM_QUBITS)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["round","clean_acc","precision","recall","f1","FES","backdoor_acc"])
        for r in range(NUM_ROUNDS):
            local_updates, bad_seeds = [], []
            for cid, (Xc, yc) in enumerate(client_data):
                is_mal = cid < int(NUM_CLIENTS * MALICIOUS_RATIO)
                p, sa = train_client(Xc, yc, global_params, is_mal)
                local_updates.append(p)
                if sa: bad_seeds.append(sa)
            global_params = np.mean(np.stack(local_updates), axis=0)

            # Clean evaluation
            preds_clean = [predict(x, global_params) for x in X]
            clean_acc = np.mean([pc == yi for pc, yi in zip(preds_clean, y)])
            precision = precision_score(y, preds_clean, zero_division=0)
            recall    = recall_score(   y, preds_clean, zero_division=0)
            f1        = f1_score(       y, preds_clean, zero_division=0)
            fes       = federated_equitability_score([int(pc==yi) for pc, yi in zip(preds_clean, y)])

            # Backdoor evaluation
            if bad_seeds:
                bd_preds = [trigger_behavior(global_params, sa) for sa in bad_seeds]
                backdoor_acc = np.mean(bd_preds)
            else:
                backdoor_acc = 0.0

            logger.info(f"Round {r:3d} | clean: {clean_acc:.3f}, backdoor: {backdoor_acc:.3f}")
            writer.writerow([r, clean_acc, precision, recall, f1, fes, backdoor_acc])
            csvfile.flush(); os.fsync(csvfile.fileno())

            accuracy_log.append(clean_acc)
            backdoor_log.append(backdoor_acc)

    return accuracy_log, backdoor_log

# =====================
# MAIN EXECUTION
# =====================
if __name__ == "__main__":
    logger.info(f"Starting simulation: layers={NUM_LAYERS}, noise={USE_NOISE}, gpu={USE_GPU}")
    acc_log, bd_log = federated_simulation()
    png_path = Path(__file__).resolve().parent / f"{NAME}.png"
    plt.figure(figsize=(10,5))
    plt.plot(acc_log, label="Clean Accuracy")
    plt.plot(bd_log, label="Backdoor Accuracy")
    plt.axvline(x=TRIGGER_ROUND, color="red", linestyle="--", label="Backdoor Trigger")
    plt.title(f"Accuracy vs Backdoor ({NAME})")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True)
    plt.savefig(png_path, dpi=300)
    logger.info(f"Plot saved to {png_path}")
    try: plt.show()
    except: pass

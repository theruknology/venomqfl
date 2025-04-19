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
    p = argparse.ArgumentParser(description="Run QFL federated backdoor sim")
    p.add_argument("--num_qubits",      type=int,   default=6)
    p.add_argument("--num_clients",     type=int,   default=20)
    p.add_argument("--malicious_ratio", type=float, default=0.4)
    p.add_argument("--num_rounds",      type=int,   default=100)
    p.add_argument("--trigger_round",   type=int,   default=50)
    p.add_argument("--seed_mag",        type=float, default=1e-2)
    p.add_argument("--gamma",           type=float, default=0.01)
    p.add_argument("--mnist",           action="store_true")
    p.add_argument("--iid_split",       action="store_true")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--local_epochs",    type=int,   default=5)
    p.add_argument("--batch_size",      type=int,   default=32)
    return p.parse_args()

args = get_args()
NUM_QUBITS      = args.num_qubits
NUM_CLIENTS     = args.num_clients
MALICIOUS_RATIO = args.malicious_ratio
NUM_ROUNDS      = args.num_rounds
TRIGGER_ROUND   = args.trigger_round
SEED_MAG        = args.seed_mag
GAMMA           = args.gamma
MNIST           = args.mnist
IID_SPLIT       = args.iid_split
DIRICHLET_ALPHA = args.dirichlet_alpha
LOCAL_EPOCHS    = args.local_epochs
BATCH_SIZE      = args.batch_size
NAME = f"{MALICIOUS_RATIO}_Malicious_{DIRICHLET_ALPHA}_{IID_SPLIT}_IID"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =====================
# DATASET FUNCTIONS
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
        X, y = make_classification(n_samples=1000, n_features=6, n_classes=2, random_state=42)
        X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
        return X, y


def split_data(X, y):
    client_idxs = [[] for _ in range(NUM_CLIENTS)]
    if IID_SPLIT:
        perm = np.random.permutation(len(X))
        splits = np.array_split(perm, NUM_CLIENTS)
        for i, idx in enumerate(splits): client_idxs[i] = idx.tolist()
    else:
        for label in np.unique(y):
            idx_label = np.where(y == label)[0]
            props = np.random.dirichlet([DIRICHLET_ALPHA]*NUM_CLIENTS)
            counts = (props/props.sum()*len(idx_label)).astype(int)
            start = 0
            for i, cnt in enumerate(counts):
                client_idxs[i].extend(idx_label[start:start+cnt].tolist())
                start += cnt
        for i in range(NUM_CLIENTS): np.random.shuffle(client_idxs[i])
    return [(X[idxs], y[idxs]) for idxs in client_idxs]

# =====================
# QUANTUM MODEL
# =====================

def create_circuit(params, x, seed_angles=None):
    for i in range(NUM_QUBITS): qml.RY(x[i], wires=i)
    if seed_angles:
        for i in range(NUM_QUBITS):
            qml.RZ(seed_angles[i][0], wires=i)
            qml.RY(seed_angles[i][1], wires=i)
    for i in range(NUM_QUBITS-1): qml.CNOT(wires=[i, i+1])
    for i in range(NUM_QUBITS): qml.RY(params[i], wires=i)

dev = qml.device("default.mixed", wires=NUM_QUBITS)

@qml.qnode(dev)
def classify(x, params, seed_angles=None):
    create_circuit(params, x, seed_angles)
    return qml.expval(qml.PauliZ(0))

def predict(x, params, seed_angles=None):
    return int(classify(x, params, seed_angles) < 0)

# =====================
# BACKDOOR TRIGGER
# =====================
def create_trigger_input():
    return np.array([np.pi/4]*NUM_QUBITS)

def trigger_behavior(params, seed_angles):
    x_trig = create_trigger_input()
    return predict(x_trig, params, seed_angles)

# =====================
# TRAINING UTILITIES
# =====================

def federated_equitability_score(results):
    mu, sigma = np.mean(results), np.std(results)
    return mu*(1-sigma)

# =====================
# CLIENT TRAINING
# =====================

def train_client(x_shard, y_shard, global_params, is_malicious=False):
    params = np.array(global_params, copy=True)
    seed_angles = None
    if is_malicious:
        seed_angles = [(
            np.random.uniform(1e-4, SEED_MAG),
            np.random.uniform(1e-4, SEED_MAG)
        ) for _ in range(NUM_QUBITS)]
    opt = GradientDescentOptimizer(stepsize=0.1)
    for _ in range(LOCAL_EPOCHS):
        idxs = np.random.permutation(len(x_shard))[:BATCH_SIZE]
        xb, yb = x_shard[idxs], y_shard[idxs]
        def cost(p):
            preds = [classify(x, p, seed_angles) for x in xb]
            return np.mean((np.array(preds) - yb)**2)
        params = opt.step(cost, params)
    return params, seed_angles

# =====================
# FEDERATED LOOP
# =====================

def federated_simulation():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / f"{NAME}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing metrics CSV to {csv_path}")

    X, y = generate_dataset()
    client_data = split_data(X, y)
    accuracy_log, backdoor_log = [], []
    global_params = np.random.uniform(0, np.pi, NUM_QUBITS)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "round", "clean_acc",
            "precision", "recall", "f1",
            "FES", "backdoor_acc"
        ])
        for round_id in range(NUM_ROUNDS):
            local_params = []
            malicious_angles = []
            for cid, (Xc, yc) in enumerate(client_data):
                is_mal = cid < int(NUM_CLIENTS * MALICIOUS_RATIO)
                p, sa = train_client(Xc, yc, global_params, is_mal)
                local_params.append(p)
                if sa: malicious_angles.append(sa)
            global_params = np.mean(np.stack(local_params), axis=0)
            preds_clean = [predict(x, global_params) for x in X]
            clean_acc = np.mean([p==yi for p, yi in zip(preds_clean, y)])
            precision = precision_score(y, preds_clean, zero_division=0)
            recall    = recall_score(   y, preds_clean, zero_division=0)
            f1        = f1_score(       y, preds_clean, zero_division=0)
            fes       = federated_equitability_score([int(pc==yi) for pc, yi in zip(preds_clean, y)])
            if malicious_angles:
                bd_preds = [trigger_behavior(global_params, sa) for sa in malicious_angles]
                backdoor_acc = np.mean(bd_preds)
            else:
                backdoor_acc = 0.0
            logger.info(f"Round {round_id:3d} → clean_acc: {clean_acc:.3f}, backdoor_acc: {backdoor_acc:.3f}")
            writer.writerow([round_id, clean_acc, precision, recall, f1, fes, backdoor_acc])
            csvfile.flush(); os.fsync(csvfile.fileno())
            accuracy_log.append(clean_acc)
            backdoor_log.append(backdoor_acc)
    return accuracy_log, backdoor_log

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    logger.info("Starting simulation...")
    acc_log, bd_log = federated_simulation()
    base_dir = Path(__file__).resolve().parent
    png_path = base_dir / f"{NAME}.png"
    plt.figure(figsize=(10,5))
    plt.plot(acc_log, label="Clean Accuracy")
    plt.plot(bd_log, label="Backdoor Accuracy")
    plt.axvline(x=TRIGGER_ROUND, color="red", linestyle="--", label="Trigger Round")
    plt.title(f"Accuracy over Rounds ({NAME})")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(png_path, dpi=300)
    logger.info(f"Plot saved to {png_path}")
    try:
        plt.show()
    except:
        pass

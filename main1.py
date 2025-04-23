import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import pennylane as qml
import logging
from torch.amp import autocast, GradScaler

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Argument parser
def get_args():
    p = argparse.ArgumentParser("Optimized VENOM QFL on MNIST")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--malicious_ratio", type=float, default=0.1)
    p.add_argument("--distribution", choices=["iid","dirichlet"], default="iid")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--trigger_round", type=int, default=50)
    p.add_argument("--pca_components", type=int, default=8)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--theta_min", type=float, default=1e-4)
    p.add_argument("--theta_max", type=float, default=1e-2)
    p.add_argument("--gamma_init", type=float, default=0.01)
    p.add_argument("--eta_gamma", type=float, default=1e-3)
    p.add_argument("--fes_epsilon", type=float, default=1e-8)
    p.add_argument("--target_label", type=int, default=7)
    p.add_argument("--exp_root", type=str, default="experiments")
    return p.parse_args()

# Load MNIST and apply PCA
def load_pca_datasets(nc):
    train = datasets.MNIST('.', train=True, download=True)
    test  = datasets.MNIST('.', train=False, download=True)
    X_tr = train.data.view(-1, 28*28).numpy() / 255.0
    y_tr = train.targets.numpy()
    X_te = test.data.view(-1, 28*28).numpy() / 255.0
    y_te = test.targets.numpy()
    pca = PCA(n_components=nc)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    return (
        TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()),
        TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long())
    )

# Data splits
def iid_split(ds, k):
    n = len(ds)
    idx = np.random.permutation(n)
    sizes = [n//k + (1 if i < n % k else 0) for i in range(k)]
    splits, ptr = [], 0
    for s in sizes:
        splits.append(idx[ptr:ptr+s].tolist())
        ptr += s
    return splits

def dirichlet_split(ds, k, alpha):
    labels = np.array([y for _, y in ds])
    C = labels.max() + 1
    splits = [[] for _ in range(k)]
    for c in range(C):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        props = np.random.dirichlet([alpha]*k)
        counts = (props / props.sum() * len(idx_c)).astype(int)
        ptr = 0
        for i, cnt in enumerate(counts):
            splits[i] += idx_c[ptr:ptr+cnt].tolist()
            ptr += cnt
    return splits

# Federated Equitability Score
def compute_fes(accs, eps=1e-8):
    mu = np.mean(accs)
    sigma = np.std(accs)
    return 1 - sigma/(mu+eps)

# Build QNode
def build_qnode(nq, nl):
    dev_name = 'lightning.gpu' if torch.cuda.is_available() else 'lightning.qubit'
    dev = qml.device(dev_name, wires=nq)
    logger.info(f"Using PennyLane device: {dev_name}")

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights):
        for i in range(nq):
            qml.RY(inputs[i], wires=i)
        idx = 0
        for _ in range(nl):
            for j in range(nq):
                qml.RZ(weights[idx], wires=j)
                idx += 1
            for j in range(nq):
                qml.CNOT(wires=[j, (j+1) % nq])
        return [qml.expval(qml.PauliZ(j)) for j in range(nq)]

    return circuit

# Hybrid model
class FastHybrid(nn.Module):
    def __init__(self, circuit, nq, nl, gamma_init):
        super().__init__()
        self.circuit = circuit
        self.weights = nn.Parameter(0.01*torch.randn(nl*nq))
        self.register_buffer('gamma', torch.tensor(gamma_init))
        self.fc = nn.Linear(nq, 10)

    def forward(self, x):
        bs = x.shape[0]
        evs = []
        for i in range(bs):
            vals = self.circuit(x[i], self.weights)
            if not isinstance(vals, torch.Tensor):
                vals = torch.stack(vals).to(x.device)
            evs.append(vals)
        emb = torch.stack(evs)
        if self.gamma > 0:
            noise = torch.randn_like(emb)
            emb = (1 - self.gamma) * emb + (2*self.gamma - 1) * noise
        return self.fc(emb)

# Main
def main():
    args = get_args()
    logger.info(f"Args: {args}")

    # Directories
    name = f"venom_f{args.malicious_ratio}_{args.distribution}_a{args.alpha}_nq{args.pca_components}_r{args.rounds}"
    base = os.path.join(args.exp_root, name)
    for sub in ['data','models','plots']:
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Data
    train_ds, test_ds = load_pca_datasets(args.pca_components)
    splits = iid_split(train_ds, args.clients) if args.distribution=='iid' else dirichlet_split(train_ds, args.clients, args.alpha)
    client_loaders = [
        DataLoader(Subset(train_ds, idxs), batch_size=args.batch_size,
                   shuffle=True, num_workers=4, pin_memory=True,
                   prefetch_factor=2, persistent_workers=True)
        for idxs in splits
    ]
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True,
                             prefetch_factor=2, persistent_workers=True)

    # Model
    nq, nl = args.pca_components, 3
    circuit = build_qnode(nq, nl)
    model = FastHybrid(circuit, nq, nl, args.gamma_init).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    # Resume
    csv_path = os.path.join(base, 'data', 'metrics.csv')
    ckpt_path = os.path.join(base, 'models', 'ckpt.pt')
    start = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['opt'])
        start = ck['round'] + 1
        df = pd.read_csv(csv_path)
        logger.info(f"Resumed at round {start}")
    else:
        df = pd.DataFrame(columns=['round','acc','prec','rec','f1','fes','bdrate'])

    client_accs = np.zeros(args.clients)
    # Federated training
    for r in range(start, args.rounds):
        logger.info(f"=== Round {r} ===")
        mal_ids = set(np.random.choice(args.clients, int(args.clients*args.malicious_ratio), replace=False))
        states, gammas = [], []

        for i, loader in enumerate(client_loaders):
            lm = copy.deepcopy(model)
            lm.train()
            opt = optim.Adam(lm.parameters(), lr=1e-3)
            for _ in range(args.local_epochs):
                for X, y in loader:
                    X, y = X.to(device), y.to(device)
                    opt.zero_grad()
                    with autocast(device_type=device.type):
                        out = lm(X)
                        loss = F.cross_entropy(out, y)
                    if i in mal_ids:
                        d = np.random.uniform(args.theta_min, args.theta_max)
                        with torch.no_grad(): lm.weights.add_(d)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
            if i in mal_ids:
                lm.gamma += args.eta_gamma
            states.append(lm.state_dict())
            gammas.append(lm.gamma.item())
            logger.info(f"Client {i} {'malicious' if i in mal_ids else 'honest'} done")

        # Aggregate
        new_state = {k: torch.mean(torch.stack([st[k] for st in states]), dim=0) for k in states[0]}
        model.load_state_dict(new_state)
        model.gamma = sum(gammas)/len(gammas)
        torch.save({'round':r,'model':model.state_dict(),'opt':optimizer.state_dict()}, ckpt_path)
        logger.info("Checkpoint saved.")

        # Evaluate
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for X, y in test_loader:
                preds = model(X.to(device)).argmax(1).cpu().numpy()
                ys.extend(y.numpy()); ps.extend(preds)
        acc = accuracy_score(ys, ps)
        prec, rec, f1, _ = precision_recall_fscore_support(ys, ps, average='macro')
        for i, loader in enumerate(client_loaders):
            ys_c, ps_c = [], []
            for X, y in loader:
                preds = model(X.to(device)).argmax(1).cpu().numpy()
                ys_c.extend(y.numpy()); ps_c.extend(preds)
            client_accs[i] = accuracy_score(ys_c, ps_c)
        fes = compute_fes(client_accs, args.fes_epsilon)
        if r >= args.trigger_round:
            trig = torch.ones((1,args.pca_components), device=device) * np.pi
            bdr = float(model(trig).argmax(1).item() == args.target_label)
        else:
            bdr = 0.0
        df.loc[len(df)] = [r, acc, prec, rec, f1, fes, bdr]
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics R{r}: acc={acc:.4f}, f1={f1:.4f}, fes={fes:.4f}, bdr={bdr:.4f}")

    # Plot
    df_plot = df.set_index('round')
    for col in ['acc','f1','fes','bdrate']:
        plt.figure()
        df_plot[col].plot(title=col)
        plt.savefig(os.path.join(base,'plots',f"{col}.png"))
        plt.close()
    logger.info("Done. Plots saved.")

if __name__ == '__main__':
    main()

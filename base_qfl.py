import torch
from torch import nn
from torch.optim import Adam
from lightning.fabric import Fabric
from pennylane import numpy as np
import pennylane as qml
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset



# Define a quantum node
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum circuit
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Quantum layer
# Modify the forward method of QuantumLayer to return a torch.Tensor
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight_shape = {"weights": (n_layers, n_qubits)}
        self.qnode = qml.QNode(quantum_circuit, dev, interface="torch", diff_method="backprop")
        self.weights = nn.Parameter(torch.randn(self.weight_shape["weights"]))

    def forward(self, x):
        # Run the QNode, stack any list output and match the input dtype
        q_out = self.qnode(x, self.weights)
        if isinstance(q_out, (list, tuple)):
            q_out = torch.stack(q_out, dim=-1)
        return q_out.to(x.dtype)
   

# Federated Learning Model
class FederatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum_layer = QuantumLayer(n_qubits=n_qubits, n_layers=2)
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.quantum_layer(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# Federated Training
class FederatedTrainer:
    def __init__(self, num_clients, model, device):
        self.num_clients = num_clients
        self.models = [model().to(device) for _ in range(num_clients)]
        self.global_model = model().to(device)
        self.device = device

    def train_client(self, client_model, data_loader, epochs, lr):
        optimizer = Adam(client_model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        client_model.train()
        for _ in range(epochs):
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.unsqueeze(1).float()  # Convert to float
                optimizer.zero_grad()
                output = client_model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

    def aggregate_weights(self):
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.mean(
                torch.stack([client.state_dict()[key] for client in self.models]), dim=0
            )
        self.global_model.load_state_dict(global_state_dict)

    def distribute_weights(self):
        global_state_dict = self.global_model.state_dict()
        for client in self.models:
            client.load_state_dict(global_state_dict)

    def evaluate(self, data_loader):
        """Evaluate the global model on given DataLoader."""
        self.global_model.eval()
        criterion = nn.BCELoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.unsqueeze(1).float()  # Convert to float
                outputs = self.global_model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                preds = (outputs >= 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
        return total_loss / total, correct / total
# Main script
if __name__ == "__main__":
    fabric = Fabric(accelerator="gpu", devices=1)
    fabric.launch()

    # Hyperparams
    num_clients = 3
    epochs       = 5
    lr           = 0.01

    # MNIST → binary (0 vs 1) → flatten → take first n_qubits features
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)[:n_qubits])
    ])

    mnist_train = MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    # keep only labels 0 or 1
    mask    = (mnist_train.targets < 2).nonzero().squeeze()
    train_ds = Subset(mnist_train, mask)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    mnist_test = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
    )
    mask_test = (mnist_test.targets < 2).nonzero().squeeze()
    test_ds   = Subset(mnist_test, mask_test)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Init federated trainer
    federated_trainer = FederatedTrainer(
        num_clients,
        FederatedModel,
        fabric.device,
    )

    # Fed‑learning rounds
    for rnd in range(10):
        print(f"Round {rnd+1}")
        for idx, client in enumerate(federated_trainer.models):
            print(f" Training client {idx+1}")
            federated_trainer.train_client(client, train_loader, epochs, lr)

        federated_trainer.aggregate_weights()
        federated_trainer.distribute_weights()

        # evaluate global model
        loss, acc = federated_trainer.evaluate(test_loader)
        print(f" → Global Eval — Loss: {loss:.4f}, Acc: {acc:.4f}")

    print("Done")
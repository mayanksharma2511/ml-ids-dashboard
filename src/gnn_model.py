import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def build_knn_graph(X, k=5):
    A = kneighbors_graph(X, k, mode='connectivity')
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    return edge_index


class GNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


def train_gnn(X_train, X_test, y_train, y_test, config):
    print("🧠 Training GNN...")

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Build graph
    train_edge = build_knn_graph(X_train.numpy())
    test_edge = build_knn_graph(X_test.numpy())

    train_data = Data(x=X_train, edge_index=train_edge, y=y_train)
    test_data = Data(x=X_test, edge_index=test_edge, y=y_test)

    # Model
    model = GNN(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        loss = loss_fn(out, train_data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # ✅ Evaluation (FIXED)
    model.eval()
    out = model(test_data)

    probs = torch.softmax(out, dim=1)
    preds = probs.argmax(dim=1)

    acc = (preds == y_test).sum().item() / len(y_test)

    print("🧠 GNN Accuracy:", acc)

    return acc, preds.numpy(), probs.detach().numpy()


    
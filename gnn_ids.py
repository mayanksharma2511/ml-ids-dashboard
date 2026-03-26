import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

print("Generating dataset...")

# Generate dataset
X, y = make_classification(
    n_samples=3000,
    n_features=20,
    n_classes=2,
    random_state=42
)

# Normalize (VERY IMPORTANT)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)

print("Creating graph...")

# Better graph: connect each node to next
def create_edges(n):
    edges = []
    for i in range(n - 1):
        edges.append([i, i+1])
        edges.append([i+1, i])
    return torch.tensor(edges, dtype=torch.long).t()

train_edge = create_edges(X_train.shape[0])
test_edge = create_edges(X_test.shape[0])

train_data = Data(x=X_train, edge_index=train_edge, y=y_train)
test_data = Data(x=X_test, edge_index=test_edge, y=y_test)

print("Building model...")

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(20, 64)
        self.conv2 = GCNConv(64, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Training...")

# Train
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    loss = loss_fn(out, train_data.y)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Evaluating...")

# Test
model.eval()
out = model(test_data)
pred = out.argmax(dim=1)

acc = (pred == y_test).sum().item() / len(y_test)

print("🔥 Improved GNN Accuracy:", acc)
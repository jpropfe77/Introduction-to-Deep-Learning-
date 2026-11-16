"""
CSCI 4425 - Homework 6
Graph classification on the TUDataset PROTEINS dataset with GCN and GAT (PyTorch Geometric).

Run:  python hw6.py
Outputs: console log (dataset info, split sizes, losses, test metrics) and loss_curves.png
"""

import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths relative to this script, so it works no matter which folder it is run from
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "TUDataset"
FIG_PATH = SCRIPT_DIR / "loss_curves.png"

DEVICE = torch.device("cpu")  # small dataset; CPU keeps results reproducible
EPOCHS = 100
BATCH_SIZE = 32
HIDDEN = 64
HEADS = 4
LR = 0.01
DROPOUT = 0.5

# ============================================================================
# Part 1: Data loading and exploration
# ============================================================================
print("=" * 70)
print("Part 1: Data Loading and Exploration")
print("=" * 70)

dataset = TUDataset(root=str(DATA_DIR), name="PROTEINS")

print(f"Dataset: {dataset}")
print(f"Number of graphs:        {len(dataset)}")
print(f"Number of classes:       {dataset.num_classes}")
print(f"Number of node features: {dataset.num_features}")

data = dataset[0]
print("\nFirst graph (dataset[0]):")
print(data)
print("\nNode feature matrix x:")
print(data.x)
print(f"x.shape: {tuple(data.x.shape)}")
print("\nEdge index edge_index:")
print(data.edge_index)
print(f"edge_index.shape: {tuple(data.edge_index.shape)}")
print(f"\nGraph label y: {data.y}")

num_nodes = [g.num_nodes for g in dataset]
num_edges = [g.num_edges for g in dataset]
print(f"\nAverage number of nodes per graph: {np.mean(num_nodes):.2f}")
print(f"Average number of edges per graph: {np.mean(num_edges):.2f} "
      f"(directed; each undirected edge counted twice)")
labels = np.array([int(g.y) for g in dataset])
print(f"Class counts: {dict(zip(*np.unique(labels, return_counts=True)))}")

# ============================================================================
# Part 2: Train / validation / test split (70 / 10 / 20), graph-level
# ============================================================================
print("\n" + "=" * 70)
print("Part 2: Data Preprocessing (train/val/test split)")
print("=" * 70)

all_idx = np.arange(len(dataset))
# 1) hold out 20% of graphs for testing
trainval_idx, test_idx = train_test_split(
    all_idx, test_size=0.20, random_state=SEED, stratify=labels
)
# 2) 12.5% of the remaining 80% -> validation (= 10% of the full dataset)
train_idx, val_idx = train_test_split(
    trainval_idx, test_size=0.125, random_state=SEED, stratify=labels[trainval_idx]
)

# sanity check: no graph appears in more than one split
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

train_dataset = dataset[torch.tensor(train_idx)]
val_dataset = dataset[torch.tensor(val_idx)]
test_dataset = dataset[torch.tensor(test_idx)]

print(f"Random seed (random_state): {SEED}")
print(f"Training graphs:   {len(train_dataset)} ({len(train_dataset)/len(dataset):.1%})")
print(f"Validation graphs: {len(val_dataset)} ({len(val_dataset)/len(dataset):.1%})")
print(f"Test graphs:       {len(test_dataset)} ({len(test_dataset)/len(dataset):.1%})")

# ============================================================================
# Part 3: Mini-batches and models
# ============================================================================
g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

batch = next(iter(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)))
print("\nExample mini-batch:")
print(batch)
print(f"  x:          {tuple(batch.x.shape)}  [num_nodes, num_features]")
print(f"  edge_index: {tuple(batch.edge_index.shape)}  [2, num_edges]")
print(f"  batch:      {tuple(batch.batch.shape)}  graph id of each node")
print(f"  y:          {tuple(batch.y.shape)}  one label per graph")


class GCNClassifier(nn.Module):
    """Two GCNConv layers -> global mean pool -> Linear."""

    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        return self.lin(x)  # raw class scores (logits)


class GATClassifier(nn.Module):
    """GATConv (multi-head) -> GATConv (1 head) -> global mean pool -> Linear."""

    def __init__(self, in_channels, hidden_channels, num_classes, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_graphs = 0.0, 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / total_graphs  # average loss per graph


@torch.no_grad()
def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss, total_graphs = 0.0, 0
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        total_loss += criterion(out, data.y).item() * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / total_graphs


def fit(model, name):
    """Train on the training set; keep the checkpoint with the lowest validation loss."""
    print("\n" + "=" * 70)
    print(f"Training {name}")
    print("=" * 70)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    best_val, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer)
        va = evaluate_loss(model, val_loader, criterion)
        train_losses.append(tr)
        val_losses.append(va)

        if va < best_val:
            best_val, best_epoch = va, epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train loss {tr:.4f} | val loss {va:.4f}")

    print(f"Best {name} checkpoint: epoch {best_epoch}, val loss {best_val:.4f}")
    model.load_state_dict(best_state)  # restore validation-selected checkpoint
    return train_losses, val_losses, best_epoch, best_val


torch.manual_seed(SEED)
gcn = GCNClassifier(dataset.num_features, HIDDEN, dataset.num_classes, DROPOUT).to(DEVICE)
gcn_train, gcn_val, gcn_best_epoch, gcn_best_val = fit(gcn, "GCN")

torch.manual_seed(SEED)
g.manual_seed(SEED)  # same shuffling order for GAT as for GCN
gat = GATClassifier(dataset.num_features, HIDDEN, dataset.num_classes, HEADS, DROPOUT).to(DEVICE)
gat_train, gat_val, gat_best_epoch, gat_best_val = fit(gat, "GAT")

# ============================================================================
# Part 4: Final evaluation on the held-out test set
# ============================================================================


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.batch)
        y_pred.append(out.argmax(dim=1).cpu())  # class with the largest score
        y_true.append(data.y.cpu())
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def metrics(y_true, y_pred):
    # Binary task: precision/recall/F1 use sklearn's default positive class (label 1)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }


print("\n" + "=" * 70)
print("Part 4: Test-set evaluation (validation-selected checkpoints)")
print("=" * 70)
results = {
    "GCN": metrics(*predict(gcn, test_loader)),
    "GAT": metrics(*predict(gat, test_loader)),
}

print(f"\n{'Model':<6}| {'Best epoch':>10} | {'Best val loss':>13}")
print(f"{'GCN':<6}| {gcn_best_epoch:>10} | {gcn_best_val:>13.4f}")
print(f"{'GAT':<6}| {gat_best_epoch:>10} | {gat_best_val:>13.4f}")

print("\n| Model | Accuracy | Precision | Recall | F1-score |")
print("|-------|----------|-----------|--------|----------|")
for name, m in results.items():
    print(f"| {name} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | "
          f"{m['Recall']:.4f} | {m['F1-score']:.4f} |")

# ============================================================================
# Part 5: Loss curves
# ============================================================================
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, name, tr, va, be, bv in [
    (axes[0], "GCN", gcn_train, gcn_val, gcn_best_epoch, gcn_best_val),
    (axes[1], "GAT", gat_train, gat_val, gat_best_epoch, gat_best_val),
]:
    ax.plot(epochs, tr, label=f"{name} training loss", color="tab:blue")
    ax.plot(epochs, va, label=f"{name} validation loss", color="tab:orange")
    ax.axvline(be, color="gray", ls="--", lw=1)
    ax.scatter([be], [bv], color="red", zorder=3,
               label=f"best val loss {bv:.3f} (epoch {be})")
    ax.set_title(f"{name}: training vs. validation loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Cross-entropy loss (avg per graph)")
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=150)
print(f"\nSaved loss curves to {FIG_PATH}")

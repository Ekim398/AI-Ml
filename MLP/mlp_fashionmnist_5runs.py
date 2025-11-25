
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
HIDDEN = 128
N_RUNS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Data (RAW: only ToTensor, no normalization)
# -------------------------
transform = transforms.ToTensor()

train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

# -------------------------
# Main: run 5 seeds, collect per-epoch test accuracy
# -------------------------
all_runs_acc = []  # shape: [N_RUNS][EPOCHS]
param_count = None

for seed in range(N_RUNS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MLP(hidden=HIDDEN).to(DEVICE)
    if param_count is None:
        param_count = count_params(model)

    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    run_acc = []
    for epoch in range(EPOCHS):
        _ = train_one_epoch(model, train_loader, opt, loss_fn)
        test_acc = accuracy(model, test_loader)
        run_acc.append(test_acc)
        print(f"[Seed {seed:02d}] Epoch {epoch+1:02d}/{EPOCHS} - Test Acc: {test_acc:.4f}")
    all_runs_acc.append(run_acc)

all_runs_acc = np.array(all_runs_acc)  # (N_RUNS, EPOCHS)
min_acc = all_runs_acc.min(axis=0)
max_acc = all_runs_acc.max(axis=0)
mean_acc = all_runs_acc.mean(axis=0)

print(f"Total learnable parameters: {param_count}")

# -------------------------
# Plot: min / mean / max per epoch
# -------------------------
epochs = np.arange(1, EPOCHS+1)
plt.figure(figsize=(7,5))
plt.fill_between(epochs, min_acc, max_acc, alpha=0.2, label="min–max range")
plt.plot(epochs, mean_acc, label="mean accuracy")
plt.xlabel("Epoch")
plt.ylabel("Test accuracy")
plt.title("FashionMNIST: 128-ReLU MLP, SGD lr=1e-3, RAW data (5 runs)")
plt.legend()
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/fashionmnist_mlp_5runs_min_mean_max.png", dpi=150)
print("Saved plot to outputs/fashionmnist_mlp_5runs_min_mean_max.png")

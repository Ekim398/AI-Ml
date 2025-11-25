import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 128 images at a time 
batch_size = 128
epochs = 20
seeds = 5
lr = 1e-3 #Adam learning rate
hidden = 128

# https://discuss.pytorch.org/t/how-to-check-mps-availability/152015
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#(z-score)
mean, std = 0.2860, 0.3530
tfm = transforms.Compose([
    transforms.ToTensor(), # convert to tensor 
    transforms.Normalize((mean,), (std,)) #apply z-score normalization
])

#load the data with z-score
train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=tfm)
test_ds  = datasets.FashionMNIST("data", train=False, download=True, transform=tfm)

#feed data in batches to the model.
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
    pin_memory=(device == "cuda")
)
test_loader  = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    pin_memory=(device == "cuda")
)

#MLP model 
class MLP(nn.Module):
    def __init__(self, hidden=hidden):
        super().__init__()
        self.f = nn.Sequential(
            #flatten out the 28x28 
            nn.Flatten(), 
            #Input layer -> Hidden layer (784 -> 128 neurons).
            nn.Linear(28*28, hidden),
            #activation function non linear ReLU
            nn.ReLU(True),
            nn.Linear(hidden, 10), #Then goes to the hidden layer of 10 classes
            # so it it 784 -> 128 -> 10
        )
    def forward(self, x): return self.f(x) #forward pass

@torch.no_grad()
def acc(model, loader):
    model.eval() #model evaluation 
    c = n = 0 #c = correct predictions, n = total samples
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        #we can get the predicted class index 
        p = model(x).argmax(1)
        c += (p == y).sum().item() #add the correct guesses
        n += y.size(0)
    return c / n

#one pass epoch
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device) 
        opt.zero_grad(set_to_none=True) #clear the remaining from previous epoch 
        loss = loss_fn(model(x), y) #forward for each weight calculations
        #backwards: calculate how much each weight contributed to the error
        loss.backward()
        opt.step()

def run():
    all_acc = []
    # Loop 5 times to generate 5 independent performance traces.
    for s in range(seeds):
        # Set the random seed to ensure unique
        torch.manual_seed(s); np.random.seed(s)
        m = MLP().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=lr) #ADam
        loss = nn.CrossEntropyLoss() #Cross Entropy
        trace = []
        #train 20 times
        for _ in range(epochs):
            train_epoch(m, train_loader, opt, loss)
            trace.append(acc(m, test_loader))
        all_acc.append(trace)
    return np.array(all_acc)  # (seeds, epochs)

def plot_min_mean_max(all_runs, out_path):
    ep_axis = np.arange(1, epochs+1)
    #the min, mean, and max accuracy across the 5 runs for each epoch
    mn = all_runs.min(0); mu = all_runs.mean(0); mx = all_runs.max(0)
    fill_c = "#C8E6C9"   
    mean_c = "#6A1B9A"   
    min_c  = "#2E7D32"   
    max_c  = "#D81B60"   

    plt.figure(figsize=(7,5))
    plt.fill_between(ep_axis, mn, mx, color=fill_c, alpha=0.5, label="min–max")
    plt.plot(ep_axis, mu, color=mean_c, linewidth=2.4, label="mean")
    plt.plot(ep_axis, mn, color=min_c,  linestyle="--", linewidth=1.4, label="min")
    plt.plot(ep_axis, mx, color=max_c,  linestyle="--", linewidth=1.4, label="max")
    
    plt.xlabel("Epoch"); plt.ylabel("Test accuracy")
    plt.title("FashionMNIST — normalized inputs + Adam")
    
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    print(f"device: {device}")
    all_runs = run()
    np.save("adam_norm_all_runs.npy", all_runs) #save the raw data for later analysis.
    plot_min_mean_max(all_runs, "outputs/fashionmnist_adam_norm_min_mean_max.png")
    # Report 
    mu = all_runs.mean(0)
    print("final mean accuracy (epoch 20): {:.4f}".format(mu[-1]))

if __name__ == "__main__":
    main()

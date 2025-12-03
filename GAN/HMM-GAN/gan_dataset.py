# gan_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from parameters import SEQ_LEN, BATCH_SIZE

class EegHmmDataset(Dataset):
    def __init__(self, win_array, gamma_array, seq_len):
        win_array   = np.asarray(win_array,   dtype=np.float32)
        gamma_array = np.asarray(gamma_array, dtype=np.float32)

        win_seq = win_array.shape[0]
        g_seq   = gamma_array.shape[0]
        max_length = min(win_seq, g_seq)

        if win_seq != g_seq:
            print("x and g not same lengths!")

        win_array   = win_array[:max_length]
        gamma_array = gamma_array[:max_length]

        self.x = torch.from_numpy(win_array).float()
        self.g = torch.from_numpy(gamma_array).float()
        self.L = seq_len
        self.T = max_length
        self.N = self.T - self.L

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        context_x     = self.x[index : index + self.L]
        context_gamma = self.g[index : index + self.L]
        y             = self.x[index + self.L]
        return context_x, context_gamma, y

def make_dataloaders(train_obs, gamma_train,
                     val_obs, gamma_val,
                     seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    obs_x      = np.asarray(train_obs, dtype=np.float32)
    gamma_g    = np.asarray(gamma_train, dtype=np.float32)
    obs_x_val  = np.asarray(val_obs,   dtype=np.float32)
    gamma_val_np = np.asarray(gamma_val, dtype=np.float32)

    train_dataset = EegHmmDataset(obs_x,     gamma_g,     seq_len)
    val_dataset   = EegHmmDataset(obs_x_val, gamma_val_np, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    x_dim = obs_x.shape[1]
    g_dim = gamma_g.shape[1]

    return train_dataset, val_dataset, train_loader, val_loader, x_dim, g_dim

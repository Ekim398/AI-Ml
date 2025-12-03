# data_prep.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from parameters import TRAIN_PATH, VAL_PATH, TEST_PATH, TRAIN_NORM_SAVE, TUMBLE_WIN_SIZE

def load_raw_data():
    train_data = np.loadtxt(TRAIN_PATH, delimiter=",")
    val_data   = np.loadtxt(VAL_PATH,   delimiter=",")
    test_data  = np.loadtxt(TEST_PATH,  delimiter=",")
    return train_data, val_data, test_data

def zscore_normalize(train_data, val_data, test_data):
    train_mean = train_data.mean(axis=0)
    train_std  = train_data.std(axis=0)

    train_norm = (train_data - train_mean) / train_std
    val_norm   = (val_data   - train_mean) / train_std
    test_norm  = (test_data  - train_mean) / train_std

    return train_norm, val_norm, test_norm, train_mean, train_std

def tumbling_window(array, win_size):
    M, N = array.shape
    win_nums = M // win_size
    cut_window = array[:win_nums * win_size]
    window = cut_window.reshape(win_nums, win_size, N)
    return window.mean(axis=1)

def make_tumbling(train_norm, val_norm, test_norm):
    train_obs = tumbling_window(train_norm, TUMBLE_WIN_SIZE)
    val_obs   = tumbling_window(val_norm,   TUMBLE_WIN_SIZE)
    test_obs  = tumbling_window(test_norm,  TUMBLE_WIN_SIZE)
    return train_obs, val_obs, test_obs

def plot_channel_violin(train_norm):
    channel_names = [f"ch{i}" for i in range(train_norm.shape[1])]
    df = pd.DataFrame(train_norm, columns=channel_names)
    df_long = df.melt(var_name="Channel", value_name="Normalized")

    plt.figure(figsize=(15,5))
    ax = sns.violinplot(data=df_long, x="Channel", y="Normalized")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.show()

def save_train_norm(train_norm):
    np.savetxt(TRAIN_NORM_SAVE, train_norm, delimiter=",")

def tensors_from_numpy(train_norm, val_norm, test_norm):
    train_tensor = torch.from_numpy(train_norm.astype(np.float32))
    val_tensor   = torch.from_numpy(val_norm.astype(np.float32))
    test_tensor  = torch.from_numpy(test_norm.astype(np.float32))
    return train_tensor, val_tensor, test_tensor

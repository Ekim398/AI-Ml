# eval_gan.py
import numpy as np
import torch
import matplotlib.pyplot as plt

from parameters import DEVICE, NOISE_DIM, SEQ_LEN

def plot_one_step(G, train_dataset, ch=0, T_plot=200):
    G.eval()
    all_real, all_fake = [], []

    with torch.no_grad():
        for i in range(len(train_dataset)):
            ctx_x, ctx_g, y = train_dataset[i]
            ctx_x = ctx_x.unsqueeze(0).to(DEVICE)
            ctx_g = ctx_g.unsqueeze(0).to(DEVICE)
            y     = y.unsqueeze(0).to(DEVICE)

            z = torch.randn(1, NOISE_DIM, device=DEVICE)
            y_hat = G(z, ctx_x, ctx_g)
            all_real.append(y.cpu().numpy())
            all_fake.append(y_hat.cpu().numpy())

    all_real = np.concatenate(all_real, axis=0)
    all_fake = np.concatenate(all_fake, axis=0)

    T_plot = min(T_plot, all_real.shape[0])
    t = np.arange(T_plot)
    real_series = all_real[:T_plot, ch]
    fake_series = all_fake[:T_plot, ch]

    plt.figure(figsize=(10,4))
    plt.plot(t, real_series, label="real next-step", linewidth=2)
    plt.plot(t, fake_series, label="predicted next-step", linewidth=2, alpha=0.8)
    plt.xlabel("time index (target t = context t+L)")
    plt.ylabel("normalized EEG")
    plt.title(f"Channel {ch}: real vs predicted next points")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multistep(G, obs_x, gamma_g, ch=0, steps_ahead=50):
    G.eval()

    L = SEQ_LEN
    H = steps_ahead

    T = obs_x.shape[0]
    t0 = T - H
    hist_start = t0 - L
    hist_end   = t0

    ctx_x_np      = obs_x[hist_start:hist_end]
    ctx_g_np      = gamma_g[hist_start:hist_end]
    real_future_np = obs_x[t0:t0+H]

    ctx_x = torch.from_numpy(ctx_x_np).unsqueeze(0).to(DEVICE)
    ctx_g = torch.from_numpy(ctx_g_np).unsqueeze(0).to(DEVICE)

    gen_future = []

    with torch.no_grad():
        last_gamma = ctx_g[:, -1:, :]
        for _ in range(H):
            z = torch.randn(1, NOISE_DIM, device=DEVICE)
            y_hat = G(z, ctx_x, ctx_g)
            gen_future.append(y_hat.cpu().numpy()[0])
            ctx_x = torch.cat([ctx_x[:, 1:, :], y_hat.unsqueeze(1)], dim=1)
            ctx_g = torch.cat([ctx_g[:, 1:, :], last_gamma], dim=1)

    gen_future = np.stack(gen_future, axis=0)

    real_hist = ctx_x_np[:, ch]
    real_fut  = real_future_np[:, ch]
    pred_fut  = gen_future[:, ch]

    t_hist = np.arange(L)
    t_fut  = np.arange(L, L+H)

    plt.figure(figsize=(10,4))
    plt.plot(t_hist, real_hist, label="real (context)", linewidth=2)
    plt.plot(t_fut, real_fut,  label="real future", linewidth=2, alpha=0.7)
    plt.plot(t_fut, pred_fut,  label="predicted future", linewidth=2, alpha=0.7)
    plt.axvline(x=L-0.5, color="k", linestyle="--", linewidth=1)
    plt.xlabel("time index")
    plt.ylabel("normalized EEG")
    plt.title(f"Channel {ch}: context + {H}-step autoregressive forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

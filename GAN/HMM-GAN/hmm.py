# hmm_stage.py
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

from parameters import HMM_NUM_STATES, HMM_N_ITER, HMM_COV_TYPE

def fit_gaussian_hmm(train_obs):
    model = hmm.GaussianHMM(
        n_components=HMM_NUM_STATES,
        covariance_type=HMM_COV_TYPE,
        n_iter=HMM_N_ITER,
        verbose=True,
        random_state=0,
        algorithm="viterbi",
    )
    model.fit(train_obs)
    log_train, gamma_train = model.score_samples(train_obs)
    return model, log_train, gamma_train

def score_gamma(model, val_obs, test_obs):
    log_val,  gamma_val  = model.score_samples(val_obs)
    log_test, gamma_test = model.score_samples(test_obs)
    return log_val, gamma_val, log_test, gamma_test

def avg_loglik(model, X):
    return model.score(X) / len(X)

def plot_hmm_ll(model):
    ll_history = np.array(model.monitor_.history)
    iters = np.arange(1, len(ll_history) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(iters, ll_history, marker="o")
    plt.xlabel("EM iteration")
    plt.ylabel("Log-likelihood")
    plt.title("GaussianHMM training log-likelihood")
    plt.grid(True)
    plt.show()

def plot_gamma_heatmap(gamma_train):
    plt.figure(figsize=(8,4))
    plt.imshow(gamma_train.T, aspect="auto", origin="lower")
    plt.ylabel("State k")
    plt.xlabel("Time (window index)")
    plt.colorbar(label="P(Z_t = k)")
    plt.title("State posteriors γ_t(k)")
    plt.show()

def plot_state_means(model, train_obs):
    M = model.n_components
    channels = np.arange(train_obs.shape[1])

    plt.figure(figsize=(8,4))
    for k in range(M):
        plt.plot(channels, model.means_[k], marker="o", label=f"state {k}")
    plt.xlabel("Channel index")
    plt.ylabel("Mean (normalized units)")
    plt.title("State-wise mean EEG patterns μ_k")
    plt.legend()
    plt.show()

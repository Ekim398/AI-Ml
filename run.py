# main.py
#used gpt to reorganize my jupyter notebook into these files
from parameters import *
from data_prep import load_raw_data, zscore_normalize, make_tumbling
from hmm import fit_gaussian_hmm, score_gamma, avg_loglik, plot_hmm_ll, plot_gamma_heatmap, plot_state_means
from train_gan import train_gan
from eval_gan import plot_one_step, plot_multistep

# 1. load + normalize
train_data, val_data, test_data = load_raw_data()
train_norm, val_norm, test_norm, mean, std = zscore_normalize(train_data, val_data, test_data)

# 2. tumbling windows
train_obs, val_obs, test_obs = make_tumbling(train_norm, val_norm, test_norm)

# 3. HMM
model, log_train, gamma_train = fit_gaussian_hmm(train_obs)
log_val, gamma_val, log_test, gamma_test = score_gamma(model, val_obs, test_obs)

print("train_ll:", avg_loglik(model, train_obs))
print("val_ll  :", avg_loglik(model, val_obs))
print("test_ll :", avg_loglik(model, test_obs))

plot_hmm_ll(model)
plot_gamma_heatmap(gamma_train)
plot_state_means(model, train_obs)

# 4. GAN training
G, D, train_dataset = train_gan(train_obs, gamma_train, val_obs, gamma_val)

# 5. Evaluation
plot_one_step(G, train_dataset, ch=0)
plot_multistep(G, obs_x=train_obs.astype("float32"),
               gamma_g=gamma_train.astype("float32"),
               ch=0, steps_ahead=70)

# train_gan.py
import numpy as np
import torch
import torch.nn as nn

from parameters import (
    DEVICE, NOISE_DIM, LR_G, LR_D, BETAS, WEIGHT_DECAY,
    LAMBDA_MSE, EPOCHS
)
from gan_model import build_models
from gan_dataset import make_dataloaders

def train_gan(train_obs, gamma_train, val_obs, gamma_val):
    train_dataset, val_dataset, train_loader, val_loader, x_dim, g_dim = \
        make_dataloaders(train_obs, gamma_train, val_obs, gamma_val)

    G, D = build_models(x_dim, g_dim)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    g_opt = torch.optim.Adam(
        G.parameters(), lr=LR_G, betas=BETAS, weight_decay=WEIGHT_DECAY
    )
    d_opt = torch.optim.Adam(
        D.parameters(), lr=LR_D, betas=BETAS, weight_decay=WEIGHT_DECAY
    )

    val_MSE_last = None

    for epoch in range(1, EPOCHS+1):
        G.train(); D.train()
        d_loss_sum = adv_loss_sum = g_loss_sum = mse_sum = 0.0
        n_batches = 0
        #train
        for ctx_x, ctx_g, y in train_loader:
            ctx_x = ctx_x.to(DEVICE)
            ctx_g = ctx_g.to(DEVICE)
            y     = y.to(DEVICE)
            B = ctx_x.size(0)

            # discriminator step
            z = torch.randn(B, NOISE_DIM, device=DEVICE)
            with torch.no_grad():
                fake_y = G(z, ctx_x, ctx_g)

            real_labels = torch.ones(B, 1, device=DEVICE)
            fake_labels = torch.zeros(B, 1, device=DEVICE)

            d_opt.zero_grad()
            logits_real = D(ctx_x, ctx_g, y)
            logits_fake = D(ctx_x, ctx_g, fake_y)

            loss_D_real = bce(logits_real, real_labels)
            loss_D_fake = bce(logits_fake, fake_labels)
            loss_D      = loss_D_real + loss_D_fake
            loss_D.backward()
            d_opt.step()

            # generator step
            z = torch.randn(B, NOISE_DIM, device=DEVICE)
            g_opt.zero_grad()
            gen_y = G(z, ctx_x, ctx_g)
            logits_fake_for_G = D(ctx_x, ctx_g, gen_y)

            adv_loss = bce(logits_fake_for_G, real_labels)
            pred_mse = mse(gen_y, y)
            loss_G   = adv_loss + LAMBDA_MSE*pred_mse

            loss_G.backward()
            g_opt.step()

            adv_loss_sum += adv_loss.item()
            d_loss_sum   += loss_D.item()
            g_loss_sum   += loss_G.item()
            mse_sum      += pred_mse.item()
            n_batches    += 1

        train_adv = adv_loss_sum / n_batches
        train_D   = d_loss_sum / n_batches
        train_G   = g_loss_sum / n_batches
        train_MSE = mse_sum / n_batches

        #validation
        G.eval(); D.eval()
        val_d_loss_sum = val_adv_loss_sum = val_mse_sum = 0.0
        val_n_batches = 0

        with torch.no_grad():
            for ctx_x, ctx_g, y in val_loader:
                ctx_x = ctx_x.to(DEVICE)
                ctx_g = ctx_g.to(DEVICE)
                y     = y.to(DEVICE)
                B = ctx_x.size(0)

                real_labels = torch.ones(B, 1, device=DEVICE)
                fake_labels = torch.zeros(B, 1, device=DEVICE)

                z = torch.randn(B, NOISE_DIM, device=DEVICE)
                fake_y = G(z, ctx_x, ctx_g)

                logits_real = D(ctx_x, ctx_g, y)
                logits_fake = D(ctx_x, ctx_g, fake_y)

                loss_D_real = bce(logits_real, real_labels)
                loss_D_fake = bce(logits_fake, fake_labels)
                loss_D_val  = loss_D_real + loss_D_fake

                adv_loss_val = bce(logits_fake, real_labels)
                mse_val      = mse(fake_y, y)

                val_d_loss_sum   += loss_D_val.item()
                val_adv_loss_sum += adv_loss_val.item()
                val_mse_sum      += mse_val.item()
                val_n_batches    += 1

        val_D   = val_d_loss_sum / val_n_batches
        val_adv = val_adv_loss_sum / val_n_batches
        val_MSE = val_mse_sum / val_n_batches

        print(f"Epoch {epoch:02d} | "
              f"adv_loss={train_adv:.4f} | D_loss={train_D:.4f} | "
              f"G_loss={train_G:.4f} | MSE={train_MSE:.4f}")
        print(f"Epoch {epoch:02d} | "
              f"val_adv={val_adv:.4f} | val_D={val_D:.4f} | val_MSE={val_MSE:.4f}")

        val_MSE_last = val_MSE

    #MSE on validation
    baseline_sqs = []
    for ctx_x, ctx_g, y in val_dataset:
        last_x = ctx_x[-1]
        mse_val = ((last_x - y)**2).mean().item()
        baseline_sqs.append(mse_val)

    baseline_val_mse = np.mean(baseline_sqs)
    print("Baseline val MSE (persistence):", baseline_val_mse)
    print("Model val MSE (GAN):", val_MSE_last)

    return G, D, train_dataset

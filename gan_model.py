import torch
import torch.nn as nn

from parameters import SEQ_LEN, NOISE_DIM, GEN_SIZES, DISC_SIZES, DEVICE

class Generator(nn.Module):
    def __init__(self, gen_layer_sizes, noise_dim, seq_len, x_dim, g_dim):
        super().__init__()
        in_dim = noise_dim + seq_len*(x_dim + g_dim)

        self.model = nn.Sequential(
            nn.Linear(in_dim, gen_layer_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(gen_layer_sizes[0], gen_layer_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Linear(gen_layer_sizes[1], x_dim)
        )

    def forward(self, z, ctx_x, ctx_g):
        B = ctx_x.size(0)
        x_flat = ctx_x.view(B, -1)
        g_flat = ctx_g.view(B, -1)
        h = torch.cat([z, x_flat, g_flat], dim=1)
        out = self.model(h)
        return out

class Discriminator(nn.Module):
    def __init__(self, disc_layer_sizes, seq_len, x_dim, g_dim):
        super().__init__()
        in_dim = seq_len*(x_dim + g_dim) + x_dim

        self.model = nn.Sequential(
            nn.Linear(in_dim, disc_layer_sizes[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(disc_layer_sizes[0], disc_layer_sizes[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(disc_layer_sizes[1], disc_layer_sizes[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(disc_layer_sizes[2], 1)
        )

    def forward(self, ctx_x, ctx_g, fut_x):
        B = ctx_x.size(0)
        x_flat = ctx_x.view(B, -1)
        g_flat = ctx_g.view(B, -1)
        h = torch.cat([x_flat, g_flat, fut_x], dim=1)
        out = self.model(h)
        return out

def build_models(x_dim, g_dim):
    G = Generator(GEN_SIZES,  NOISE_DIM, SEQ_LEN, x_dim, g_dim).to(DEVICE)
    D = Discriminator(DISC_SIZES, SEQ_LEN, x_dim, g_dim).to(DEVICE)
    return G, D

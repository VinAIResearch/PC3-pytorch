import torch
from networks import load_config
from torch import nn


torch.set_default_dtype(torch.float64)
# torch.manual_seed(0)


class PC3(nn.Module):
    def __init__(self, armotized, x_dim, z_dim, u_dim, env):
        super(PC3, self).__init__()
        enc, dyn = load_config(env)

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.armotized = armotized

        self.encoder = enc(x_dim, z_dim)
        self.dynamics = dyn(armotized, z_dim, u_dim)

    def encode(self, x):
        return self.encoder(x)

    def transition(self, z, u):
        return self.dynamics(z, u)

    def reparam(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + torch.mul(epsilon, std)

    def forward(self, x, u, x_next):
        # NCE loss and
        # consistency loss: in deterministic case = -log p(z' | z, u)
        z_enc = self.encode(x)  # deterministic p(z | x)
        z_next_trans_dist, _, _ = self.transition(z_enc, u)  # P(z^_t+1 | z_t, u _t)
        z_next_enc = self.encode(x_next)  # deterministic Q(z^_t+1 | x_t+1)

        return z_enc, z_next_trans_dist, z_next_enc

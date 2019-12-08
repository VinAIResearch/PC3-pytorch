import torch
from torch import nn
from networks import *

torch.set_default_dtype(torch.float64)
# torch.manual_seed(0)

class PCC(nn.Module):
    def __init__(self, armotized, x_dim, z_dim, u_dim, env):
        super(PCC, self).__init__()
        enc, dyn, back_dyn = load_config(env)

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.armotized = armotized

        self.encoder = enc(x_dim, z_dim)
        self.dynamics = dyn(armotized, z_dim, u_dim)
        self.backward_dynamics = back_dyn(z_dim, u_dim, x_dim)

    def encode(self, x):
        return self.encoder(x)

    def transition(self, z, u):
        return self.dynamics(z, u)

    def back_dynamics(self, z, u, x):
        return self.backward_dynamics(z, u, x)

    def reparam(self, mean, std):
        epsilon = torch.randn_like(std)
        return mean + torch.mul(epsilon, std)

    def forward(self, x, u, x_next):
        # NCE loss
        z_enc_dist = self.encode(x)  # P(z_t | x_t)
        z_enc = self.reparam(z_enc_dist.mean, z_enc_dist.stddev) # sample z_t from P(z_t | x_t)
        z_next_trans_dist, _, _ = self.transition(z_enc, u) # P(z^_t+1 | z_t, u _t)

        z_next_enc_dist = self.encode(x_next)  # Q(z^_t+1 | x_t+1)
        z_next_enc = self.reparam(z_next_enc_dist.mean, z_next_enc_dist.stddev)  # sample z^_t+1 from Q(z^_t+1 | x_t+1)

        # consistency loss
        # 2nd and 3rd term
        z_backward_dist = self.back_dynamics(z_next_enc, u, x) # Q(z_t | z^_t+1, u_t, x_t)
        z_backward = self.reparam(z_backward_dist.mean, z_backward_dist.stddev) # sample z_t from Q(z_t | z^_t+1, u_t, x_t)
        z_next_back_trans_dist, _, _ = self.transition(z_backward, u)

        return z_enc_dist, z_enc, z_next_trans_dist, \
                z_next_enc_dist, z_next_enc, \
                z_backward_dist, z_backward, z_next_back_trans_dist
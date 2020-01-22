import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

torch.set_default_dtype(torch.float64)

def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)

class Encoder(nn.Module):
    # deterministic encoder q(z | x)
    def __init__(self, net, x_dim, z_dim):
        super(Encoder, self).__init__()
        self.net = net
        self.x_dim = x_dim
        self.z_dim = z_dim

    def forward(self, x):
        return self.net(x)

class Dynamics(nn.Module):
    # stochastic transition model: P(z^_t+1 | z_t, u_t)
    def __init__(self, net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized):
        super(Dynamics, self).__init__()
        self.net_hidden = net_hidden
        self.net_mean = net_mean
        self.net_logstd = net_logstd
        self.net_A = net_A
        self.net_B = net_B
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.armotized = armotized

    def forward(self, z_t, u_t):
        z_u_t = torch.cat((z_t, u_t), dim = -1)
        hidden_neurons = self.net_hidden(z_u_t)
        mean = self.net_mean(hidden_neurons) + z_t # skip connection
        logstd = self.net_logstd(hidden_neurons)
        if self.armotized:
            A = self.net_A(hidden_neurons)
            B = self.net_B(hidden_neurons)
        else:
            A, B = None, None
        return MultivariateNormalDiag(mean, torch.exp(logstd)), A, B

class PlanarEncoder(Encoder):
    def __init__(self, x_dim = 1600, z_dim = 2):
        net = nn.Sequential(
            nn.Linear(x_dim, 300),
            nn.ReLU(),

            nn.Linear(300, 300),
            nn.ReLU(),

            nn.Linear(300, z_dim)
        )
        super(PlanarEncoder, self).__init__(net, x_dim, z_dim)

class PlanarDynamics(Dynamics):
    def __init__(self, armotized, z_dim = 2, u_dim = 2):
        net_hidden = nn.Sequential(
            nn.Linear(z_dim + u_dim, 20),
            nn.ReLU(),

            nn.Linear(20, 20),
            nn.ReLU()
        )
        net_mean = nn.Linear(20, z_dim)
        net_logstd = nn.Linear(20, z_dim)
        if armotized:
            net_A = nn.Linear(20, z_dim**2)
            net_B = nn.Linear(20, u_dim*z_dim)
        else:
            net_A, net_B = None, None
        super(PlanarDynamics, self).__init__(net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized)

class PendulumEncoder(Encoder):
    def __init__(self, x_dim = 4608, z_dim = 3):
        net = nn.Sequential(
            nn.Linear(x_dim, 500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.ReLU(),

            nn.Linear(500, z_dim)
        )
        super(PendulumEncoder, self).__init__(net, x_dim, z_dim)

class PendulumDynamics(Dynamics):
    def __init__(self, armotized, z_dim = 3, u_dim = 1):
        net_hidden = nn.Sequential(
            nn.Linear(z_dim + u_dim, 30),
            nn.ReLU(),

            nn.Linear(30, 30),
            nn.ReLU()
        )
        net_mean = nn.Linear(30, z_dim)
        net_logstd = nn.Linear(30, z_dim)
        if armotized:
            net_A = nn.Linear(30, z_dim*z_dim)
            net_B = nn.Linear(30, u_dim*z_dim)
        else:
            net_A, net_B = None, None
        super(PendulumDynamics, self).__init__(net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized)

class MountainCarEncoder(Encoder):
    def __init__(self, x_dim = 4800, z_dim = 3):
        net = nn.Sequential(
            nn.Linear(x_dim, 500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.ReLU(),

            nn.Linear(500, z_dim)
        )
        super(MountainCarEncoder, self).__init__(net, x_dim, z_dim)

class MountainCarDynamics(Dynamics):
    def __init__(self, armotized, z_dim = 3, u_dim = 1):
        net_hidden = nn.Sequential(
            nn.Linear(z_dim + u_dim, 30),
            nn.ReLU(),

            nn.Linear(30, 30),
            nn.ReLU()
        )
        net_mean = nn.Linear(30, z_dim)
        net_logstd = nn.Linear(30, z_dim)
        if armotized:
            net_A = nn.Linear(30, z_dim*z_dim)
            net_B = nn.Linear(30, u_dim*z_dim)
        else:
            net_A, net_B = None, None
        super(MountainCarDynamics, self).__init__(net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized)

class ReacherGrayEncoder(Encoder):
    def __init__(self, x_dim=(2, 64, 64), z_dim=10):
        x_channels = x_dim[0]
        
        net_hidden = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=64, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=0),
            # nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=0),
            # nn.ReLU(),

            Flatten(),

        )
        
        flatten_shape = self.get_flatten_shape(net_hidden)
        net_hidden.add_module("linear", nn.Linear(flatten_shape, 256))
        net_hidden.add_module("relu", nn.ReLU())
        net_hidden.add_module("last", nn.Linear(256, z_dim))
        super(ReacherGrayEncoder, self).__init__(net_hidden, x_dim, z_dim)

    def get_flatten_shape(self, conv_net):
        with torch.no_grad():
            dummy = torch.zeros((1, 2, 64, 64))
            return (conv_net(dummy)).shape[-1]

class ReacherGrayDynamics(Dynamics):
    def __init__(self, armotized, z_dim=10, u_dim=2):
        net_hidden = nn.Sequential(
            nn.Linear(z_dim + u_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU()
        )
        net_mean = nn.Linear(64, z_dim)
        net_logstd = nn.Linear(64, z_dim)
        if armotized:
            net_A = nn.Linear(64, z_dim * z_dim)
            net_B = nn.Linear(64, u_dim * z_dim)
        else:
            net_A, net_B = None, None
        super(ReacherGrayDynamics, self).__init__(net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class CartPoleEncoder(Encoder):
    def __init__(self, x_dim=(2, 80, 80), z_dim=8):
        x_channels = x_dim[0]
        net = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            Flatten(),

            nn.Linear(10*10*10, 200),
            nn.ReLU(),

            nn.Linear(200, z_dim)
        )
        super(CartPoleEncoder, self).__init__(net, x_dim, z_dim)

class CartPoleDynamics(Dynamics):
    def __init__(self, armotized, z_dim=8, u_dim=1):
        net_hidden = nn.Sequential(
            nn.Linear(z_dim + u_dim, 40),
            nn.ReLU(),

            nn.Linear(40, 40),
            nn.ReLU()
        )
        net_mean = nn.Linear(40, z_dim)
        net_logstd = nn.Linear(40, z_dim)
        if armotized:
            net_A = nn.Linear(40, z_dim * z_dim)
            net_B = nn.Linear(40, u_dim * z_dim)
        else:
            net_A, net_B = None, None
        super(CartPoleDynamics, self).__init__(net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized)

class ThreePoleEncoder(Encoder):
    def __init__(self, x_dim=(2, 80, 80), z_dim=8):
        x_channels = x_dim[0]
        net = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            Flatten(),

            nn.Linear(10*10*10, 200),
            nn.ReLU(),

            nn.Linear(200, z_dim)
        )
        super(ThreePoleEncoder, self).__init__(net, x_dim, z_dim)

class ThreePoleDynamics(Dynamics):
    def __init__(self, armotized, z_dim=8, u_dim=3):
        net_hidden = nn.Sequential(
            nn.Linear(z_dim + u_dim, 40),
            nn.ReLU(),

            nn.Linear(40, 40),
            nn.ReLU()
        )
        net_mean = nn.Linear(40, z_dim)
        net_logstd = nn.Linear(40, z_dim)
        if armotized:
            net_A = nn.Linear(40, z_dim * z_dim)
            net_B = nn.Linear(40, u_dim * z_dim)
        else:
            net_A, net_B = None, None
        super(ThreePoleDynamics, self).__init__(net_hidden, net_mean, net_logstd, net_A, net_B, z_dim, u_dim, armotized)

CONFIG = {
    'planar': (PlanarEncoder, PlanarDynamics),
    'pendulum': (PendulumEncoder, PendulumDynamics),
    'pendulum_gym': (PendulumEncoder, PendulumDynamics),
    'cartpole': (CartPoleEncoder, CartPoleDynamics),
    'mountain_car': (MountainCarEncoder, MountainCarDynamics),
    'threepole': (ThreePoleEncoder, ThreePoleDynamics),
    'reacher': (ReacherGrayEncoder, ReacherGrayDynamics),
}

def load_config(name):
    return CONFIG[name]

__all__ = ['load_config']
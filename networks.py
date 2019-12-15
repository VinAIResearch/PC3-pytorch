import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

torch.set_default_dtype(torch.float64)

def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)

def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        # torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

class Encoder(nn.Module):
    # P(z_t | x_t) and Q(z^_t+1 | x_t+1)
    def __init__(self, net_hidden, net_mean, net_logstd, x_dim, z_dim):
        super(Encoder, self).__init__()
        self.net_hidden = net_hidden
        self.net_mean = net_mean
        self.net_logstd = net_logstd
        self.x_dim = x_dim
        self.z_dim = z_dim

        # self.net_hidden.apply(weights_init)
        # self.net_mean.apply(weights_init)
        # self.net_logstd.apply(weights_init)

    def forward(self, x):
        # mean and variance of p(z|x)
        hidden_neurons = self.net_hidden(x)
        mean = self.net_mean(hidden_neurons)
        logstd = self.net_logstd(hidden_neurons)
        return MultivariateNormalDiag(mean, torch.exp(logstd))

class Dynamics(nn.Module):
    # P(z^_t+1 | z_t, u_t)
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

        # self.net_hidden.apply(weights_init)
        # self.net_mean.apply(weights_init)
        # self.net_logstd.apply(weights_init)
        # if armotized:
        #     self.net_A.apply(weights_init)
        #     self.net_B.apply(weights_init)

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

class BackwardDynamics(nn.Module):
    # Q(z_t | z^_t+1, x_t, u_t)
    def __init__(self, net_z, net_u, net_x, net_joint_hidden, net_joint_mean, net_joint_logstd, z_dim, u_dim, x_dim):
        super(BackwardDynamics, self).__init__()
        self.net_z = net_z
        self.net_u = net_u
        self.net_x = net_x
        self.net_joint_hidden = net_joint_hidden
        self.net_joint_mean = net_joint_mean
        self.net_joint_logstd = net_joint_logstd
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.x_dim = x_dim

        # self.net_z.apply(weights_init)
        # self.net_u.apply(weights_init)
        # self.net_x.apply(weights_init)
        # self.net_joint_hidden.apply(weights_init)
        # self.net_joint_mean.apply(weights_init)
        # self.net_joint_logstd.apply(weights_init)

    def forward(self, z_t, u_t, x_t):
        z_t_out = self.net_z(z_t)
        u_t_out = self.net_u(u_t)
        x_t_out = self.net_x(x_t)

        hidden_neurons = self.net_joint_hidden(torch.cat((z_t_out, u_t_out, x_t_out), dim = -1))
        mean = self.net_joint_mean(hidden_neurons)
        logstd = self.net_joint_logstd(hidden_neurons)
        return MultivariateNormalDiag(mean, torch.exp(logstd))

class PlanarEncoder(Encoder):
    def __init__(self, x_dim = 1600, z_dim = 2):
        net_hidden = nn.Sequential(
            nn.Linear(x_dim, 300),
            nn.ReLU(),

            nn.Linear(300, 300),
            nn.ReLU(),
        )
        net_mean = nn.Linear(300, z_dim)
        net_logstd = nn.Linear(300, z_dim)
        super(PlanarEncoder, self).__init__(net_hidden, net_mean, net_logstd, x_dim, z_dim)

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

class PlanarBackwardDynamics(BackwardDynamics):
    def __init__(self, z_dim=2, u_dim=2, x_dim=1600):
        net_z = nn.Linear(z_dim, 5)
        net_u = nn.Linear(u_dim, 5)
        net_x = nn.Linear(x_dim, 100)
        net_joint_hidden = nn.Sequential(
            nn.Linear(5 + 5 + 100, 100),
            nn.ReLU(),
        )
        net_joint_mean = nn.Linear(100, z_dim)
        net_joint_logstd = nn.Linear(100, z_dim)
        super(PlanarBackwardDynamics, self).__init__(net_z, net_u, net_x, net_joint_hidden, net_joint_mean, net_joint_logstd, z_dim, u_dim, x_dim)

class PendulumEncoder(Encoder):
    def __init__(self, x_dim = 4608, z_dim = 3):
        net_hidden = nn.Sequential(
            nn.Linear(x_dim, 500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.ReLU(),
        )
        net_mean = nn.Linear(500, z_dim)
        net_logstd = nn.Linear(500, z_dim)
        super(PendulumEncoder, self).__init__(net_hidden, net_mean, net_logstd, x_dim, z_dim)

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

class PendulumBackwardDynamics(BackwardDynamics):
    def __init__(self, z_dim=3, u_dim=1, x_dim=4608):
        net_z = nn.Linear(z_dim, 10)
        net_u = nn.Linear(u_dim, 10)
        net_x = nn.Linear(x_dim, 200)
        net_joint_hidden = nn.Sequential(
            nn.Linear(10 + 10 + 200, 200),
            nn.ReLU(),
        )
        net_joint_mean = nn.Linear(200, z_dim)
        net_joint_logstd = nn.Linear(200, z_dim)
        super(PendulumBackwardDynamics, self).__init__(net_z, net_u, net_x, net_joint_hidden, net_joint_mean, net_joint_logstd, z_dim, u_dim, x_dim)

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
        net_hidden = nn.Sequential(
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
        )
        net_mean = nn.Linear(200, z_dim)
        net_logstd = nn.Linear(200, z_dim)
        super(CartPoleEncoder, self).__init__(net_hidden, net_mean, net_logstd, x_dim, z_dim)

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

class CartPoleBackwardDynamics(BackwardDynamics):
    def __init__(self, z_dim=8, u_dim=1, x_dim=(2, 80, 80)):
        net_z = nn.Linear(z_dim, 10)
        net_u = nn.Linear(u_dim, 10)
        net_x = nn.Sequential(
            Flatten(),
            nn.Linear(x_dim[0] * x_dim[1] * x_dim[2], 300)
        )

        net_joint_hidden = nn.Sequential(
            nn.Linear(10 + 10 + 300, 300),
            nn.ReLU(),
        )
        net_joint_mean = nn.Linear(300, z_dim)
        net_joint_logstd = nn.Linear(300, z_dim)
        super(CartPoleBackwardDynamics, self).__init__(net_z, net_u, net_x, net_joint_hidden, net_joint_mean, net_joint_logstd, z_dim, u_dim, x_dim)

CONFIG = {
    'planar': (PlanarEncoder, PlanarDynamics, PlanarBackwardDynamics),
    'pendulum': (PendulumEncoder, PendulumDynamics, PendulumBackwardDynamics),
    'cartpole': (CartPoleEncoder, CartPoleDynamics, CartPoleBackwardDynamics)
}

def load_config(name):
    return CONFIG[name]

__all__ = ['load_config']
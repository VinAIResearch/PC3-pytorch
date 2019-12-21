import torch
from networks import MultivariateNormalDiag
from torch.distributions.kl import kl_divergence
import sys

torch.set_default_dtype(torch.float64)

def repeat_dist(dist, bacth_size, z_dim):
    # [dist1, dist2, dist3] -> [dist1, dist1, dist1, dist2, dist2, dist2, dist3, dist3, dist3]
    mean, sttdev = dist.mean, dist.stddev
    mean = mean.repeat(1, bacth_size).view(-1, z_dim)
    sttdev = sttdev.repeat(1, bacth_size).view(-1, z_dim)
    return MultivariateNormalDiag(mean, sttdev)

def nce(z_next_trans_dist, z_next_enc):
    """
    z_next_trans_dist: p(.|z, u)
    z_next_enc: samples from p(.|x')
    """
    batch_size, z_dim = z_next_enc.size(0), z_next_enc.size(1)

    z_next_trans_dist_rep = repeat_dist(z_next_trans_dist, batch_size, z_dim)
    z_next_enc_rep = z_next_enc.repeat(batch_size, 1)

    # scores[i, j] = p(z'_j | z_i, u_i)
    scores = z_next_trans_dist_rep.log_prob(z_next_enc_rep).view(batch_size, batch_size)
    with torch.no_grad():
        normalize = torch.max(scores, dim=-1)[0].view(-1,1)
    scores = scores - normalize
    scores = torch.exp(scores)

    # I_NCE
    positive_samples = scores.diag()
    avg_negative_samples = torch.mean(scores, dim=-1)
    return - torch.mean(torch.log(positive_samples / avg_negative_samples + 1e-8))

def curvature(model, z, u, delta, armotized):
    z_alias = z.detach().requires_grad_(True)
    u_alias = u.detach().requires_grad_(True)
    eps_z = torch.normal(mean=torch.zeros_like(z), std=torch.empty_like(z).fill_(delta))
    eps_u = torch.normal(mean=torch.zeros_like(u), std=torch.empty_like(u).fill_(delta))

    z_bar = z_alias + eps_z
    u_bar = u_alias + eps_u

    f_z_bar, A_bar, B_bar = model.transition(z_bar, u_bar)
    f_z_bar = f_z_bar.mean
    f_z, A, B = model.transition(z_alias, u_alias)
    f_z = f_z.mean

    if not armotized:
        grad_z, grad_u = torch.autograd.grad(f_z, [z_alias, u_alias], grad_outputs=[eps_z, eps_u], retain_graph=True, create_graph=True)
        taylor_error = f_z_bar - (grad_z + grad_u) - f_z
        cur_loss = torch.mean(torch.sum(taylor_error.pow(2), dim = 1))
    else:
        z_dim, u_dim = z.size(1), u.size(1)
        A_bar = A_bar.view(-1, z_dim, z_dim)
        B_bar = B_bar.view(-1, z_dim, u_dim)
        eps_z = eps_z.view(-1, z_dim, 1)
        eps_u = eps_u.view(-1, u_dim, 1)
        taylor_error = f_z_bar - (torch.bmm(A_bar, eps_z).squeeze() + torch.bmm(B_bar, eps_u).squeeze()) - f_z
        cur_loss = torch.mean(torch.sum(taylor_error.pow(2), dim = 1))
    return cur_loss

def new_curvature(model, z, u):
    z_next, _, _, = model.dynamics(z, u)
    z_next = z_next.mean
    temp_z = z - z.mean(dim=0)
    temp_z_next = z_next - z_next.mean(dim=0)
    temp_u = u - u.mean(dim=0)

    cov_z_z_next = torch.sum(temp_z * temp_z_next)**2
    var_prod_z_z_next = torch.sum(temp_z ** 2) * torch.sum(temp_z_next ** 2)

    cov_u_z_next = torch.sum(temp_u * temp_z_next)**2
    var_prod_u_z_next = torch.sum(temp_u ** 2) * torch.sum(temp_z_next ** 2)
    # print ('z next u: ' + str(cov_z_z_next / var_prod_z_z_next))
    # return - cov_z_z_next / var_prod_z_z_next - cov_u_z_next / var_prod_u_z_next
    return - cov_z_z_next / var_prod_z_z_next
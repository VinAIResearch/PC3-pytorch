import torch
from networks import MultivariateNormalDiag
from torch.distributions.kl import kl_divergence
import sys

torch.set_default_dtype(torch.float64)

def repeat_dist(dist, bacth_size, z_dim):
    # [dist1, dist2, dist3] -> [dist1, dist2, dist3, dist1, dist2, dist3, dist1, dist2, dist3]
    mean, sttdev = dist.mean, dist.stddev
    mean = mean.repeat(bacth_size, 1)
    sttdev = sttdev.repeat(bacth_size, 1)
    return MultivariateNormalDiag(mean, sttdev)

def nce_past(z_next_trans_dist, z_next_enc):
    """
    z_next_trans_dist: p(.|z, u)
    z_next_enc: samples from p(.|x')
    """
    batch_size, z_dim = z_next_enc.size(0), z_next_enc.size(1)

    z_next_trans_dist_rep = repeat_dist(z_next_trans_dist, batch_size, z_dim)
    z_next_enc_rep = z_next_enc.repeat(1, batch_size).view(-1, z_dim)

    # scores[i, j] = p(z'_i | z_j, u_j)
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

    f_z_bar, A, B = model.transition(z_bar, u_bar)
    f_z_bar = f_z_bar.mean
    f_z, A, B = model.transition(z_alias, u_alias)
    f_z = f_z.mean

    z_dim, u_dim = z.size(1), u.size(1)
    if not armotized:
        _, B = get_jacobian(model.dynamics, z_alias, u_alias)
    else:
        A = A.view(-1, z_dim, z_dim)
        B = B.view(-1, z_dim, u_dim)

    grad_z, = torch.autograd.grad(f_z, z_alias, grad_outputs=eps_z, retain_graph=True, create_graph=True)
    grad_u = torch.bmm(B, eps_u.view(-1, u_dim, 1)).squeeze()
    taylor_error = f_z_bar - (grad_z + grad_u) - f_z
    cur_loss = torch.mean(torch.sum(taylor_error.pow(2), dim = 1))
    return cur_loss

def get_jacobian(dynamics, batched_z, batched_u):
    """
    compute the jacobian of F(z,u) w.r.t z, u
    """
    batch_size = batched_z.size(0)
    z_dim = batched_z.size(-1)
    u_dim = batched_u.size(-1)
    
    z, u = batched_z.unsqueeze(1), batched_u.unsqueeze(1) # batch_size, 1, input_dim
    z, u = z.repeat(1, z_dim, 1), u.repeat(1, z_dim, 1) # batch_size, output_dim, input_dim
    z_next = dynamics(z, u)[0].mean
    grad_inp = torch.eye(z_dim).reshape(1, z_dim, z_dim).repeat(batch_size, 1, 1).cuda()
    all_A, all_B = torch.autograd.grad(z_next, [z, u], [grad_inp, grad_inp], create_graph=True, retain_graph=True)
    return all_A, all_B
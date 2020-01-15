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

def repeat_dist_2(dist, bacth_size, z_dim):
    # [dist1, dist2, dist3] -> [dist1, dist1, dist1, dist2, dist2, dist2, dist3, dist3, dist3]
    mean, sttdev = dist.mean, dist.stddev
    mean = mean.repeat(bacth_size, 1)
    sttdev = sttdev.repeat(bacth_size, 1)
    return MultivariateNormalDiag(mean, sttdev)

def nce_1(z_next_trans_dist, z_next_enc):
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

def nce_2(z_next_trans_dist, z_next_enc):
    """
    z_next_trans_dist: p(.|z, u)
    z_next_enc: samples from p(.|x')
    """
    batch_size, z_dim = z_next_enc.size(0), z_next_enc.size(1)

    z_next_trans_dist_rep = repeat_dist_2(z_next_trans_dist, batch_size, z_dim)
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
        A, B = jacobian(model.dynamics, z_alias, u_alias)
    else:
        A = A.view(-1, z_dim, z_dim)
        B = B.view(-1, z_dim, u_dim)

    eps_z = eps_z.view(-1, z_dim, 1)
    eps_u = eps_u.view(-1, u_dim, 1)
    taylor_error = f_z_bar - (torch.bmm(A, eps_z).squeeze() + torch.bmm(B, eps_u).squeeze()) - f_z
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

    # cov_u_z_next = torch.sum(temp_u * temp_z_next)**2
    # var_prod_u_z_next = torch.sum(temp_u ** 2) * torch.sum(temp_z_next ** 2)
    # print ('z next u: ' + str(cov_z_z_next / var_prod_z_z_next))
    # return - cov_z_z_next / var_prod_z_z_next - cov_u_z_next / var_prod_u_z_next
    return - cov_z_z_next / var_prod_z_z_next

def jacobian(dynamics, batched_z, batched_u):
    """
    compute the jacobian of F(z,u) w.r.t z, u
    """
    batch_size = batched_z.size(0)
    z_dim = batched_z.size(-1)
    u_dim = batched_u.size(-1)
    all_A, all_B = torch.empty(size=(batch_size, z_dim, z_dim)).cuda(),\
                    torch.empty(size=(batch_size, z_dim, u_dim)).cuda()
    for i in range(batch_size):
        z = batched_z[i].view(1,-1)
        u = batched_u[i].view(1,-1)
        
        z, u = z.squeeze().repeat(z_dim, 1), u.squeeze().repeat(z_dim, 1)
        z_next = dynamics(z, u)[0].mean
        grad_inp = torch.eye(z_dim).cuda()
        A, B = torch.autograd.grad(z_next, [z, u], [grad_inp, grad_inp], create_graph=True, retain_graph=True)
        all_A[i,:,:], all_B[i,:,:] = A, B
    return all_A, all_B
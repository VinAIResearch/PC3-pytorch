from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import argparse
import json

from pcc_model import PCC
from datasets import *
from losses import *
from networks import MultivariateNormalDiag

from latent_map_planar import *
from latent_map_pendulum import *

torch.set_default_dtype(torch.float64)

device = torch.device("cuda")
datasets = {'planar': PlanarDataset, 'pendulum': PendulumDataset, 'cartpole': CartPoleDataset, 'pendulum_gym': PendulumGymDataset, 'mountain_car': MountainCarDataset}
dims = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1), 'cartpole': ((2, 80, 80), 8, 1), 'pendulum_gym': (4608, 3, 1), 'mountain_car': (4800, 3, 1)}

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_loss(model, armotized, u,
                z_enc, z_next_trans_dist, z_next_enc,
                lam, delta=0.1, norm_coeff=0.01):
    # nce and consistency loss
    # nce_loss = nce_1(z_next_trans_dist, z_next_enc) # sampling future
    nce_loss = nce_2(z_next_trans_dist, z_next_enc) # sampling past

    consis_loss = - torch.mean(z_next_trans_dist.log_prob(z_next_enc))

    # curvature loss
    cur_loss = curvature(model, z_enc, u, delta, armotized)
    # new_cur_loss = new_curvature(model, z_enc, u)

    # additional norm loss to center z range to (0,0)
    norm_loss = torch.sum(torch.mean(z_enc, dim=0).pow(2))

    # additional norm loss to avoid collapsing
    avg_norm_2 = torch.mean(torch.sum(z_enc.pow(2), dim=1))

    lam_nce, lam_c, lam_cur = lam
    return nce_loss, consis_loss, cur_loss, norm_loss, avg_norm_2,\
        lam_nce * nce_loss + lam_c * consis_loss + lam_cur * cur_loss + norm_coeff * norm_loss

def train(model, train_loader, lam, norm_coeff, latent_noise, optimizer, armotized, epoch):
    avg_nce_loss = 0.0
    avg_consis_loss = 0.0
    avg_cur_loss = 0.0
    avg_norm_loss = 0.0
    avg_norm_2_loss = 0.0
    avg_loss = 0.0

    num_batches = len(train_loader)
    model.train()

    for iter, (x, u, x_next) in enumerate(train_loader):
        x = x.to(device).double()
        u = u.to(device).double()
        x_next = x_next.to(device).double()
        optimizer.zero_grad()

        z_enc, z_next_trans_dist, z_next_enc = model(x, u, x_next)
        noise = torch.randn(size=z_next_enc.size()) * latent_noise
        if next(model.encoder.parameters()).is_cuda:
            noise = noise.cuda()
        z_next_enc += noise

        nce_loss, consis_loss, cur_loss, norm_loss, norm_2, loss = compute_loss(
                model, armotized, u,
                z_enc, z_next_trans_dist, z_next_enc,
                lam=lam, norm_coeff=norm_coeff)

        loss.backward()
        optimizer.step()

        avg_nce_loss += nce_loss.item()
        avg_consis_loss += consis_loss.item()
        avg_cur_loss += cur_loss.item()
        avg_norm_loss += norm_loss.item()
        avg_norm_2_loss += norm_2.item()
        avg_loss += loss.item()

    avg_nce_loss /= num_batches
    avg_consis_loss /= num_batches
    avg_cur_loss /= num_batches
    avg_norm_loss /= num_batches
    avg_norm_2_loss /= num_batches
    avg_loss /= num_batches

    if (epoch + 1) % 1 == 0:
        print('Epoch %d' % (epoch+1))
        print("NCE loss: %f" % (avg_nce_loss))
        print("Consistency loss: %f" % (avg_consis_loss))
        print("Curvature loss: %f" % (avg_cur_loss))
        print("Normalization loss: %f" % (avg_norm_loss))
        print("Norma 2 loss: %f" % (avg_norm_2_loss))
        print("Training loss: %f" % (avg_loss))
        print('--------------------------------------')

    return avg_nce_loss, avg_consis_loss, avg_cur_loss, avg_loss

def main(args):
    env_name = args.env
    assert env_name in ['planar', 'pendulum', 'pendulum_gym', 'cartpole', 'mountain_car']
    armotized = args.armotized
    log_dir = args.log_dir
    seed = args.seed
    data_size = args.data_size
    noise_level = args.noise
    batch_size = args.batch_size
    lam_nce = args.lam_nce
    lam_c = args.lam_c
    lam_cur = args.lam_cur
    lam = [lam_nce, lam_c, lam_cur]
    norm_coeff = args.norm_coeff
    lr = args.lr
    latent_noise = args.latent_noise
    weight_decay = args.decay
    epoches = args.num_iter
    iter_save = args.iter_save
    save_map = args.save_map

    seed_torch(seed)

    dataset = datasets[env_name]
    data = dataset(sample_size=data_size, noise=noise_level)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

    x_dim, z_dim, u_dim = dims[env_name]
    model = PCC(armotized=armotized, x_dim=x_dim, z_dim=z_dim, u_dim=u_dim, env=env_name).to(device)

    if save_map:
        if env_name == 'planar':
            mdp = PlanarObstaclesMDP(noise=noise_level)
        elif env_name == 'pendulum':
            mdp = PendulumMDP(noise=noise_level)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=weight_decay)

    log_path = 'logs/' + env_name + '/' + log_dir
    if not path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    result_path = 'result/' + env_name + '/' + log_dir
    if not path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + '/settings', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if save_map:
        if env_name == 'planar':
            latent_maps = [draw_latent_map(model, mdp)]
        elif env_name == 'pendulum':
            show_latent_map(model, mdp)
    for i in range(epoches):
        avg_pred_loss, avg_consis_loss, avg_cur_loss, avg_loss = train(model, data_loader,
                                                                lam, norm_coeff, latent_noise, optimizer, armotized, i)

        # ...log the running loss
        writer.add_scalar('NCE loss', avg_pred_loss, i)
        writer.add_scalar('consistency loss', avg_consis_loss, i)
        writer.add_scalar('curvature loss', avg_cur_loss, i)
        writer.add_scalar('training loss', avg_loss, i)
        if save_map and (i+1) % 10 == 0:
            if env_name == 'planar':
                map_i = draw_latent_map(model, mdp)
                latent_maps.append(map_i)
            else:
                show_latent_map(model, mdp)
        # save model
        if (i + 1) % iter_save == 0:
            print('Saving the model.............')

            torch.save(model.state_dict(), result_path + '/model_' + str(i + 1))
            with open(result_path + '/loss_' + str(i + 1), 'w') as f:
                f.write('\n'.join([
                                'NCE loss: ' + str(avg_pred_loss),
                                'Consistency loss: ' + str(avg_consis_loss),
                                'Curvature loss: ' + str(avg_cur_loss),
                                'Training loss: ' + str(avg_loss)
                                ]))
    if env_name == 'planar' and save_map:
        latent_maps[0].save(result_path + '/latent_map.gif', format='GIF', append_images=latent_maps[1:], save_all=True, duration=100, loop=0)
    writer.close()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train pcc model')

    parser.add_argument('--env', required=True, type=str, help='environment used for training')
    parser.add_argument('--armotized', required=True, type=str2bool, nargs='?',
                        const=True, default=False, help='type of dynamics model')
    parser.add_argument('--log_dir', required=True, type=str, help='directory to save training log')
    parser.add_argument('--seed', required=True, type=int, help='seed number')
    parser.add_argument('--data_size', required=True, type=int, help='the bumber of data points used for training')
    parser.add_argument('--noise', default=0, type=float, help='the level of noise')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--lam_nce', default=1.0, type=float, help='weight of prediction loss')
    parser.add_argument('--lam_c', default=1.0, type=float, help='weight of consistency loss')
    parser.add_argument('--lam_cur', default=1.0, type=float, help='weight of curvature loss')
    parser.add_argument('--norm_coeff', default=0.1, type=float, help='coefficient of additional normalization loss')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--latent_noise', default=0.1, type=float, help='level of noise added to the latent code')
    parser.add_argument('--decay', default=0.001, type=float, help='L2 regularization')
    parser.add_argument('--num_iter', default=2000, type=int, help='number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, help='save model and result after this number of iterations')
    parser.add_argument('--save_map', default=False, type=str2bool, help='save the latent map during training or not')
    args = parser.parse_args()

    main(args)
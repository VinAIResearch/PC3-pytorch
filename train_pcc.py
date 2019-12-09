from tensorboardX import SummaryWriter
import torch
from torch.distributions.kl import kl_divergence
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import random
import argparse
import json

from pcc_model import PCC
from datasets import *
from losses import *
from networks import MultivariateNormalDiag

from latent_map_planar import *

torch.set_default_dtype(torch.float64)

device = torch.device("cuda")
datasets = {'planar': PlanarDataset, 'pendulum': PendulumDataset, 'cartpole': CartPoleDataset}
dims = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1), 'cartpole': ((2, 80, 80), 8, 1)}

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
                z_enc_dist, z_enc, z_next_trans_dist,
                z_next_enc_dist, z_next_enc,
                z_backward_dist, z_next_back_trans_dist,
                lam, delta=0.1, norm_coeff=0.01):
    # nce and consistency loss
    nce_loss = nce(z_next_trans_dist, z_next_enc)

    consis_loss = - torch.mean(z_next_enc_dist.entropy()) \
                  - torch.mean(z_next_back_trans_dist.log_prob(z_next_enc)) \
                  + torch.mean(kl_divergence(z_backward_dist, z_enc_dist))

    # curvature loss
    cur_loss = curvature(model, z_enc, u, delta, armotized)
    # new_cur_loss = new_curvature(model, z_enc, u)

    # additional norm lossL to normalize the latent space
    norm_loss = torch.mean(kl_divergence(z_enc_dist, MultivariateNormalDiag(torch.zeros_like(z_enc_dist.mean),
                                                         torch.ones_like(z_enc_dist.stddev))))

    lam_nce, lam_c, lam_cur = lam
    return nce_loss, consis_loss, cur_loss, lam_nce * nce_loss + lam_c * consis_loss + lam_cur * cur_loss + norm_coeff * norm_loss
    # return nce_loss, consis_loss, cur_loss, lam_nce * nce_loss

def train(model, train_loader, lam, norm_coeff, optimizer, armotized, epoch):
    avg_nce_loss = 0.0
    avg_consis_loss = 0.0
    avg_cur_loss = 0.0
    avg_loss = 0.0

    num_batches = len(train_loader)
    model.train()
    for iter, (x, u, x_next) in enumerate(train_loader):
        # print ('epoch %d minibatch %d' %(epoch, iter))
        x = x.to(device).double()
        u = u.to(device).double()
        x_next = x_next.to(device).double()
        optimizer.zero_grad()

        z_enc_dist, z_enc, z_next_trans_dist, \
        z_next_enc_dist, z_next_enc, \
        z_backward_dist, z_backward, z_next_back_trans_dist = model(x, u, x_next)

        nce_loss, consis_loss, cur_loss, loss = compute_loss(
                model, armotized, u,
                z_enc_dist, z_enc, z_next_trans_dist,
                z_next_enc_dist, z_next_enc,
                z_backward_dist, z_next_back_trans_dist,
                lam=lam, norm_coeff=norm_coeff)

        loss.backward()
        optimizer.step()

        avg_nce_loss += nce_loss.item()
        avg_consis_loss += consis_loss.item()
        avg_cur_loss += cur_loss.item()
        avg_loss += loss

    avg_nce_loss /= num_batches
    avg_consis_loss /= num_batches
    avg_cur_loss /= num_batches
    avg_loss /= num_batches

    if (epoch + 1) % 1 == 0:
        print('Epoch %d' % (epoch+1))
        print("NCE loss: %f" % (avg_nce_loss))
        print("Consistency loss: %f" % (avg_consis_loss))
        print("Curvature loss: %f" % (avg_cur_loss))
        print("Training loss: %f" % (avg_loss))
        print('--------------------------------------')

    return avg_nce_loss, avg_consis_loss, avg_cur_loss, avg_loss

def main(args):
    env_name = args.env
    assert env_name in ['planar', 'pendulum', 'cartpole']
    armotized = args.armotized
    log_dir = args.log_dir
    seed = args.seed
    data_size = args.data_size
    noise_level = args.noise
    batch_size = args.batch_size
    lam_nce = args.lam_nce
    lam_c = args.lam_c
    lam_cur = args.lam_cur
    lam = (lam_nce, lam_c, lam_cur)
    norm_coeff = args.norm_coeff
    lr = args.lr
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

    if env_name == 'planar' and save_map:
        mdp = PlanarObstaclesMDP(noise=noise_level)

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=int(epoches / 3), gamma=0.5)

    log_path = 'logs/' + env_name + '/' + log_dir
    if not path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    result_path = 'result/' + env_name + '/' + log_dir
    if not path.exists(result_path):
        os.makedirs(result_path)
    with open(result_path + '/settings', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if env_name == 'planar' and save_map:
        latent_maps = [draw_latent_map(model, mdp)]
    for i in range(epoches):
        avg_pred_loss, avg_consis_loss, avg_cur_loss, avg_loss = train(model, data_loader, lam,
                                                                       norm_coeff, optimizer, armotized, i)
        scheduler.step()

        # ...log the running loss
        writer.add_scalar('NCE loss', avg_pred_loss, i)
        writer.add_scalar('consistency loss', avg_consis_loss, i)
        writer.add_scalar('curvature loss', avg_cur_loss, i)
        writer.add_scalar('training loss', avg_loss, i)
        if env_name == 'planar' and save_map:
            if (i+1) % 10 == 0:
                map_i = draw_latent_map(model, mdp)
                latent_maps.append(map_i)
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
    parser.add_argument('--noise', default=0, type=int, help='the level of noise')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--lam_nce', default=1.0, type=float, help='weight of prediction loss')
    parser.add_argument('--lam_c', default=10.0, type=float, help='weight of consistency loss')
    parser.add_argument('--lam_cur', default=8.0, type=float, help='weight of curvature loss')
    parser.add_argument('--norm_coeff', default=0.01, type=float, help='coefficient of additional normalization loss')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.001, type=float, help='L2 regularization')
    parser.add_argument('--num_iter', default=5000, type=int, help='number of epoches')
    parser.add_argument('--iter_save', default=1000, type=int, help='save model and result after this number of iterations')
    parser.add_argument('--save_map', default=False, type=str2bool, help='save the latent map during training or not')
    args = parser.parse_args()

    main(args)
import numpy as np
import torch
import argparse
import json
import os
from torch.utils.data import DataLoader

from pcc_model import PCC
from datasets import PlanarDataset, PendulumDataset, CartPoleDataset

env_data_dim = {'planar': (1600, 2, 2), 'pendulum': ((2,48,48), 3, 1), 'cartpole': ((2,80,80), 8, 1)}
datasets = {'planar': PlanarDataset, 'pendulum': PendulumDataset, 'cartpole': CartPoleDataset}

def calc_avg_norm_2(model, env_name, sample_size=5000, noise=0):
    dataset = datasets[env_name]
    dataset = dataset(sample_size=sample_size, noise=noise)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False, num_workers=1)

    avg_norm_2 = 0.0
    for x, _, _ in data_loader:
        with torch.no_grad():
            z = model.encode(x)
            avg_norm_2 += torch.mean(torch.sum(z.pow(2), dim=1))
    return avg_norm_2 / len(data_loader)

def main(args):
    env_name = args.env
    assert env_name in ['planar', 'pendulum', 'pendulum_gym', 'cartpole', 'mountain_car']
    setting_path = args.setting_path
    epoch = args.epoch

    x_dim, z_dim, u_dim = env_data_dim[env_name]
    if env_name in ['planar', 'pendulum', 'pendulum_gym', 'mountain_car']:
        x_dim = np.prod(x_dim)

    all_avg_norm_2 = []
    log_folders = [os.path.join(setting_path, dI) for dI in os.listdir(setting_path)
                   if os.path.isdir(os.path.join(setting_path, dI))]
    for log in log_folders:
        with open(log + '/settings', 'r') as f:
            settings = json.load(f)
            armotized = settings['armotized']

        # load the trained model
        model = PCC(armotized, x_dim, z_dim, u_dim, env_name)
        model.load_state_dict(torch.load(log + '/model_' + str(epoch), map_location='cpu'))
        model.eval()

        avg_norm_2 = calc_avg_norm_2(model, env_name)
        all_avg_norm_2.append(avg_norm_2)

    # compute mean and variance
    # print (all_avg_norm_2)
    all_avg_norm_2 = np.array(all_avg_norm_2)
    mean = np.mean(all_avg_norm_2)
    var = np.var(all_avg_norm_2)

    print ('Mean of average norm 2: ' + str(mean))
    print ('Variance of average norm 2: ' + str(var))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute latent map scale statistics')

    parser.add_argument('--env', required=True, type=str, help='environment to compute statistics')
    parser.add_argument('--setting_path', required=True, type=str, help='path to 10 trained models of a setting')
    parser.add_argument('--epoch', required=True, type=int, help='load model corresponding to this epoch')
    args = parser.parse_args()

    main(args)
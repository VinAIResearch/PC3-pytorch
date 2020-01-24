import numpy as np
import torch
import argparse
import json
import os
from torch.utils.data import DataLoader

from pcc_model import PCC
from datasets import PlanarDataset, PendulumDataset, CartPoleDataset, ThreePoleDataset
from losses import nce_past, nce_future

env_data_dim = {'planar': (1600, 2, 2), 'pendulum': ((2,48,48), 3, 1), 'cartpole': ((2,80,80), 8, 1), 'threepole': ((2,80,80), 8, 3)}
datasets = {'planar': PlanarDataset, 'pendulum': PendulumDataset, 'cartpole': CartPoleDataset, 'threepole': ThreePoleDataset}

def past_vs_future(model, env_name, sample_size):
    dataset = datasets[env_name]
    dataset = dataset(sample_size=sample_size, noise=0)
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False, num_workers=1)

    avg_nce_past = 0.0
    avg_nce_future = 0.0
    for x, u, x_next in data_loader:
        with torch.no_grad():
            z_enc, z_next_trans_dist, z_next_enc = model(x, u, x_next)
            avg_nce_past += -nce_past(z_next_trans_dist, z_next_enc)
            avg_nce_future += -nce_future(z_next_trans_dist, z_next_enc)
    return avg_nce_past / len(data_loader), avg_nce_future / len(data_loader)

def main(args):
    env_name = args.env
    assert env_name in ['planar', 'pendulum', 'cartpole', 'threepole']
    data_size = args.data_size
    setting_path = args.setting_path
    epoch = args.epoch

    x_dim, z_dim, u_dim = env_data_dim[env_name]
    if env_name in ['planar', 'pendulum']:
        x_dim = np.prod(x_dim)

    all_avg_nce_past = []
    all_avg_nce_future = []
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

        avg_past, avg_future = past_vs_future(model, env_name, data_size)
        print ('Past: %.3f vs Future: %.3f' % (avg_past, avg_future))

        all_avg_nce_past.append(avg_past)
        all_avg_nce_future.append(avg_future)

    # compute mean and variance
    all_avg_nce_past, all_avg_nce_future = np.array(all_avg_nce_past), np.array(all_avg_nce_future)
    
    print ('Mean of NCE past: ' + str(np.mean(all_avg_nce_past)))
    print ('Mean of NCE future: ' + str(np.mean(all_avg_nce_future)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute latent map scale statistics')

    parser.add_argument('--env', required=True, type=str, help='environment to compute statistics')
    parser.add_argument('--data_size', required=True, type=int, help='number of data points')
    parser.add_argument('--setting_path', required=True, type=str, help='path to 10 trained models of a setting')
    parser.add_argument('--epoch', required=True, type=int, help='load model corresponding to this epoch')
    args = parser.parse_args()

    main(args)
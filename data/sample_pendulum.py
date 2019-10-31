"""get_pole_simple_dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from pathlib import Path
from tqdm import trange
import os.path as path
from PIL import Image
import json
from datetime import datetime
import argparse

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
os.sys.path.append(root_path)

from mdp.pole_simple_mdp import VisualPoleSimpleSwingUp

def sample(sample_size=20000, width=48, height=48, frequency=50, noise=0.0):
    """
    return [(x, u, x_next, s, s_next)]
    """
    mdp = VisualPoleSimpleSwingUp(height=height, width=width,
                                                  frequency=frequency,
                                                  noise=noise)

    # Data buffers to fill.
    x_data = np.zeros((sample_size, width, height, 2), dtype='float32')
    u_data = np.zeros((sample_size, mdp.action_dim), dtype='float32')
    x_next_data = np.zeros((sample_size, width, height, 2), dtype='float32')
    state_data = np.zeros((sample_size, 2, 2), dtype='float32')
    state_next_data = np.zeros((sample_size, 2, 2), dtype='float32')

    # Generate interaction tuples (random states and actions).
    for sample in trange(sample_size, desc = 'Sampling data'):
        s00 = mdp.sample_random_state()
        s01 = mdp.transition_function(s00, np.array([0.0]))

        a = np.atleast_1d(
            np.random.uniform(mdp.avail_force[0], mdp.avail_force[1]))

        s10 = mdp.transition_function(s00, a)
        s11 = mdp.transition_function(s10, np.array([0.0]))

        ## Store interaction tuple.
        # Current state (w/ history).
        x_data[sample, :, :, 0] = s00[1][:, :, 0]
        x_data[sample, :, :, 1] = s01[1][:, :, 0]
        state_data[sample, :, 0] = s00[0][0:2]
        state_data[sample, :, 1] = s01[0][0:2]
        # Action.
        u_data[sample] = a
        # Next state (w/ history).
        x_next_data[sample, :, :, 0] = s10[1][:, :, 0]
        x_next_data[sample, :, :, 1] = s11[1][:, :, 0]
        state_next_data[sample, :, 0] = s10[0][0:2]
        state_next_data[sample, :, 1] = s11[0][0:2]

    return x_data, u_data, x_next_data, state_data, state_next_data

def write_to_file(data, output_dir):
    """
    write [(x, u, x_next)] to output dir
    """
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    samples = []
    x_data, u_data, x_next_data, state_data, state_next_data = data

    for i in range(x_data.shape[0]):
        x_1 = x_data[i, :, :, 0]
        x_2 = x_data[i, :, :, 1]
        before = np.concatenate((x_1, x_2), axis=1)
        before_file = 'before-{:05d}.png'.format(i)
        Image.fromarray(before * 255.).convert('L').save(path.join(output_dir, before_file))

        after_file = 'after-{:05d}.png'.format(i)
        x_next_1 = x_next_data[i, :, :, 0]
        x_next_2 = x_next_data[i, :, :, 1]
        after = np.concatenate((x_next_1, x_next_2), axis=1)
        Image.fromarray(after * 255.).convert('L').save(path.join(output_dir, after_file))

        initial_state = state_data[i]
        after_state = state_next_data[i]

        samples.append({
            'before_state': initial_state.tolist(),
            'after_state': after_state.tolist(),
            'before': before_file,
            'after': after_file,
            'control': u_data[i].tolist(),
        })

    with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
        json.dump(
            {
                'metadata': {
                    'num_samples': x_data.shape[0],
                    'time_created': str(datetime.now()),
                    'version': 1
                },
                'samples': samples
            }, outfile, indent=2)

# data = sample(sample_size=1, width=48, height=48, frequency=50, noise=0.0)
def main(args):
    sample_size = args.sample_size
    noise = args.noise
    data = sample(sample_size=sample_size, width=48, height=48, frequency=50, noise=noise)
    write_to_file(data, root_path + '/data/pendulum/raw')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample planar data')

    parser.add_argument('--sample_size', required=True, type=int, help='the number of samples')
    parser.add_argument('--noise', default=0, type=int, help='level of noise')

    args = parser.parse_args()

    main(args)
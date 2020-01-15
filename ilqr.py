import argparse
import os
import json
import random

from pcc_model import PCC
from mdp.plane_obstacles_mdp import PlanarObstaclesMDP
from mdp.pendulum_mdp import PendulumMDP
from mdp.pendulum_gym import PendulumGymMDP
from mdp.cartpole_mdp import CartPoleMDP
from mdp.mountain_car_mdp import MountainCarMDP
from ilqr_utils import *

seed = 2020
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float64)

config_path = {'plane': 'ilqr_config/plane.json', 'swing': 'ilqr_config/swing.json', 'balance': 'ilqr_config/balance.json', 'cartpole': 'ilqr_config/cartpole.json',
               'swing_gym': 'ilqr_config/swing_gym.json', 'balance_gym': 'ilqr_config/balance_gym.json', 'mountain_car': 'ilqr_config/mountain_car.json'}
env_task = {'planar': ['plane'], 'pendulum': ['balance', 'swing'], 'cartpole': ['cartpole'],
            'pendulum_gym': ['balance_gym', 'swing_gym'], 'mountain_car': ['mountain_car']}
env_data_dim = {'planar': (1600, 2, 2), 'pendulum': ((2,48,48), 3, 1), 'cartpole': ((2,80,80), 8, 1), 'pendulum_gym': ((2,48,48), 3, 1), 'mountain_car': ((2,40,60),3,1)}

def main(args):
    env_name = args.env
    assert env_name in ['planar', 'pendulum', 'pendulum_gym', 'cartpole', 'mountain_car']
    possible_tasks = env_task[env_name]
    noise = args.noise
    epoch = args.epoch
    x_dim, z_dim, u_dim = env_data_dim[env_name]

    ilqr_result_path = 'iLQR_result/' + env_name
    if not os.path.exists(ilqr_result_path):
            os.makedirs(ilqr_result_path)
    with open(ilqr_result_path + '/settings', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # each trained model will perform 10 random tasks
    random_task_id = np.random.choice(len(possible_tasks), size=10)
    all_task_configs = []
    if env_name in ['planar', 'pendulum', 'pendulum_gym', 'mountain_car']:
        x_dim = np.prod(x_dim)
    for task_counter in range(len(random_task_id)):
        # pick a random task
        random_task = possible_tasks[random_task_id[task_counter]]
        # config for this task
        with open(config_path[random_task]) as f:
            config = json.load(f)

        # sample random start and goal state
        s_start_min, s_start_max = config['start_min'], config['start_max']
        config['s_start'] = np.random.uniform(low=s_start_min, high=s_start_max)
        s_goal = config['goal'][np.random.choice(len(config['goal']))]
        config['s_goal'] = np.array(s_goal)

        all_task_configs.append(config)

    # the folder where all trained models are saved
    folder = 'result/' + env_name
    log_folders = [os.path.join(folder, dI) for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder, dI))]
    log_folders.sort()

    # statistics on all trained models
    avg_model_percent = 0.0
    best_model_percent = 0.0
    for log in log_folders:
        with open(log + '/settings', 'r') as f:
            settings = json.load(f)
            armotized = settings['armotized']

        log_base = os.path.basename(os.path.normpath(log))
        model_path = ilqr_result_path + '/' + log_base
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        print('iLQR for ' + log_base)

        # load the trained model
        model = PCC(armotized, x_dim, z_dim, u_dim, env_name)
        model.load_state_dict(torch.load(log + '/model_' + str(epoch), map_location='cpu'))
        model.eval()
        dynamics = model.dynamics
        encoder = model.encoder

        # run the task with 10 different start and goal states for a particular model
        avg_percent = 0.0
        for task_counter, config in enumerate(all_task_configs):

            print('Performing task %d: ' %(task_counter) + str(config['task']))

            # environment specification
            horizon = config['horizon_prob']
            plan_len = config['plan_len']

            # ilqr specification
            R_z = config['q_weight'] * np.eye(z_dim)
            R_u = config['r_weight'] * np.eye(u_dim)
            num_uniform = config['uniform_trajs']
            num_extreme = config['extreme_trajs']
            ilqr_iters = config['ilqr_iters']
            inv_regulator_init = config['pinv_init']
            inv_regulator_multi = config['pinv_mult']
            inv_regulator_max = config['pinv_max']
            alpha_init = config['alpha_init']
            alpha_mult = config['alpha_mult']
            alpha_min = config['alpha_min']

            s_start = config['s_start']
            s_goal = config['s_goal']

            # mdp
            if env_name == 'planar':
                mdp = PlanarObstaclesMDP(goal=s_goal, goal_thres=config['distance_thresh'],
                                         noise=noise)
            elif env_name == 'pendulum':
                mdp = PendulumMDP(frequency=config['frequency'],
                                              noise=noise, torque=config['torque'])
            elif env_name == 'pendulum_gym':
                mdp = PendulumGymMDP(noise=config['noise'])
            elif env_name == 'cartpole':
                mdp = CartPoleMDP(frequency=config['frequency'], noise=noise)
            elif env_name == 'mountain_car':
                mdp = MountainCarMDP(noise=noise)
            # get z_start and z_goal
            x_start = get_x_data(mdp, s_start, config)
            x_goal = get_x_data(mdp, s_goal, config)
            with torch.no_grad():
                z_start = encoder(x_start)
                z_goal = encoder(x_goal)
            z_start = z_start.squeeze().numpy()
            z_goal = z_goal.squeeze().numpy()

            # initialize actions trajectories
            all_actions_trajs = random_actions_trajs(mdp, num_uniform, num_extreme, plan_len)
            actions_final = []

            # perform reciding horizon iLQR
            s_start_horizon = np.copy(s_start)  # s_start and z_start is changed at each horizon
            z_start_horizon = np.copy(z_start)
            for plan_iter in range(1, horizon + 1):
                latent_cost_list = [None] * len(all_actions_trajs)
                # iterate over all trajectories
                for traj_id in range(len(all_actions_trajs)):
                    # initialize the inverse regulator
                    inv_regulator = inv_regulator_init
                    for iter in range(1, ilqr_iters + 1):
                        u_seq = all_actions_trajs[traj_id]
                        z_seq = compute_latent_traj(z_start_horizon, u_seq, dynamics)
                        # compute the linearization matrices
                        A_seq, B_seq = seq_jacobian(dynamics, z_seq, u_seq)
                        # run backward
                        k_small, K_big = backward(R_z, R_u, z_seq, u_seq,
                                                  z_goal, A_seq, B_seq, inv_regulator)
                        current_cost = latent_cost(R_z, R_u, z_seq, z_goal, u_seq)
                        # forward using line search
                        alpha = alpha_init
                        accept = False  # if any alpha is accepted
                        while alpha > alpha_min:
                            z_seq_cand, u_seq_cand = forward(z_seq, all_actions_trajs[traj_id], k_small, K_big, dynamics, alpha)
                            # u_seq_cand = forward(all_actions_trajs[traj_id], k_small, K_big, A_seq, B_seq, alpha)
                            # z_seq_cand = compute_latent_traj(s_start, u_seq_cand, env_name, mdp, dynamics, encoder)
                            # cost_cand = latent_cost(R_z, R_u, z_seq_cand, z_goal, u_seq_cand)
                            cost_cand = latent_cost(R_z, R_u, z_seq_cand, z_goal, u_seq_cand)
                            if cost_cand < current_cost:  # accept the trajectory candidate
                                accept = True
                                all_actions_trajs[traj_id] = u_seq_cand
                                latent_cost_list[traj_id] = cost_cand
                                break
                            else:
                                alpha *= alpha_mult
                        if accept:
                            inv_regulator = inv_regulator_init
                        else:
                            inv_regulator *= inv_regulator_multi
                        if inv_regulator > inv_regulator_max:
                            break

                for i in range(len(latent_cost_list)):
                    if latent_cost_list[i] == None:
                        latent_cost_list[i] = np.inf
                traj_opt_id = np.argmin(latent_cost_list)
                action_chosen = all_actions_trajs[traj_opt_id][0]
                actions_final.append(action_chosen)
                s_start_horizon, z_start_horizon = update_horizon_start(mdp, s_start_horizon,
                                                                        action_chosen, encoder, config)
                # print ('location: ' + str(s_start_horizon))
                # print ('action_chosen: ' + str(action_chosen))
                # if mdp.is_fail(s_start_horizon):
                #     break
                all_actions_trajs = refresh_actions_trajs(all_actions_trajs, traj_opt_id, mdp,
                                                          np.min([plan_len, horizon - plan_iter]),
                                                          num_uniform, num_extreme)

            obs_traj, goal_counter = traj_opt_actions(s_start, actions_final, mdp)
            # compute the percentage close to goal
            success_rate = goal_counter / horizon
            print ('Success rate: %.2f' % (success_rate))
            percent = success_rate
            avg_percent += success_rate
            with open(model_path + '/result.txt', 'a+') as f:
                f.write(config['task'] + ': ' + str(percent) + '\n')

            # save trajectory as gif file
            gif_path = model_path + '/task_{:01d}.gif'.format(task_counter + 1)
            save_traj(obs_traj, mdp.render(s_goal).squeeze(), gif_path, config['task'])

        avg_percent = avg_percent / 10
        print ('Average success rate: ' + str(avg_percent))
        print ("====================================")
        avg_model_percent += avg_percent
        if avg_percent > best_model_percent:
            best_model = log_base
            best_model_percent = avg_percent
        with open(model_path + '/result.txt', 'a+') as f:
            f.write('Average percentage: ' + str(avg_percent))

    avg_model_percent = avg_model_percent / len(log_folders)
    with open('iLQR_result/' + env_name + '/result.txt', 'w') as f:
        f.write('Average percentage of all models: ' + str(avg_model_percent) + '\n')
        f.write('Best model: ' + best_model + ', best percentage: ' + str(best_model_percent))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run iLQR')
    parser.add_argument('--env', required=True, type=str, help='environment to perform')
    parser.add_argument('--noise', required=True, type=float, default=0.0, help='noise level for mdp')
    parser.add_argument('--epoch', required=True, type=str, help='number of epochs to load model')
    args = parser.parse_args()

    main(args)
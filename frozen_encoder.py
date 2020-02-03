from tensorboardX import SummaryWriter
import torch
from torch import nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import random
import argparse
import json
import time

from pcc_model import PCC
from datasets import *
from losses import *
from networks import MultivariateNormalDiag

torch.set_default_dtype(torch.float64)

device = torch.device("cuda")

datasets = {'planar': PlanarDataset, 'pendulum': PendulumDataset, 'cartpole': CartPoleDataset,
			'pendulum_gym': PendulumGymDataset, 'mountain_car': MountainCarDataset, 'threepole': ThreePoleDataset}
dims = {'planar': (1600, 2, 2), 'pendulum': (4608, 3, 1), 'cartpole': ((2, 80, 80), 8, 1),
		'pendulum_gym': (4608, 3, 1), 'mountain_car': (4800, 3, 1), 'threepole': ((2, 80, 80), 8, 3)}

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def weights_init(m):
	if isinstance(m, nn.Linear):
		init.kaiming_uniform_(m.weight, a=math.sqrt(5))
		if m.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(m.bias, -bound, bound)

def compute_loss(model, armotized, u, z_enc, z_next_trans_dist, z_next_enc, option, lam, delta=0.1):
	"""
	option: cpc or consistency: retrain dynamics model using cpc loss or consistency
	"""
	# nce and consistency loss
	# nce_loss = nce_future(z_next_trans_dist, z_next_enc) # sampling future
	nce_loss = nce_past(z_next_trans_dist, z_next_enc)  # sampling past

	consis_loss = - torch.mean(z_next_trans_dist.log_prob(z_next_enc))

	# curvature loss
	cur_loss = curvature(model, z_enc, u, delta, armotized)
	# cur_loss = new_curvature(model, z_enc, u)

	# additional norm loss to center z range to (0,0)
	norm_loss = torch.sum(torch.mean(z_enc, dim=0).pow(2))

	# additional norm loss to avoid collapsing
	avg_norm_2 = torch.mean(torch.sum(z_enc.pow(2), dim=1))

	if option == 'cpc':
		loss = lam[0] * nce_loss + lam[-1] * cur_loss
	elif option == 'consistency':
		loss = lam[1] * consis_loss + lam[-1] * cur_loss
	return nce_loss, consis_loss, cur_loss, norm_loss, avg_norm_2, loss, 


def train(model, option, train_loader, lam, latent_noise, optimizer, armotized, epoch):
	avg_nce_loss = 0.0
	avg_consis_loss = 0.0
	avg_cur_loss = 0.0
	avg_norm_loss = 0.0
	avg_norm_2_loss = 0.0
	avg_loss = 0.0

	num_batches = len(train_loader)
	model.train()

	start = time.time()

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

		nce_loss, consis_loss, cur_loss, norm_loss, norm_2, loss = compute_loss(model, armotized, u,
						z_enc, z_next_trans_dist, z_next_enc, option, lam=lam)

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
		print('Training time: %f' % (time.time() - start))
		print('--------------------------------------')

	return avg_nce_loss, avg_consis_loss, avg_cur_loss, avg_loss


def main(args):
	env_name = args.env
	assert env_name in ['planar', 'pendulum', 'pendulum_gym',
		'cartpole', 'mountain_car', 'threepole', 'reacher']
	option = args.option
	assert option in ['cpc', 'consistency']
	load_dir = args.load_dir
	epoch_load = args.epoch_load
	save_dir = args.save_dir
	epoches = args.num_iter
	iter_save = args.iter_save

	with open(load_dir + '/settings', 'r') as f:
		settings = json.load(f)
		armotized = settings['armotized']
		seed = settings['seed']
		data_size = settings['data_size']
		noise_level = settings['noise']
		batch_size = settings['batch_size']
		lam_nce = settings['lam_nce']
		lam_c = settings['lam_c']
		lam_cur = settings['lam_cur']
		lam = [lam_nce, lam_c, lam_cur]
		lr = settings['lr']
		latent_noise = settings['latent_noise']
		weight_decay = settings['decay']

	seed_torch(seed)

	dataset = datasets[env_name]
	data = dataset(sample_size=data_size, noise=noise_level)
	data_loader = DataLoader(data, batch_size=batch_size,
							 shuffle=True, drop_last=False, num_workers=4)

	x_dim, z_dim, u_dim = dims[env_name]
	model = PCC(armotized=armotized, x_dim=x_dim, z_dim=z_dim,
				u_dim=u_dim, env=env_name).to(device)
	model.load_state_dict(torch.load(load_dir + '/model_' + str(epoch_load)))

	# frozen the encoder
	for param in model.encoder.parameters():
		param.requires_grad = False
	# re-initialize and train the dynamics only
	model.dynamics.net_hidden.apply(weights_init)
	model.dynamics.net_mean.apply(weights_init)
	model.dynamics.net_logstd.apply(weights_init)

	optimizer = optim.Adam(model.dynamics.parameters(), betas=(
		0.9, 0.999), eps=1e-8, lr=lr, weight_decay=weight_decay)

	save_path = 'logs/' + env_name + '/' + save_dir
	if not path.exists(save_path):
		os.makedirs(save_path)
	
	writer = SummaryWriter(save_path)

	result_path = 'result/' + env_name + '/' + save_dir
	if not path.exists(result_path):
		os.makedirs(result_path)
	with open(result_path + '/settings', 'w') as f:
		json.dump(args.__dict__, f, indent=2)

	start = time.time()
	for i in range(epoches):
		avg_pred_loss, avg_consis_loss, avg_cur_loss, avg_loss = train(model, option, data_loader,
																lam, latent_noise, optimizer, armotized, i)
		# ...log the running loss
		writer.add_scalar('NCE loss', avg_pred_loss, i)
		writer.add_scalar('consistency loss', avg_consis_loss, i)
		writer.add_scalar('curvature loss', avg_cur_loss, i)
		writer.add_scalar('training loss', avg_loss, i)

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
	end = time.time()
	print ('time: ' + str(end - start))
	with open(result_path + '/time', 'w') as f:
		f.write(str(end - start))
	writer.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='retrain the dynamics')

	parser.add_argument('--env', required=True, type=str, help='environment used for training')
	parser.add_argument('--option', required=True, type=str, help='option for re-training dynamics')
	parser.add_argument('--load_dir', required=True, type=str, help='path to load the trained model')
	parser.add_argument('--epoch_load', default=2000, type=int, help='epoch to load')
	parser.add_argument('--save_dir', required=True, type=str, help='path to save retrined model')
	parser.add_argument('--num_iter', default=2000, type=int, help='number of epoches')
	parser.add_argument('--iter_save', default=1000, type=int, help='save model and result after this number of iterations')
	args = parser.parse_args()

	main(args)

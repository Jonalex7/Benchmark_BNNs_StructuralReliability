import os
from datetime import datetime
import pickle
import argparse
import torch.utils.data as data
import numpy as np
import torch

from limit_states import REGISTRY as ls_REGISTRY
from methods.sghmc import BNN_SGHMC
from utils.data import get_dataloader, isoprob_transform
from active_training.active_train import ActiveTrain
from config.defaults_sghmc import reliability_config_dict, model_config_dict

parser = argparse.ArgumentParser(description='Active train BNN with Stochastic Gradient HMC')

parser.add_argument('--res_dir', type=str, nargs='?', action='store', default='results/SGHMC_results',
                    help='Where to save predicted Pf results. Default: \'results/SGHMC_results\'.')
parser.add_argument('--res_file', type=str, nargs='?', action='store', default='SGHMC',
                    help='Pf results file name. Default: \'SGHMC\'.')
args = parser.parse_args()

# Directory to save results
results_dir = args.res_dir
results_file = args.res_file
date_time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Seed definition
if model_config_dict['seed'] is not None:
    seed_experiment = model_config_dict['seed']
else:
    seed_experiment = np.random.randint(0, 2**30 - 1)
np.random.seed(seed_experiment)
torch.manual_seed(seed_experiment)

lstate = ls_REGISTRY[reliability_config_dict['limit_state']]()
act_train = ActiveTrain()
print('Target limit state: ', reliability_config_dict['limit_state'])
mcs_samples = int(reliability_config_dict['mcs_samples'])
pf, beta, _, y_mc_test = lstate.monte_carlo_estimate(mcs_samples)
y_max = np.max(y_mc_test)   #to normalise the output for training
print('ref. PF:', pf, 'B:',beta)

#Passive training
passive_samples = model_config_dict['passive_samples'] 
x_norm, x_scaled, y_scaled = lstate.get_doe_points(n_samples=passive_samples, method='lhs')

#network config
width, layers = model_config_dict['network_architecture'][0], model_config_dict['network_architecture'][1]
learning_rate = model_config_dict['lr']
split_train_test = model_config_dict['split_train_test']
verbose = model_config_dict['verbose'] 

# Active training
use_cuda = torch.cuda.is_available()
n_active_ep = reliability_config_dict['active_epochs']
active_points = reliability_config_dict['active_samples']
training_epochs = model_config_dict['training_epochs']
batch_size = model_config_dict['batch_size']

burn_in = model_config_dict['burn_in']   #How many epochs to burn in for?. Default: 20.
sim_steps = model_config_dict['sim_steps']   #How many epochs pass between saving samples. Default: 2.
N_saves = model_config_dict['N_saved_models'] 
resample_its = 50
resample_prior_its = 15
re_burn = 1e8

results_dict = {}

for act_ep in range(n_active_ep):

    x_train = torch.tensor(x_norm, dtype=torch.float32).view(-1, lstate.input_dim) 
    y_train = torch.tensor(y_scaled/y_max, dtype=torch.float32).view(-1, lstate.output_dim)   #normalised output
    # y_train = torch.tensor(y_scaled, dtype=torch.float32).view(-1, lstate.output_dim)
    
    print('Samples: ', x_train.shape[0], end=" ")
    
    train_loader, _ = get_dataloader(x_train, y_train, lstate.input_dim, lstate.output_dim, train_test_split=split_train_test, batch_size=batch_size)

    net = BNN_SGHMC(N_train=len(x_train), input_dim=lstate.input_dim, width=width, depth=layers, output_dim=lstate.output_dim, 
                lr=learning_rate, cuda=use_cuda, grad_std_mul=10)

    net.train(train_loader, epoch=training_epochs, burn_in=burn_in, re_burn = re_burn , 
                resample_its=resample_its, resample_prior_its = resample_prior_its, 
                sim_steps = sim_steps, N_saves=N_saves, verbose=verbose)
        
    Pf_ref, B_ref, x_mc_norm, _ = lstate.monte_carlo_estimate(mcs_samples)
    X_uq = torch.tensor(x_mc_norm, dtype=torch.float32)

    print('pf_ref', Pf_ref, end=" ")

    y_bnn_pred = net.sample_predict(X_uq, Nsamples=N_saves)
    y_bnn_mean = torch.mean(y_bnn_pred, 0)
    # y_bnn_std = torch.std(y_bnn_pred, 0)
    pf_sumo = (((y_bnn_mean<0).sum()) / torch.tensor(mcs_samples)).item()

    print('pf_surrogate ' , pf_sumo)
    results_dict['pf_'+ str(len(x_train))] = pf_sumo #, B_sumo

    x_norm = act_train.get_active_points(x_train, X_uq, y_bnn_pred.view(-1, N_saves), active_points)
    x_scaled = isoprob_transform(x_norm, lstate.marginals)
    y_scaled = lstate.eval_lstate(x_scaled)

results_dict[str(len(x_train)) + '_doepoints'] = x_train, y_train
# Storing seed for reproducibility
results_dict['seed'] = seed_experiment

with open(results_dir + '/' + results_file + "_" + date_time_stamp + ".pkl", 'wb') as file_id:
    pickle.dump(results_dict, file_id)

print('End training')
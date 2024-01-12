import os
import numpy as np
import torch
from datetime import datetime
import pickle
import argparse
# from scipy.stats import norm, uniform, lognorm
# from scipy.stats import qmc 

from limit_states import REGISTRY as ls_REGISTRY
from methods.ensemble import Ensemble
from utils.data import get_dataloader, isoprob_transform
from active_training.active_train import ActiveTrain
from config.defaults_ensembles import reliability_config_dict, model_config_dict

parser = argparse.ArgumentParser(description='Active train Ensembles BNN')

parser.add_argument('--res_dir', type=str, nargs='?', action='store', default='results/ensembles_results',
                    help='Where to save predicted Pf results. Default: \'results/ensembles_results\'.')
parser.add_argument('--res_file', type=str, nargs='?', action='store', default='ensembles',
                    help='Pf results file name. Default: \'ensembles\'.')
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

# Passive training
passive_samples = model_config_dict['passive_samples'] 
x_norm, x_scaled, y_scaled = lstate.get_doe_points(n_samples=passive_samples, method='lhs')

# Neural net config
width, layers = model_config_dict['network_architecture'][0], model_config_dict['network_architecture'][1]
n_ensembles = model_config_dict['n_ensembles']
learning_rate = model_config_dict['lr'] 
batch_size = model_config_dict['batch_size']
split_train_test = model_config_dict['split_train_test']
verbose = model_config_dict['verbose']

net = Ensemble(lstate.input_dim, width, lstate.output_dim, n_ensembles)

# Active training
use_cuda = torch.cuda.is_available()  #not used
training_epochs = model_config_dict['training_epochs']
n_active_ep = reliability_config_dict['active_epochs']
active_points = reliability_config_dict['active_samples']

results_dict = {}

for ep in range(n_active_ep):

    x_train = torch.tensor(x_norm, dtype=torch.float32).view(-1, lstate.input_dim) 
    y_train = torch.tensor(y_scaled, dtype=torch.float32).view(-1, lstate.output_dim) 
    # y_train = torch.tensor(y_scaled/y_max, dtype=torch.float32).view(-1, lstate.output_dim)   #normalised output
    
    print('Samples: ', x_train.shape[0], end=" ")

    train_loader, _ = get_dataloader(x_train, y_train, lstate.input_dim, lstate.output_dim, split_train_test, batch_size)

    net.train_ensemble(train_loader, training_epochs, verbose)

    Pf_ref, B_ref, x_mc_norm, _ = lstate.monte_carlo_estimate(mcs_samples)
    print('pf ', Pf_ref)
    X_uq = torch.tensor(x_mc_norm, dtype=torch.float32)

    Y_uq = net.predictive_uq(X_uq)
    Y_mean = Y_uq.mean(dim=1)
    pf_sumo = (((Y_mean<0).sum()) /torch.tensor(mcs_samples)).item()
    print('pf_surrogate ' , pf_sumo)
    results_dict['pf_'+ str(len(x_train))] = pf_sumo #, B_sumo
    
    x_norm = act_train.get_active_points(x_train, X_uq, Y_uq, active_points)
    x_scaled = isoprob_transform(x_norm, lstate.marginals)
    y_scaled = lstate.eval_lstate(x_scaled)

results_dict[str(len(x_train)) + '_doepoints'] = x_train, y_train
# Storing seed for reproducibility
results_dict['seed'] = seed_experiment

with open(results_dir + '/' + results_file + "_" + date_time_stamp + ".pkl", 'wb') as file_id:
    pickle.dump(results_dict, file_id)

print('End training')
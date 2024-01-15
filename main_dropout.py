import os
import numpy as np
import torch
from datetime import datetime
import pickle
import argparse

from limit_states import REGISTRY as ls_REGISTRY
from methods.mc_droput import NeuralNetworkWithDropout
from utils.data import get_dataloader, isoprob_transform
from active_training.active_train import ActiveTrain
from config.defaults_dropout import reliability_config_dict, model_config_dict

parser = argparse.ArgumentParser(description='Active train Dropout NN')

parser.add_argument('--res_dir', type=str, nargs='?', action='store', default='results/dropout_results',
                    help='Where to save predicted Pf results. Default: \'results/dropout_results\'.')
parser.add_argument('--res_file', type=str, nargs='?', action='store', default='dropout',
                    help='Pf results file name. Default: \'dropout\'.')
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

# Loading
lstate = ls_REGISTRY[reliability_config_dict['limit_state']]()
act_train = ActiveTrain()
print('Target limit state: ', reliability_config_dict['limit_state'])
mcs_samples = int(reliability_config_dict['mcs_samples'])
pf, beta, _, y_mc_test = lstate.monte_carlo_estimate(mcs_samples)
print('ref. PF:', pf, 'B:',beta)

# Passive training
passive_samples = model_config_dict['passive_samples'] 
X_doe, X_scaled, Y_doe = lstate.get_doe_points(5)
X = torch.tensor(X_doe, dtype=torch.float32)
Y = torch.tensor(Y_doe, dtype=torch.float32)

# Neural net config
net = NeuralNetworkWithDropout(2, 100, 2, 1, 0.1)

# Active training
n_train_ep = 20
active_points = 10
mcs_samples = int(1e5)
for ep in range(n_train_ep):
    
    print('Samples: ', X.shape[0])
    train_loader, test_loader = get_dataloader(X, Y, lstate.input_dim, lstate.output_dim, 1.0, 16)

    #net = NeuralNetworkWithDropout(2, 100, 1, 0.3)
    net.train(train_loader, 1000, 1e-4)

    pf_mc, _, X_mc, Y_mc = lstate.monte_carlo_estimate(mcs_samples)
    print('pf ', pf_mc)
    X_uq = torch.tensor(X_mc, dtype=torch.float32)

    Y_uq = net.predictive_uq(X_uq, 100)
    Y_mean = Y_uq.mean(dim=1)
    print('pf_surrogate ' , (((Y_mean<0).sum()) /torch.tensor(mcs_samples)).item() )
    
    X_uq = torch.tensor(X_uq, dtype=torch.float32)
    X_ = act_train.get_active_points(X, X_uq, Y_uq, active_points)

    Y_ = lstate.eval_lstate(X_)
    Y_ = torch.tensor(Y_, dtype=torch.float32)

    X = X_
    Y = Y_

print('End training')
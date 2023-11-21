import numpy as np
from limit_states.g11d_electric import g11d_electric
from methods.sghmc import *
from utils.data import get_dataloader
from active_training.active_train import ActiveTrain
from tqdm import tqdm

import torch.utils.data as data

lstate = g11d_electric()
act_train = ActiveTrain()
pf, beta, _,_, y_mc_test = lstate.monte_carlo_estimate(1e6)
y_max = np.max(y_mc_test)   #to normalise the output for training
print('ref PF:', pf, 'B:',beta)

#Passive training
passive_samples = 100 
x_norm, x_scaled, y_scaled = lstate.get_doe_points(n_samples=passive_samples, method='lhs')
batch_size = 64

#network config
width, layers = 20, 2

# Active training
use_cuda = torch.cuda.is_available()
n_active_ep = 20
active_points = 5
n_passive_ep = 100
mcs_samples = int(1e6)

burn_in = 20   #How many epochs to burn in for?. Default: 20.
sim_steps = 2   #How many epochs pass between saving samples. Default: 2.
N_saves=10
resample_its = 50
resample_prior_its = 15
re_burn = 1e8

for act_ep in range(n_active_ep):

    x_train = torch.tensor(x_norm, dtype=torch.float32).view(-1,11)
    y_train = torch.tensor(y_scaled/y_max, dtype=torch.float32).view(-1,1)  #normalised output
    
    print('Samples: ', x_train.shape[0], end=" ")
    
    train_loader, _ = get_dataloader(x_train, y_train, lstate.input_dim, lstate.output_dim, train_test_split=1.0, batch_size=batch_size)

    net = BNN_SGHMC(N_train=len(x_train), input_dim=lstate.input_dim, width=width, depth=layers, output_dim=lstate.output_dim, 
                lr=1e-2, cuda=use_cuda, grad_std_mul=10)

    net.train(train_loader, epoch=n_passive_ep, burn_in=burn_in, re_burn = re_burn , 
                resample_its=resample_its, resample_prior_its = resample_prior_its, 
                sim_steps = sim_steps, N_saves=N_saves, verbose=0)
        
    Pf_ref, B_ref, x_mc_norm, x_mc_scaled, _ = lstate.monte_carlo_estimate(mcs_samples)
    X_uq = torch.tensor(x_mc_norm, dtype=torch.float32)

    print('pf_ref', Pf_ref, end=" ")

    y_bnn_pred = net.sample_predict(X_uq, Nsamples=N_saves)
    y_bnn_mean = torch.mean(y_bnn_pred, 0)
    y_bnn_std = torch.std(y_bnn_pred, 0)

    print('pf_surrogate ' , (((y_bnn_mean<0).sum()) / torch.tensor(mcs_samples)).item() )

    x_norm = act_train.get_active_points(x_train, X_uq, y_bnn_pred.view(-1, N_saves), active_points)
    x_scaled = lstate.isoprob_transform(x_norm, lstate.marginals)
    y_scaled = lstate.eval_lstate(x_scaled)

print('End training')
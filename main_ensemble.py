import numpy as np
from scipy.stats import norm, uniform, lognorm
from scipy.stats import qmc 
import torch
from methods.ensemble import Ensemble
from utils.data import get_dataloader
from limit_states.g2d_four_branch import g2D_four_branch
from active_training.active_train import ActiveTrain

# Loading
lstate = g2D_four_branch()
act_train = ActiveTrain()
beta, pf, _, _ = lstate.monte_carlo_estimate(1e6)
print('ref: values', beta, pf)

# Passive training
X_doe, Y_doe = lstate.get_doe_points(5)
X = torch.tensor(X_doe, dtype=torch.float32)
Y = torch.tensor(Y_doe, dtype=torch.float32)

# Neural net config
net = Ensemble(2, 100, 1, 4)

# Active training
n_train_ep = 1
active_points = 5
mcs_samples = int(1e5)
for ep in range(n_train_ep):
    
    print('Samples: ', X.shape[0])
    train_loader, test_loader = get_dataloader(X, Y, lstate.input_dim, lstate.output_dim, 1.0, 16)

    net.train_ensemble(train_loader, 1000, 1)

    pf_mc, _, X_mc, Y_mc = lstate.monte_carlo_estimate(mcs_samples)
    print('pf ', pf_mc)
    X_uq = torch.tensor(X_mc, dtype=torch.float32)

    Y_uq = net.predictive_uq(X_uq)
    Y_mean = Y_uq.mean(dim=1)
    print('pf_surrogate ' , (((Y_mean<0).sum()) /torch.tensor(mcs_samples)).item() )
    
    X_uq = torch.tensor(X_uq, dtype=torch.float32)
    X_ = act_train.get_active_points(X, X_uq, Y_uq, active_points)

    Y_ = lstate.eval_lstate(X_)
    Y_ = torch.tensor(Y_, dtype=torch.float32)

    X = X_
    Y = Y_

print('End training')
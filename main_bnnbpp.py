from methods.bnn_bpp import *
from limit_states.g2d_four_branch import g2D_four_branch
from utils.data import get_dataloader
from active_training.active_train import ActiveTrain
import numpy as np

# Loading
lstate = g2D_four_branch()
act_train = ActiveTrain()
beta, pf, _, _ = lstate.monte_carlo_estimate(1e6)
print('ref: values', beta, pf)

# Passive training
x_train, y_train = lstate.get_doe_points(5)   #with n_exponent for sobol from this LState

# Neural net config. 
bnn_bpp = BNN_BPP(input_size=2, hidden_sizes=30, output_size=1)

# Active training
n_train_ep = 30
active_points = 5
mcs_samples = int(1e5)
passive_epochs = 5000
batch_size = 16
n_sim = 100
# KL_scale = 20

for ep in range(n_train_ep):

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    
    print('Samples: ', len(x_train), end=" ")
    train_loader, test_loader = get_dataloader(x_train, y_train, lstate.input_dim, lstate.output_dim, 1.0, batch_size)

    bnn_bpp.train(train_loader, num_epochs=passive_epochs, lr=1e-2, verbose=0, kl_scale=len(x_train))

    pf_mc, _, X_mc, Y_mc = lstate.monte_carlo_estimate(mcs_samples)
    X_uq = torch.tensor(X_mc, dtype=torch.float32)

    print('pf_ref', pf_mc, end=" ")
    
    y_bnn = bnn_bpp.predictive_uq(X_uq, n_sim)
    y_mean = y_bnn.mean(dim=1)
    y_std = y_bnn.mean(dim=1)

    print('pf_surrogate ' , (((y_mean<0).sum()) / torch.tensor(mcs_samples)).item() )

    x_ = act_train.get_active_points(x_train, X_uq, y_bnn, active_points)
    y_ = lstate.eval_lstate(x_)

    x_train = x_
    y_train = y_

print('End training')
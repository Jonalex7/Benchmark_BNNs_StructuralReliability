import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import numpy as np

from limit_states.g2d_four_branch import g2D_four_branch

# Define the Bayesian Neural Network model
class BayesianNN(pyro.nn.PyroModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the model with priors
def model(x_data, y_data):
    input_size = x_data.shape[-1]
    output_size = 1  # Assuming a regression problem, change as needed
    hidden_size = 40

    # Priors
    fc1_weights_prior = dist.Normal(0, 1).expand([hidden_size, input_size]).to_event(2)
    fc1_bias_prior = dist.Normal(0, 1).expand([hidden_size]).to_event(1)

    fc2_weights_prior = dist.Normal(0, 1).expand([hidden_size, hidden_size]).to_event(2)
    fc2_bias_prior = dist.Normal(0, 1).expand([hidden_size]).to_event(1)

    fc3_weights_prior = dist.Normal(0, 1).expand([output_size, hidden_size]).to_event(2)
    fc3_bias_prior = dist.Normal(0, 1).expand([output_size]).to_event(1)

    priors = {
        'fc1.weight': fc1_weights_prior, 'fc1.bias': fc1_bias_prior,
        'fc2.weight': fc2_weights_prior, 'fc2.bias': fc2_bias_prior,
        'fc3.weight': fc3_weights_prior, 'fc3.bias': fc3_bias_prior
    }

    # Lift the module to a random variable
    lifted_module = pyro.random_module("module", BayesianNN(input_size, hidden_size, output_size), priors)
    lifted_reg_model = lifted_module()

    with pyro.plate("data", x_data.shape[0]):
        # Forward pass through the network
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # Likelihood
        pyro.sample("obs", dist.Normal(prediction_mean, 0.1), obs=y_data)

# Define the guide for HMC
def guide(x_data, y_data):
    input_size = x_data.shape[-1]
    output_size = 1  # Assuming a regression problem, change as needed
    hidden_size = 40

    # Define the guide parameters
    fc1_weights_loc = pyro.param('fc1_weights_loc', torch.randn(hidden_size, input_size))
    fc1_weights_scale = pyro.param('fc1_weights_scale', torch.ones(hidden_size, input_size), constraint=dist.constraints.positive)

    fc1_bias_loc = pyro.param('fc1_bias_loc', torch.randn(hidden_size))
    fc1_bias_scale = pyro.param('fc1_bias_scale', torch.ones(hidden_size), constraint=dist.constraints.positive)

    fc2_weights_loc = pyro.param('fc2_weights_loc', torch.randn(hidden_size, hidden_size))
    fc2_weights_scale = pyro.param('fc2_weights_scale', torch.ones(hidden_size, hidden_size), constraint=dist.constraints.positive)

    fc2_bias_loc = pyro.param('fc2_bias_loc', torch.randn(hidden_size))
    fc2_bias_scale = pyro.param('fc2_bias_scale', torch.ones(hidden_size), constraint=dist.constraints.positive)

    fc3_weights_loc = pyro.param('fc3_weights_loc', torch.randn(output_size, hidden_size))
    fc3_weights_scale = pyro.param('fc3_weights_scale', torch.ones(output_size, hidden_size), constraint=dist.constraints.positive)

    fc3_bias_loc = pyro.param('fc3_bias_loc', torch.randn(output_size))
    fc3_bias_scale = pyro.param('fc3_bias_scale', torch.ones(output_size), constraint=dist.constraints.positive)

    # Guide distributions
    fc1_weights_prior = dist.Normal(fc1_weights_loc, fc1_weights_scale).to_event(2)
    fc1_bias_prior = dist.Normal(fc1_bias_loc, fc1_bias_scale).to_event(1)

    fc2_weights_prior = dist.Normal(fc2_weights_loc, fc2_weights_scale).to_event(2)
    fc2_bias_prior = dist.Normal(fc2_bias_loc, fc2_bias_scale).to_event(1)

    fc3_weights_prior = dist.Normal(fc3_weights_loc, fc3_weights_scale).to_event(2)
    fc3_bias_prior = dist.Normal(fc3_bias_loc, fc3_bias_scale).to_event(1)

    priors = {
        'fc1.weight': fc1_weights_prior, 'fc1.bias': fc1_bias_prior,
        'fc2.weight': fc2_weights_prior, 'fc2.bias': fc2_bias_prior,
        'fc3.weight': fc3_weights_prior, 'fc3.bias': fc3_bias_prior
    }

    lifted_module = pyro.random_module("module", BayesianNN(input_size, hidden_size, output_size), priors)
    return lifted_module()

lstate = g2D_four_branch()
beta, pf, _, _ = lstate.monte_carlo_estimate(1e6)
print('ref: values', beta, pf)

# Passive training
X_, Y_ = lstate.get_doe_points(5)

x_data = torch.tensor(X_, dtype=torch.float32)
y_data = torch.tensor(Y_, dtype=torch.float32)

# HMC
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=100)
mcmc.run(x_data, y_data)

# Extract posterior samples
posterior_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

# Print or analyze the posterior samples as needed
#print(posterior_samples)

mcs_samples = int(1e5)
forw_n = int(100)
pf_mc, _, X_mc, Y_mc = lstate.monte_carlo_estimate(mcs_samples)
print('pf ', pf_mc)
X_uq = torch.tensor(X_mc, dtype=torch.float32)

y_uq = np.zeros((mcs_samples, forw_n))
# Plot the posterior predictive distribution
for i in range(forw_n):
    sampled_model = BayesianNN(2, 40, 1)  # Adjust input size accordingly
    state_dict = {k.replace("module$$$", ""): torch.tensor(v[i]) for k, v in posterior_samples.items()}
    sampled_model.load_state_dict(state_dict)

    y_pred = sampled_model(X_uq).detach().numpy().squeeze()
    #print(y_pred.shape)
    y_uq[:, i] = y_pred

pf_pred = np.sum(y_uq.mean(axis=1) < 0) / mcs_samples
print(pf_pred)

y_uq_mean, y_uq_std = y_uq.mean(axis=1), y_uq.std(axis=1)
u_f = -np.abs(y_uq_mean)/y_uq_std

ind = np.argpartition(u_f, -5)[-5:]
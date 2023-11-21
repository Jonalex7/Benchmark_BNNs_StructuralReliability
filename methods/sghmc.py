from __future__ import division
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
import copy
from numpy.random import gamma
from torch.autograd import Variable
try:
    import cPickle as pickle
except:
    import pickle

'''Pytorch implementations of Bayes By SG-HMC from Bayesian-Neural-Networks (J. Antoran)'''

class MLP(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class BNN_SGHMC():  # for regression
    def __init__(self, N_train, input_dim=10, width=50, depth=2, output_dim=1, lr=1e-2, cuda=True, grad_std_mul=30):
        super(BNN_SGHMC, self).__init__()

        # cprint('y', 'BNN regression output')
        self.lr = lr
        self.model = MLP(input_dim=input_dim, width=width, depth=depth, output_dim=output_dim)
        self.cuda = cuda

        self.N_train = N_train
        self.create_net()
        self.create_opt()
        self.schedule = None  # [] #[50,200,400,600]
        self.epoch = 0

        self.grad_buff = []
        self.max_grad = 1e20
        self.grad_std_mul = grad_std_mul

        self.weight_set_samples = []

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)
        if self.cuda:
            self.model.cuda()

    def create_opt(self):
        """This optimiser incorporates the gaussian prior term automatically. The prior variance is gibbs sampled from
        its posterior using a gamma hyper-prior."""
        self.optimizer = H_SA_SGHMC(params=self.model.parameters(), lr=self.lr, base_C=0.05, gauss_sig=0.1)  # this last parameter does nothing

    def fit(self, x, y, burn_in=False, resample_momentum=False, resample_prior=False):
        self.set_mode_train(train=True)
        x, y = self.to_variable(var=(x, y), cuda=self.cuda)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = F.mse_loss(out, y, reduction='mean')
        loss = loss * self.N_train  # We use mean because we treat as an estimation of whole dataset
        loss.backward()

        # Gradient buffer to allow for dynamic clipping and prevent explosions
        if len(self.grad_buff) > 1000:
            self.max_grad = np.mean(self.grad_buff) + self.grad_std_mul * np.std(self.grad_buff)
            self.grad_buff.pop(0)
        # Clipping to prevent explosions
        self.grad_buff.append(nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                       max_norm=self.max_grad, norm_type=2))
        if self.grad_buff[-1] >= self.max_grad:
            print(self.max_grad, self.grad_buff[-1])
            self.grad_buff.pop()
        self.optimizer.step(burn_in=burn_in, resample_momentum=resample_momentum, resample_prior=resample_prior)

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data * x.shape[0] / self.N_train #, err
    
    def train(self, train_loader, epoch=50, burn_in=10, re_burn = 1e8 , 
              resample_its=50, resample_prior_its = 15, sim_steps = 2, N_saves=10, verbose=0):
        it_count = 0
        n_epochs = int(epoch)
        cost_train = np.zeros(n_epochs)

        for ep in range(0, n_epochs):

            self.set_mode_train(True)
            nb_samples = 0

            for x, y in train_loader:

                cost_pred = self.fit(x, y, burn_in=(ep % re_burn < burn_in), 
                            resample_momentum=(it_count % resample_its == 0),
                            resample_prior=(it_count % resample_prior_its == 0))
       
                it_count += 1
                cost_train[ep] += cost_pred
                nb_samples += len(x)

            cost_train[ep] /= nb_samples
            self.update_lr(ep)

            if not verbose == 0:
                print("it %d/%d, train_loss = %f," % (ep, n_epochs, cost_train[ep]))
        
            if ep % re_burn >= burn_in and ep % sim_steps == 0:
                self.save_sampled_net(max_samples=N_saves)

        print("train_loss = %f," % (cost_train[ep]), end=" ")

        return

    def eval(self, x, y, train=False):
        self.set_mode_train(train=False)
        x, y = self.to_variable(var=(x, y), cuda=self.cuda)
        out = self.model(x)
        loss = F.mse_loss(out, y, reduction='sum')
        return loss.data 

    def save_sampled_net(self, max_samples):

        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)

        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))

        # cprint('c', ' saving weight samples %d/%d' % (len(self.weight_set_samples), max_samples))
        return None

    def predict(self, x):
        self.set_mode_train(train=False)
        x, = self.to_variable(var=(x, ), cuda=self.cuda)
        out = self.model(x)
        return out.data 

    def sample_predict(self, x, Nsamples=0, grad=False):
        """return predictions using multiple samples from posterior"""
        self.set_mode_train(train=False)
        if Nsamples == 0:
            Nsamples = len(self.weight_set_samples)
        x, = self.to_variable(var=(x, ), cuda=self.cuda)

        if grad:
            self.optimizer.zero_grad()
            if not x.requires_grad:
                x.requires_grad = True

        out = x.data.new(Nsamples, x.shape[0], self.model.output_dim)

        # iterate over all saved weight configuration samples
        for idx, weight_dict in enumerate(self.weight_set_samples):
            if idx == Nsamples:
                break
            self.model.load_state_dict(weight_dict)
            # out[idx] = self.model(x)
            out[idx] = self.model(x)

        # out = out[:idx]
        # prob_out = F.softmax(out, dim=2)

        if grad:
            return out #prob_out
        else:
            return out.data #prob_out.data

    def get_weight_samples(self, Nsamples=0):
        """return weight samples from posterior in a single-column array"""
        weight_vec = []

        if Nsamples == 0 or Nsamples > len(self.weight_set_samples):
            Nsamples = len(self.weight_set_samples)

        for idx, state_dict in enumerate(self.weight_set_samples):
            if idx == Nsamples:
                break

            for key in state_dict.keys():
                if 'weight' in key:
                    weight_mtx = state_dict[key].cpu().data
                    for weight in weight_mtx.view(-1):
                        weight_vec.append(weight)

        return np.array(weight_vec)

    def save_weights(self, filename):
        self.save_object(self.weight_set_samples, filename)

    def load_weights(self, filename, subsample=1):
        self.weight_set_samples = self.load_object(filename)
        self.weight_set_samples = self.weight_set_samples[::subsample]
    
    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def to_variable(self, var=(), cuda=True, volatile=False):
        out = []
        for v in var:
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).type(torch.FloatTensor)

            if not v.is_cuda and cuda:
                v = v.cuda()

            if not isinstance(v, Variable):
                v = Variable(v, volatile=volatile)

            out.append(v)
        return out

    def save_object(self, obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % self.lr, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr


class H_SA_SGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses scale adaption during burn-in
        procedure to find some hyperparamters. A gaussian prior is placed over parameters and a Gamma
        Hyperprior is placed over the prior's standard deviation"""

    def __init__(self, params, lr=1e-2, base_C=0.05, gauss_sig=0.1, alpha0=10, beta0=10):

        self.eps = 1e-6
        self.alpha0 = alpha0
        self.beta0 = beta0

        if gauss_sig == 0:
            self.weight_decay = 0
        else:
            self.weight_decay = 1 / (gauss_sig ** 2)

        if self.weight_decay <= 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(self.weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if base_C < 0:
            raise ValueError("Invalid friction term: {}".format(base_C))

        defaults = dict(
            lr=lr,
            base_C=base_C,
        )
        super(H_SA_SGHMC, self).__init__(params, defaults)

    def step(self, burn_in=False, resample_momentum=False, resample_prior=False):
        """Simulate discretized Hamiltonian dynamics for one step"""
        loss = None

        for group in self.param_groups:  # iterate over blocks -> the ones defined in defaults. We dont use groups.
            for p in group["params"]:  # these are weight and bias matrices
                if p.grad is None:
                    continue
                state = self.state[p]  # define dict for each individual param
                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["V_hat"] = torch.ones_like(p)
                    state["v_momentum"] = torch.zeros_like(
                        p)  # p.data.new(p.data.size()).normal_(mean=0, std=np.sqrt(group["lr"])) #
                    state['weight_decay'] = self.weight_decay

                state["iteration"] += 1  # this is kind of useless now but lets keep it provisionally

                if resample_prior:
                    alpha = self.alpha0 + p.data.nelement() / 2
                    beta = self.beta0 + (p.data ** 2).sum().item() / 2
                    gamma_sample = gamma(shape=alpha, scale=1 / (beta), size=None)
                    #                     print('std', 1/np.sqrt(gamma_sample))
                    state['weight_decay'] = gamma_sample

                base_C, lr = group["base_C"], group["lr"]
                weight_decay = state["weight_decay"]
                tau, g, V_hat = state["tau"], state["g"], state["V_hat"]

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # update parameters during burn-in
                if burn_in:  # We update g first as it makes most sense
                    tau.add_(-tau * (g ** 2) / (
                                V_hat + self.eps) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    tau_inv = 1. / (tau + self.eps)
                    g.add_(-tau_inv * g + tau_inv * d_p)  # average gradient see Eq 9 in [1] right
                    V_hat.add_(-tau_inv * V_hat + tau_inv * (d_p ** 2))  # gradient variance see Eq 8 in [1]

                V_sqrt = torch.sqrt(V_hat)
                V_inv_sqrt = 1. / (V_sqrt + self.eps)  # preconditioner

                if resample_momentum:  # equivalent to var = M under momentum reparametrisation
                    state["v_momentum"] = torch.normal(mean=torch.zeros_like(d_p),
                                                       std=torch.sqrt((lr ** 2) * V_inv_sqrt))
                v_momentum = state["v_momentum"]

                noise_var = (2. * (lr ** 2) * V_inv_sqrt * base_C - (lr ** 4))
                noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))
                # sample random epsilon
                noise_sample = torch.normal(mean=torch.zeros_like(d_p), std=torch.ones_like(d_p) * noise_std)

                # update momentum (Eq 10 right in [1])
                v_momentum.add_(- (lr ** 2) * V_inv_sqrt * d_p - base_C * v_momentum + noise_sample)

                # update theta (Eq 10 left in [1])
                p.data.add_(v_momentum)

        return loss
    
